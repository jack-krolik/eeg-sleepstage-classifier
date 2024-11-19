import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging

class ImprovedFeatureExtractor(nn.Module):
    def __init__(self, input_size=3000):
        super(ImprovedFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 50), stride=1, padding=(0, 25))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 25), stride=1, padding=(0, 12))
        self.pool = nn.AdaptiveAvgPool2d((1, 64))
        self.fc = nn.Linear(64 * 64, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
    
    def forward(self, lstm_output):
        # Pre-allocate output tensor
        batch_size = lstm_output.size(0)
        device = lstm_output.device
        attention_outputs = torch.empty(batch_size, self.num_heads * lstm_output.size(-1),
                                     device=device)
        
        # Process each head
        for i, head in enumerate(self.attention_heads):
            # Compute attention weights
            attention_weights = F.softmax(head(lstm_output), dim=1)
            # Compute context vector efficiently
            start_idx = i * lstm_output.size(-1)
            end_idx = (i + 1) * lstm_output.size(-1)
            attention_outputs[:, start_idx:end_idx] = (attention_weights * lstm_output).sum(dim=1)
        
        return attention_outputs


class ImprovedSleepDetectorLSTM(nn.Module):
    def __init__(self, input_size=1029, hidden_size=256, n_layers=2, n_classes=5, num_heads=4, dropout=0.5):
        super(ImprovedSleepDetectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads)  # *2 for bidirectional
        self.fc1 = nn.Linear(hidden_size * 2 * num_heads, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        output = self.fc2(x)
        return output
    
    def _init_lstm_state(self, batch_size, device):
        """Initialize LSTM states"""
        num_directions = 2 if self.lstm.bidirectional else 1
        return (torch.zeros(self.lstm.num_layers * num_directions, batch_size, 
                        self.lstm.hidden_size, device=device),
                torch.zeros(self.lstm.num_layers * num_directions, batch_size, 
                        self.lstm.hidden_size, device=device))

    def _process_lstm_chunk(self, chunk, h=None):
        """Process a single LSTM chunk"""
        if h is None:
            h = self._init_lstm_state(chunk.size(0), chunk.device)
        elif isinstance(h, tuple):
            h = (h[0].detach(), h[1].detach())
        return self.lstm(chunk, h)


class ImprovedSleepDetectorCNN(nn.Module):
    def __init__(self, in_channels=4, n_filters=[32, 64, 128], kernel_sizes=[9, 5, 3]):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for filters, kernel_size in zip(n_filters, kernel_sizes):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(current_channels, filters, 
                         kernel_size=kernel_size, 
                         stride=1, 
                         padding=padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ])
            current_channels = filters
            
        self.cnn = nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        """Implementation of forward pass for checkpointing"""
        return self.cnn(x)
        
    def forward(self, x):
        return self._forward_impl(x)


class ImprovedSleepdetector(nn.Module):
    def __init__(self, n_filters=[32, 64, 96], lstm_hidden=64, lstm_layers=2, dropout=0.5):
        super().__init__()
        
        # Store configuration
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # CNN for feature extraction
        self.cnn = ImprovedSleepDetectorCNN(
            in_channels=4,
            n_filters=n_filters
        )
        
        # Projection layer
        self.project = nn.Sequential(
            nn.Linear(n_filters[-1] + 16, lstm_hidden),
            nn.ReLU()
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(lstm_hidden * 2, num_heads=2)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 5)
        )
        
        # Register buffers for statistics
        self.register_buffer('channel_medians', torch.zeros(4))
        self.register_buffer('channel_iqrs', torch.ones(4))
        
        # Configuration flags
        self.normalize_channels = True
        self.use_checkpointing = True
        
        # Processing chunks
        self.cnn_chunk_size = 1000
        self.lstm_chunk_size = 50
        
    def _init_lstm_state(self, batch_size, device):
        """Initialize LSTM hidden states"""
        num_directions = 2 if self.lstm.bidirectional else 1
        return (torch.zeros(self.lstm_layers * num_directions, batch_size, 
                          self.lstm_hidden, device=device),
               torch.zeros(self.lstm_layers * num_directions, batch_size, 
                          self.lstm_hidden, device=device))
    
    def _process_cnn_chunk(self, chunk):
        """Process a single CNN chunk with optional checkpointing"""
        if self.use_checkpointing:
            return checkpoint(
                self.cnn._forward_impl,  # Use internal implementation
                chunk,
                use_reentrant=False,
                preserve_rng_state=False
            )
        return self.cnn(chunk)
    
    def _process_lstm_chunk(self, chunk, h=None):
        """Process a single LSTM chunk"""
        if h is None:
            h = self._init_lstm_state(chunk.size(0), chunk.device)
        else:
            h = (h[0].detach(), h[1].detach())
        return self.lstm(chunk, h)
    
    def _normalize_channels(self, x):
        """Apply channel-wise normalization"""
        epsilon = 1e-8
        for i in range(x.shape[1]):
            x[:, i] = (x[:, i] - self.channel_medians[i]) / (self.channel_iqrs[i] + epsilon)
        return x
    
    def _normalize_spectral(self, spectral_features):
        """Normalize spectral features"""
        epsilon = 1e-8
        with torch.no_grad():
            mean = spectral_features.mean(dim=0, keepdim=True)
            std = spectral_features.std(dim=0, keepdim=True)
            std = torch.where(std == 0, torch.ones_like(std), std)
            return (spectral_features - mean) / (std + epsilon)

    def forward(self, x, spectral_features):
        # Initial preprocessing
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        # Apply normalizations
        if self.normalize_channels:
            x = self._normalize_channels(x)
        spectral_features = self._normalize_spectral(spectral_features)
        
        # Process CNN in chunks
        cnn_outputs = []
        for i in range(0, x.size(2), self.cnn_chunk_size):
            # Prepare chunk
            end_idx = min(i + self.cnn_chunk_size, x.size(2))
            chunk = x[:, :, i:end_idx]
            
            # Initialize padding size
            padding_size = 0
            
            # Pad if necessary
            if chunk.size(2) < self.cnn_chunk_size:
                padding_size = self.cnn_chunk_size - chunk.size(2)
                chunk = F.pad(chunk, (0, padding_size))
            
            # Process chunk
            chunk_out = self._process_cnn_chunk(chunk)
            
            # Remove padding if added
            if padding_size > 0:
                chunk_out = chunk_out[..., :-padding_size]
            
            cnn_outputs.append(chunk_out)
            
        
            
        # Combine CNN outputs
        cnn_out = torch.cat(cnn_outputs, dim=2)
        cnn_out = cnn_out.transpose(1, 2)  # (batch_size, seq_len, channels)
        
        # Expand spectral features and combine
        spectral_features = spectral_features.unsqueeze(1).expand(-1, cnn_out.shape[1], -1)
        combined = torch.cat([cnn_out, spectral_features], dim=-1)
        projected = self.project(combined)
        
        # Free memory
        del combined, cnn_out, cnn_outputs
        torch.cuda.empty_cache()
        
        # Process LSTM in chunks
        lstm_outputs = []
        h = None
        
        for i in range(0, projected.size(1), self.lstm_chunk_size):
            end_idx = min(i + self.lstm_chunk_size, projected.size(1))
            chunk = projected[:, i:end_idx]
            
            # Process chunk
            chunk_out, h = self._process_lstm_chunk(chunk, h)
            lstm_outputs.append(chunk_out)
            
            # Periodic memory cleanup
            if i % 200 == 0:
                torch.cuda.empty_cache()
        
        # Combine LSTM outputs and process
        lstm_out = torch.cat(lstm_outputs, dim=1)
        del lstm_outputs, projected
        torch.cuda.empty_cache()
        
        # Final processing
        attended = self.attention(lstm_out)
        output = self.classifier(attended)
        
        return output

    def update_normalization_stats(self, dataset_medians, dataset_iqrs):
        """Update the stored dataset statistics"""
        self.channel_medians = dataset_medians.to(self.project[0].weight.device)
        self.channel_iqrs = dataset_iqrs.to(self.project[0].weight.device)

    def predict(self, x, spectral_features=None):
        """Make predictions with the model"""
        if not self.check_input_dimensions(x):
            print("Error in input dimensions")
            return -1
        
        with torch.no_grad():
            output = self.forward(x, spectral_features)
        
        return torch.argmax(output, dim=-1).numpy()

    def check_input_dimensions(self, x):
        """Verify input dimensions"""
        if x.dim() != 4:
            print("Input dimensions is different than 4")
            return False
        if x.shape[1] != 4:
            print("Second dimension should be equal to 4")
            return False
        if x.shape[2] != 3000:
            print("Third dimension should be equal to 3000")
            return False
        if x.shape[3] != 1:
            print("Final dimension should be equal to 1")
            return False
        return True