import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Add these new module classes after your imports
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),  # Prevent 0 dimensions
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch, channel, time]
        b, c, _ = x.size()
        # Global average pooling
        y = self.avg_pool(x)  # [batch, channel, 1]
        y = y.squeeze(-1)     # [batch, channel]
        # Channel attention
        y = self.fc(y)        # [batch, channel]
        y = y.unsqueeze(-1)   # [batch, channel, 1]
        # Scale features
        return x * y.expand_as(x)

class TemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)
        self.temperature = nn.Parameter(torch.sqrt(torch.FloatTensor([channels])))
        
    def forward(self, x):
        # x shape: [batch, channels, time]
        # Compute attention scores
        attn = self.conv(x)                    # [batch, 1, time]
        attn = attn / self.temperature         # Scale for stable softmax
        attn = F.softmax(attn, dim=2)         # Softmax over time dimension
        # Apply attention
        return x * attn.expand_as(x)          # Broadcasting will handle expansion

class CrossChannelInteraction(nn.Module):
    def __init__(self, in_channels, feature_channels):
        super().__init__()
        self.conv = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(feature_channels)
        
    def forward(self, x):
        # x shape: [batch, n_channels, feature_channels]
        # Need to make it [batch, feature_channels, n_channels, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv(x)
        x = self.norm(x)
        # Return to [batch, n_channels, feature_channels]
        return x.squeeze(-1).permute(0, 2, 1)

class FeatureFusion(nn.Module):
    def __init__(self, cnn_dim, spectral_dim, hidden_dim):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.spectral_proj = nn.Linear(spectral_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, cnn_features, spectral_features):
        cnn_proj = self.cnn_proj(cnn_features)
        spectral_proj = self.spectral_proj(spectral_features)
        gate = torch.sigmoid(self.gate(torch.cat([cnn_proj, spectral_proj], dim=-1)))
        return gate * cnn_proj + (1 - gate) * spectral_proj


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

class ImprovedSleepDetectorCNN(nn.Module):
    def __init__(self, n_filters=[32, 64, 128], kernel_size=[50, 25, 12], n_classes=5):
        super(ImprovedSleepDetectorCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        
        # Single channel processing block
        layers = []
        in_channels = 1
        for j, (filters, kernel) in enumerate(zip(n_filters, kernel_size)):
            layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4)
            ])
            in_channels = filters
        self.conv_block = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters[-1] * 47, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, n_filters[-1])
        self.bn2 = nn.BatchNorm1d(n_filters[-1])
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
    
    def forward(self, lstm_output):
        attention_outputs = []
        for head in self.attention_heads:
            attention_weights = F.softmax(head(lstm_output), dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            attention_outputs.append(context_vector)
        return torch.cat(attention_outputs, dim=1)

class ImprovedSleepDetectorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, n_layers=2, n_classes=5, num_heads=4, dropout=0.5):
        super(ImprovedSleepDetectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0)
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

class ImprovedSleepdetector(nn.Module):
    def __init__(self, n_channels=4, n_filters=[32, 64, 128], lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super(ImprovedSleepdetector, self).__init__()
        self.n_channels = n_channels
        
        # Core components
        self.cnn = ImprovedSleepDetectorCNN(n_filters=n_filters)
        
        # Attention and interaction components
        self.se_blocks = nn.ModuleList([SEBlock(n_filters[-1]) for _ in range(n_channels)])
        self.temporal_attention = TemporalAttention(n_filters[-1])
        self.cross_channel = CrossChannelInteraction(n_channels, n_filters[-1])
        self.feature_fusion = FeatureFusion(n_filters[-1] * n_channels, 16, lstm_hidden)
        
        self.lstm = ImprovedSleepDetectorLSTM(
            input_size=lstm_hidden, 
            hidden_size=lstm_hidden, 
            n_layers=lstm_layers, 
            dropout=dropout
        )
        
        # Register buffers for dataset statistics
        self.register_buffer('channel_medians', torch.zeros(n_channels))
        self.register_buffer('channel_iqrs', torch.ones(n_channels))
        self.normalize_channels = True

    

    def update_normalization_stats(self, dataset_medians, dataset_iqrs):
        """Update the stored dataset statistics"""
        self.channel_medians = dataset_medians.to(self.lstm.fc1.weight.device)
        self.channel_iqrs = dataset_iqrs.to(self.lstm.fc1.weight.device)

    def forward(self, x, spectral_features):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        if x.dim() != 3 or x.shape[1] != self.n_channels or x.shape[2] != 3000:
            raise ValueError(
                f"Expected input shape (batch_size, {self.n_channels}, 3000), got {x.shape}"
            )
        
        # Channel normalization
        if self.normalize_channels:
            epsilon = 1e-8
            for i in range(self.n_channels):
                x[:, i] = (x[:, i] - self.channel_medians[i]) / (self.channel_iqrs[i] + epsilon)

        # Spectral feature normalization
        mean = spectral_features.mean(dim=0, keepdim=True)
        std = spectral_features.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        spectral_features = (spectral_features - mean) / (std + epsilon)

        # Feature extraction
        x_list = []
        for i in range(self.n_channels):
            x_i = x[:, i:i+1]  # [batch, 1, 3000]
            x_i = self.cnn(x_i)  # [batch, n_filters[-1]]
            x_i = x_i.unsqueeze(-1)  # [batch, n_filters[-1], 1]
            x_i = self.se_blocks[i](x_i).squeeze(-1)
            x_i = self.temporal_attention(x_i.unsqueeze(-1)).squeeze(-1)
            x_list.append(x_i)
        
        # Stack channels: [batch, n_channels, n_filters[-1]]
        x = torch.stack(x_list, dim=1)
        
        # Cross-channel interaction
        x = self.cross_channel(x)  # Still [batch, n_channels, n_filters[-1]]
        
        # Flatten: [batch, n_channels * n_filters[-1]]
        x = x.reshape(x.size(0), -1)
        
        # Feature fusion and LSTM
        combined_features = self.feature_fusion(x, spectral_features)
        x_lstm = combined_features.unsqueeze(1)
        y_lstm = self.lstm(x_lstm)
        
        return y_lstm

    # Keep existing predict method unchanged
    def predict(self, x, spectral_features):
        """Make predictions on input data"""
        with torch.no_grad():
            output = self.forward(x, spectral_features)
            predictions = torch.argmax(output, dim=-1)
            return predictions.cpu().numpy()


# def forward(self, x, spectral_features):
    #     if x.shape[-1] == 1:
    #         x = x.squeeze(-1)
        
    #     epsilon = 1e-8
    #     for i in range(4):
    #         x[:, i] = self.med_target[i] + (x[:, i] - x[:, i].median(dim=-1, keepdim=True).values) * \
    #             (self.iqr_target[i] / (x[:, i].quantile(0.75, dim=-1, keepdim=True) - x[:, i].quantile(0.25, dim=-1, keepdim=True) + epsilon))

    #     # Normalize spectral features
    #     # spectral_features = (spectral_features - spectral_features.mean(dim=0, keepdim=True)) / (spectral_features.std(dim=0, keepdim=True) + epsilon)

    #     # Modified spectral features normalization
    #     mean = spectral_features.mean(dim=0, keepdim=True)
    #     std = spectral_features.std(dim=0, keepdim=True)
    #     # Replace zero standard deviations with 1 to avoid division by zero
    #     std = torch.where(std == 0, torch.ones_like(std), std)
    #     spectral_features = (spectral_features - mean) / (std + epsilon)

    #     x_cnn = self.cnn(x)
    #     if not torch.isfinite(x_cnn).all():
    #         logging.error(f"Non-finite values detected after CNN: {x_cnn}")
    #         x_cnn = torch.nan_to_num(x_cnn, nan=0.0, posinf=1e6, neginf=-1e6)

    #     combined_features = torch.cat([x_cnn, spectral_features], dim=1)
    #     combined_features = F.relu(self.combine_features(combined_features))
    #     if not torch.isfinite(combined_features).all():
    #         logging.error(f"Non-finite values detected after combining features: {combined_features}")
    #         combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

    #     x_lstm = combined_features.unsqueeze(1)
    #     y_lstm = self.lstm(x_lstm)
    #     if not torch.isfinite(y_lstm).all():
    #         logging.error(f"Non-finite values detected after LSTM: {y_lstm}")
    #         y_lstm = torch.nan_to_num(y_lstm, nan=0.0, posinf=1e6, neginf=-1e6)

    #     return y_lstm