import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ImprovedSleepDetectorCNN(nn.Module):
    def __init__(self, n_filters=[32, 64, 128], kernel_size=[50, 25, 12], n_classes=5):
        super(ImprovedSleepDetectorCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        
        for i in range(4):  # 4 input signals
            layers = []
            in_channels = 1
            for j, (filters, kernel) in enumerate(zip(n_filters, kernel_size)):
                layers.extend([
                    nn.Conv1d(in_channels, filters, kernel_size=kernel, stride=1, padding=kernel//2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4)
                ])
                in_channels = filters
            self.conv_blocks.append(nn.Sequential(*layers))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_filters[-1] * 4 * 47, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, n_filters[-1])
        self.bn2 = nn.BatchNorm1d(n_filters[-1])
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(n_filters[-1], n_classes)

    def forward(self, x):
        x_list = []
        for i in range(4):
            x_i = x[:, i:i+1]
            x_i = self.conv_blocks[i](x_i)
            x_list.append(x_i)
        
        x = torch.cat(x_list, dim=1)
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x

# class ImprovedSleepDetectorCNN(nn.Module):
#     def __init__(self, n_filters=[32, 64, 128], kernel_size=[50, 25, 12], n_classes=5):
#         super(ImprovedSleepDetectorCNN, self).__init__()
#         self.conv_blocks = nn.ModuleList()
        
#         for i in range(4):  # 4 input signals
#             layers = []
#             in_channels = 1
#             for j, (filters, kernel) in enumerate(zip(n_filters, kernel_size)):
#                 layers.extend([
#                     nn.Conv1d(in_channels, filters, kernel_size=kernel, stride=1, padding=kernel//2),
#                     nn.BatchNorm1d(filters),
#                     nn.ReLU(),
#                     nn.MaxPool1d(kernel_size=4, stride=4)
#                 ])
#                 in_channels = filters
#             self.conv_blocks.append(nn.Sequential(*layers))
        
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(n_filters[-1] * 4 * 47, 512)  # Updated input size
#         self.fc2 = nn.Linear(512, n_filters[-1])
#         self.dropout = nn.Dropout(0.5)
#         self.output = nn.Linear(n_filters[-1], n_classes)

#     def forward(self, x):
#         x_list = []
#         for i in range(4):
#             x_i = x[:, i:i+1]
#             x_i = self.conv_blocks[i](x_i)
#             x_list.append(x_i)
        
#         x = torch.cat(x_list, dim=1)
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         return x  # This should now be of shape (batch_size, n_filters[-1])


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

class ImprovedSleepdetector(nn.Module):
    def __init__(self, n_filters=[32, 64, 128], lstm_hidden=256, lstm_layers=2, dropout=0.5, seq_length=32):
        super(ImprovedSleepdetector, self).__init__()
        self.feature_extractor = ImprovedFeatureExtractor()
        self.cnn = ImprovedSleepDetectorCNN(n_filters=n_filters)
        self.combine_features = nn.Linear(n_filters[-1] + 16, lstm_hidden)
        self.lstm = ImprovedSleepDetectorLSTM(input_size=lstm_hidden, 
                                             hidden_size=lstm_hidden, 
                                             n_layers=lstm_layers, 
                                             dropout=dropout)
        self.seq_length = seq_length
        
        # Register buffers for dataset statistics
        self.register_buffer('channel_medians', torch.zeros(4))
        self.register_buffer('channel_iqrs', torch.ones(4))
        self.normalize_channels = True  # Flag to control normalization

    def update_normalization_stats(self, dataset_medians, dataset_iqrs):
        """Update the stored dataset statistics"""
        self.channel_medians = dataset_medians.to(self.combine_features.weight.device)
        self.channel_iqrs = dataset_iqrs.to(self.combine_features.weight.device)

    def forward(self, x, spectral_features):
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        # Channel normalization using dataset statistics
        if self.normalize_channels:
            epsilon = 1e-8
            for i in range(x.shape[1]):
                x[:, i] = (x[:, i] - self.channel_medians[i]) / (self.channel_iqrs[i] + epsilon)

        # Spectral feature normalization (batch-wise is fine here)
        mean = spectral_features.mean(dim=0, keepdim=True)
        std = spectral_features.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        spectral_features = (spectral_features - mean) / (std + epsilon)

        # Rest of the forward pass
        x_cnn = self.cnn(x)
        if not torch.isfinite(x_cnn).all():
            logging.error(f"Non-finite values detected after CNN: {x_cnn}")
            x_cnn = torch.nan_to_num(x_cnn, nan=0.0, posinf=1e6, neginf=-1e6)

        combined_features = torch.cat([x_cnn, spectral_features], dim=1)
        combined_features = F.relu(self.combine_features(combined_features))
        if not torch.isfinite(combined_features).all():
            logging.error(f"Non-finite values detected after combining features: {combined_features}")
            combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

        x_lstm = combined_features.unsqueeze(1)
        y_lstm = self.lstm(x_lstm)
        if not torch.isfinite(y_lstm).all():
            logging.error(f"Non-finite values detected after LSTM: {y_lstm}")
            y_lstm = torch.nan_to_num(y_lstm, nan=0.0, posinf=1e6, neginf=-1e6)

        return y_lstm



    def predict(self, x):
        if not self.check_input_dimensions(x):
            print("Error in input dimensions")
            return -1
        
        with torch.no_grad():
            output = self.forward(x)
        
        return torch.argmax(output, dim=-1).numpy()

    def check_input_dimensions(self, x):
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