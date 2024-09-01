import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc1 = nn.Linear(n_filters[-1] * 4 * 47, 512)  # Updated input size
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, n_classes)

    def forward(self, x):
        print(f"Input shape to CNN: {x.shape}")  # Debugging
        
        x_list = []
        for i in range(4):
            x_i = x[:, i:i+1]
            print(f"Shape of x_{i}: {x_i.shape}")  # Debugging
            x_i = self.conv_blocks[i](x_i)
            print(f"Shape after conv_blocks[{i}]: {x_i.shape}")  # Debugging
            x_list.append(x_i)
        
        x = torch.cat(x_list, dim=1)
        print(f"Shape after concatenation: {x.shape}")  # Debugging
        
        x = self.flatten(x)
        print(f"Shape after flatten: {x.shape}")  # Debugging
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        
        print(f"Final output shape: {x.shape}")  # Debugging
        return x


class ImprovedAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(ImprovedAttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class ImprovedSleepDetectorLSTM(nn.Module):
    def __init__(self, input_size=1029, hidden_size=256, n_layers=2, n_classes=5):
        super(ImprovedSleepDetectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.attention = ImprovedAttentionLayer(hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out)
        x = F.relu(self.fc1(attn_out))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

class ImprovedSleepdetector(nn.Module):
    def __init__(self, cnn_path=None, lstm_path=None, seq_length=32):
        super(ImprovedSleepdetector, self).__init__()
        self.feature_extractor = ImprovedFeatureExtractor()
        self.cnn = ImprovedSleepDetectorCNN()
        self.lstm = ImprovedSleepDetectorLSTM(input_size=1029)  # Updated input size
        
        if cnn_path is not None:
            self.cnn.load_state_dict(torch.load(cnn_path))
        if lstm_path is not None:
            self.lstm.load_state_dict(torch.load(lstm_path))
        
        self.seq_length = seq_length
        self.iqr_target = torch.tensor([7.90, 11.37, 7.92, 11.56])
        self.med_target = torch.tensor([0.0257, 0.0942, 0.02157, 0.1055])

    def forward(self, x):
        print(f"Input shape to ImprovedSleepdetector: {x.shape}")  # Debugging
        
        # Remove the last dimension if it's 1
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        
        print(f"Shape after squeeze: {x.shape}")  # Debugging
        
        # Apply normalization
        for i in range(4):
            x[:, i] = self.med_target[i] + (x[:, i] - x[:, i].median(dim=-1, keepdim=True).values) * \
                      (self.iqr_target[i] / (x[:, i].quantile(0.75, dim=-1, keepdim=True) - x[:, i].quantile(0.25, dim=-1, keepdim=True)))
        
        print(f"Shape after normalization: {x.shape}")  # Debugging
        
        # Extract additional features
        features = [self.feature_extractor(x[:, i:i+1].unsqueeze(-1)) for i in range(4)]
        features = torch.cat(features, dim=1)
        
        print(f"Shape of extracted features: {features.shape}")  # Debugging
        
        # CNN forward pass
        x_cnn = self.cnn(x)
        
        print(f"Shape of CNN output: {x_cnn.shape}")  # Debugging
        
        # Combine CNN output with extracted features
        combined_features = torch.cat([x_cnn, features], dim=1)
        
        print(f"Shape of combined features: {combined_features.shape}")  # Debugging
        
        # Reshape for LSTM
        batch_size, feature_dim = combined_features.shape
        x_lstm = combined_features.unsqueeze(1)  # Add sequence dimension
        
        print(f"Shape of LSTM input: {x_lstm.shape}")  # Debugging
        
        # LSTM forward pass
        y_lstm = self.lstm(x_lstm)
        
        print(f"Final output shape: {y_lstm.shape}")  # Debugging
        
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
