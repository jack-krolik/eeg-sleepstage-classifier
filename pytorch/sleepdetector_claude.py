import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, input_size=3000):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(1, 50), stride=1, padding=(0, 25))
        self.pool = nn.AdaptiveAvgPool2d((1, 64))
        self.fc = nn.Linear(32 * 64, 128)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class SleepDetectorCNN(nn.Module):
    def __init__(self, n_filters=[16, 32, 64], kernel_size=[50, 8, 8], Fs=100, n_classes=5):
        super(SleepDetectorCNN, self).__init__()
        self.conv_blocks = nn.ModuleList()
        
        for i in range(4):  # 4 input signals
            layers = []
            in_channels = 1
            for j, (filters, kernel) in enumerate(zip(n_filters, kernel_size)):
                layers.extend([
                    nn.Conv2d(in_channels, filters, kernel_size=(1, kernel), stride=1, padding=(0, kernel//2)),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
                ])
                in_channels = filters
            self.conv_blocks.append(nn.Sequential(*layers))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_filters[-1] * 4 * (3000 // (4 ** 3)), 256)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(256, n_classes)

    def forward(self, x):
        print(f"Input shape to SleepDetectorCNN: {x.shape}")
        
        x = [self.conv_blocks[i](x[:, i:i+1]) for i in range(4)]
        x = torch.cat(x, dim=1)
        
        print(f"Shape after conv blocks: {x.shape}")
        
        x = self.flatten(x)
        print(f"Shape after flatten: {x.shape}")
        
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.output(x)
        
        print(f"Final output shape: {x.shape}")
        
        return x


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

class SleepDetectorLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, n_layers=2, n_classes=5):
        super(SleepDetectorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        output = self.fc(attn_out)
        return output

class Sleepdetector(nn.Module):
    def __init__(self, cnn_path=None, lstm_path=None, seq_length=32):
        super(Sleepdetector, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.cnn = SleepDetectorCNN()
        self.lstm = SleepDetectorLSTM(input_size=256+512)  # Update input size
        
        if cnn_path is not None:
            self.cnn.load_state_dict(torch.load(cnn_path))
        if lstm_path is not None:
            self.lstm.load_state_dict(torch.load(lstm_path))
        
        self.seq_length = seq_length
        self.iqr_target = torch.tensor([7.90, 11.37, 7.92, 11.56])
        self.med_target = torch.tensor([0.0257, 0.0942, 0.02157, 0.1055])

    def forward(self, x):
        print(f"Input shape to Sleepdetector: {x.shape}")
        
        # Apply normalization
        for i in range(4):
            x[:, i] = self.med_target[i] + (x[:, i] - x[:, i].median(dim=-1, keepdim=True).values) * \
                      (self.iqr_target[i] / (x[:, i].quantile(0.75, dim=-1, keepdim=True) - x[:, i].quantile(0.25, dim=-1, keepdim=True)))
        
        print(f"Shape after normalization: {x.shape}")
        
        # Extract additional features
        features = [self.feature_extractor(x[:, i:i+1]) for i in range(4)]
        features = torch.cat(features, dim=1)
        
        print(f"Shape of extracted features: {features.shape}")
        
        # CNN forward pass
        x_cnn = self.cnn(x)
        
        print(f"Shape of CNN output: {x_cnn.shape}")
        
        # Combine CNN output with extracted features
        combined_features = torch.cat([x_cnn, features], dim=1)
        
        print(f"Shape of combined features: {combined_features.shape}")
        
        # Reshape for LSTM
        batch_size, feature_dim = combined_features.shape
        n_seqs = batch_size // self.seq_length
        x_lstm = combined_features[:n_seqs * self.seq_length].view(n_seqs, self.seq_length, feature_dim)
        
        print(f"Shape of LSTM input: {x_lstm.shape}")
        
        # LSTM forward pass
        y_lstm = self.lstm(x_lstm)
        
        print(f"Final output shape: {y_lstm.shape}")
        
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
