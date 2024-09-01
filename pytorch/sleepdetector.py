import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class SleepDetectorCNN(nn.Module):
    def __init__(self, n_filters=[8, 16, 32], kernel_size=[50, 8, 8], Fs=100, n_classes=5):
        super(SleepDetectorCNN, self).__init__()

        # Convolutional layers for each input signal
        self.conv1_1 = nn.Conv1d(1, n_filters[0], kernel_size[0], padding='same')
        self.conv1_2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size[1], padding='same')
        self.conv1_3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size[2], padding='same')
        
        self.conv2_1 = nn.Conv1d(1, n_filters[0], kernel_size[0], padding='same')
        self.conv2_2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size[1], padding='same')
        self.conv2_3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size[2], padding='same')

        self.conv3_1 = nn.Conv1d(1, n_filters[0], kernel_size[0], padding='same')
        self.conv3_2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size[1], padding='same')
        self.conv3_3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size[2], padding='same')

        self.conv4_1 = nn.Conv1d(1, n_filters[0], kernel_size[0], padding='same')
        self.conv4_2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size[1], padding='same')
        self.conv4_3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size[2], padding='same')

        self.bn1 = nn.BatchNorm1d(n_filters[0])
        self.bn2 = nn.BatchNorm1d(n_filters[1])
        self.bn3 = nn.BatchNorm1d(n_filters[2])

        self.pool = nn.MaxPool1d(8)

        # Fully connected layer
        self.fc = nn.Linear(4 * n_filters[2] * (30*Fs // 64), n_classes)

    def conv_block(self, x, conv1, conv2, conv3):
        x = F.relu(self.bn1(conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(conv3(x)))
        x = self.pool(x)
        return x

    def forward(self, sig1, sig2, sig3, sig4):
        x1 = self.conv_block(sig1, self.conv1_1, self.conv1_2, self.conv1_3)
        x2 = self.conv_block(sig2, self.conv2_1, self.conv2_2, self.conv2_3)
        x3 = self.conv_block(sig3, self.conv3_1, self.conv3_2, self.conv3_3)
        x4 = self.conv_block(sig4, self.conv4_1, self.conv4_2, self.conv4_3)

        # Concatenate along the channel dimension
        merged_vector = torch.cat((x1, x2, x3, x4), dim=1)

        flattened_vector = merged_vector.view(merged_vector.size(0), -1)
        out = self.fc(flattened_vector)
        return out