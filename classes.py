import torch
import torch.nn as nn
from sleepdetector_new import ImprovedSleepdetector
import numpy as np

class EnsembleModel(nn.Module):
    def __init__(self, model_params, n_models=3):
        super().__init__()
        self.models = nn.ModuleList([ImprovedSleepdetector(**model_params) for _ in range(n_models)])
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.1)  # Reduced 'a' parameter
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.1)  # Reduced gain
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x, spectral_features):
        outputs = [model(x.clone(), spectral_features.clone()) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

class DiverseEnsembleModel(nn.Module):
    def __init__(self, model_params, n_models=3):
        super().__init__()
        self.models = nn.ModuleList([
            ImprovedSleepdetector(**{**model_params, 'dropout': model_params['dropout'] * (i+1)/n_models})
            for i in range(n_models)
        ])
    
    def forward(self, x, spectral_features):
        outputs = [model(x.clone(), spectral_features.clone()) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
    
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.label_to_indices = {label: np.where(labels == label)[0] for label in set(labels)}
        self.used_label_indices_count = {label: 0 for label in set(labels)}
        self.count = 0
        self.n_classes = len(set(labels))
        self.n_samples = len(labels)
        
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_samples:
            classes = list(self.label_to_indices.keys())
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                    self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.batch_size // self.n_classes
                ])
                self.used_label_indices_count[class_] += self.batch_size // self.n_classes
                if self.used_label_indices_count[class_] + self.batch_size // self.n_classes > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_samples // self.batch_size