import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import Sampler
import numpy as np
from sleepdetector_new import ImprovedSleepdetector

class EnsembleModel(nn.Module):
    """Ensemble of sleep detection models"""
    def __init__(self, model_params: Dict, n_models: int = 3):
        super().__init__()
        self.models = nn.ModuleList([
            ImprovedSleepdetector(**model_params) 
            for _ in range(n_models)
        ])
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights with specific schemes per layer type"""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(
                module.weight, 
                mode='fan_out', 
                nonlinearity='relu', 
                a=0.1
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging predictions from all models"""
        outputs = [
            model(x.clone(), spectral_features.clone()) 
            for model in self.models
        ]
        return torch.mean(torch.stack(outputs), dim=0)

class DiverseEnsembleModel(nn.Module):
    """Ensemble with varying dropout rates for diversity"""
    def __init__(self, model_params: Dict, n_models: int = 3):
        super().__init__()
        self.models = nn.ModuleList([
            ImprovedSleepdetector(**{
                **model_params,
                'dropout': model_params['dropout'] * (i + 1) / n_models
            })
            for i in range(n_models)
        ])
    
    def forward(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging predictions from diverse models"""
        outputs = [
            model(x.clone(), spectral_features.clone()) 
            for model in self.models
        ]
        return torch.mean(torch.stack(outputs), dim=0)

class BalancedBatchSampler(Sampler):
    """Sampler that ensures balanced class representation in each batch"""
    def __init__(self, labels: np.ndarray, batch_size: int):
        self.labels = labels
        self.batch_size = batch_size
        self.label_to_indices = {
            label: np.where(labels == label)[0] 
            for label in set(labels)
        }
        self.used_label_indices_count = {
            label: 0 for label in set(labels)
        }
        self.count = 0
        self.n_classes = len(set(labels))
        self.n_samples = len(labels)
        
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_samples:
            classes = list(self.label_to_indices.keys())
            indices = []
            for class_ in classes:
                class_indices = self.label_to_indices[class_]
                start_idx = self.used_label_indices_count[class_]
                samples_per_class = self.batch_size // self.n_classes
                
                # Get indices for this class
                if start_idx + samples_per_class > len(class_indices):
                    np.random.shuffle(class_indices)
                    self.used_label_indices_count[class_] = 0
                    start_idx = 0
                    
                selected_indices = class_indices[
                    start_idx:start_idx + samples_per_class
                ]
                indices.extend(selected_indices)
                self.used_label_indices_count[class_] += samples_per_class
                
            yield indices
            self.count += self.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

class EarlyStopping:
    """Early stopping handler with multiple criteria and validation tracking"""
    def __init__(
        self,
        patience: int = 10,
        min_epochs: int = 20,
        min_delta: float = 0.001,
        monitor: List[str] = ['loss', 'accuracy'],
        mode: str = 'auto'
    ):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.monitor = monitor if isinstance(monitor, list) else [monitor]
        self.mode = mode
        self.best_metrics = {
            m: float('inf') if mode == 'min' else float('-inf') 
            for m in monitor
        }
        self.counter = 0
        self.best_epoch = 0
        self.stop = False
        self.best_state = None

    def __call__(
        self,
        metrics: Dict[str, float],
        epoch: int,
        state_dict: Optional[Dict] = None
    ) -> bool:
        """
        Check if training should stop based on monitored metrics
        
        Args:
            metrics: Dictionary of metric names and values to monitor
            epoch: Current epoch number
            state_dict: Optional model state dict to save if metrics improve
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        improved = False
        
        for metric in self.monitor:
            current = metrics[metric]
            best = self.best_metrics[metric]
            
            if self.mode == 'min':
                delta = best - current
            else:
                delta = current - best
                
            if delta > self.min_delta:
                self.best_metrics[metric] = current
                improved = True
                
        if improved:
            self.counter = 0
            self.best_epoch = epoch
            if state_dict is not None:
                self.best_state = state_dict.copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience and epoch >= self.min_epochs:
            self.stop = True
            
        return self.stop