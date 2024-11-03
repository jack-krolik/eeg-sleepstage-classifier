import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import Sampler
import numpy as np
from sleepdetector_new import ImprovedSleepdetector
from tools.utils import *
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tools.config import CONFIG, device, cuda_manager
from collections import Counter
import torch.nn.functional as F
import logging
from collections import defaultdict
import os
import random


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
    

class BalancedSleepDataset(Dataset):
    def __init__(self, X, X_spectral, y, night_indices, augment=True, temporal_window=5):
        self.X = X
        self.X_spectral = X_spectral
        self.y = y
        self.night_indices = night_indices
        self.augment = augment
        
        # Calculate sampling weights
        self.weights = self._create_balanced_sampling_strategy(temporal_window)
        
        # Create index mappings
        self.night_sequences = defaultdict(list)
        for idx in range(len(y)):
            night = night_indices[idx].item()
            self.night_sequences[night].append(idx)
    
    def _create_balanced_sampling_strategy(self, temporal_window):
        """Calculate importance scores for balanced sampling"""
        class_counts = Counter(self.y.numpy())
        total_samples = len(self.y)
        
        # Calculate effective numbers
        beta = 0.9999
        effective_samples = {
            cls: (1 - beta**count) / (1 - beta)
            for cls, count in class_counts.items()
        }
        
        # Analyze transitions
        transitions = defaultdict(int)
        for night in torch.unique(self.night_indices):
            night_mask = self.night_indices == night
            night_y = self.y[night_mask]
            
            for i in range(len(night_y) - 1):
                transition = (night_y[i].item(), night_y[i + 1].item())
                transitions[transition] += 1
        
        # Calculate importance scores
        importance_scores = torch.zeros(len(self.y))
        
        for idx in range(len(self.y)):
            night_mask = self.night_indices == self.night_indices[idx]
            night_y = self.y[night_mask]
            pos_in_night = torch.where(night_mask)[0].tolist().index(idx)
            
            # Base importance from class rarity
            cls = self.y[idx].item()
            importance = 1 / effective_samples[cls]
            
            # Adjust for transitions
            if pos_in_night > 0:
                prev_transition = (night_y[pos_in_night-1].item(), cls)
                importance *= 1 + 1/max(transitions[prev_transition], 1)
            
            # Consider local patterns
            start_idx = max(0, pos_in_night - temporal_window)
            end_idx = min(len(night_y), pos_in_night + temporal_window + 1)
            sequence = night_y[start_idx:end_idx]
            unique_stages = len(torch.unique(sequence))
            importance *= (1 + 0.1 * unique_stages)
            
            importance_scores[idx] = importance
        
        return importance_scores
    
    def _is_valid_transition(self, stage1, stage2):
        valid_transitions = {
            0: {0, 1},      # N3 → N3/N2
            1: {0, 1, 2},   # N2 → N3/N2/N1
            2: {1, 2, 3},   # N1 → N2/N1/REM
            3: {2, 3, 4},   # REM → N1/REM/Wake
            4: {2, 3, 4}    # Wake → N1/REM/Wake
        }
        return stage2.item() in valid_transitions[stage1.item()]
    
    def _augment_signal(self, signal, stage):
        from tools.functions import time_warp
        if stage in [0, 1]:  # N3, N2
            return time_warp(signal, sigma=0.1, knot=3)
        elif stage == 3:  # REM
            return time_warp(signal, sigma=0.2, knot=5)
        else:  # Wake, N1
            return time_warp(signal, sigma=0.3, knot=4)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        x_spectral = self.X_spectral[idx]
        y = self.y[idx]
        
        if self.augment:
            night = self.night_indices[idx].item()
            night_seq = self.night_sequences[night]
            pos_in_night = night_seq.index(idx)
            
            if 0 < pos_in_night < len(night_seq) - 1:
                prev_stage = self.y[night_seq[pos_in_night - 1]]
                next_stage = self.y[night_seq[pos_in_night + 1]]
                
                if self._is_valid_transition(prev_stage, y) and \
                   self._is_valid_transition(y, next_stage):
                    x = self._augment_signal(x, y.item())
        
        return x, x_spectral, y

class SleepStageEvaluator:
    def __init__(self, model_dir=CONFIG['model_dir']):
        self.model_dir = model_dir
        self.device = device
        # Map class indices to sleep stage names
        self.class_mapping = SLEEP_STAGES
        self.class_names = [self.class_mapping[i] for i in range(5)]
        ensure_dir(os.path.join(model_dir, 'test_results'))
        self.save_outputs = True
    
    def plot_confusion_matrices(self, y_true, y_pred, model_name):
        """Plot both absolute and percentage confusion matrices side by side"""
        # Calculate confusion matrices
        cm_absolute = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        
        # Handle division by zero for percentage calculation
        row_sums = cm_absolute.sum(axis=1)
        # Add small epsilon to avoid division by zero
        row_sums = np.where(row_sums == 0, 1e-10, row_sums)
        cm_percentage = (cm_absolute.astype('float') / row_sums[:, np.newaxis] * 100)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Plot absolute confusion matrix
        sns.heatmap(cm_absolute, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_ylabel('True Sleep Stage')
        ax1.set_xlabel('Predicted Sleep Stage')
        ax1.set_title(f'{model_name}\nAbsolute Confusion Matrix')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=45, ha='right')
        
        # Plot percentage confusion matrix
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', square=True,
                xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_ylabel('True Sleep Stage')
        ax2.set_xlabel('Predicted Sleep Stage')
        ax2.set_title(f'{model_name}\nPercentage Confusion Matrix')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if self.save_outputs:
            save_path = os.path.join(self.model_dir, 'test_results', f'{model_name}_confusion_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Display in notebook and save
        display(plt.gcf())
        plt.close()
        
        return cm_absolute, cm_percentage
    
    def display_metrics(self, y_true, y_pred, model_name):
        """Calculate and display comprehensive metrics"""
        # Log class distributions before calculation
        logging.info("\nClass distribution in evaluation:")
        logging.info(f"True labels distribution: {Counter(y_true)}")
        logging.info(f"Predicted labels distribution: {Counter(y_pred)}")
        
        # Calculate overall metrics with all classes
        labels = list(range(len(self.class_names)))  # Ensure all classes are included
        accuracy = (y_pred == y_true).mean() * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels) * 100
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels) * 100
        
        # Generate detailed classification report with all classes
        report = classification_report(y_true, y_pred, 
                                    target_names=self.class_names,
                                    labels=labels,
                                    output_dict=True,
                                    zero_division=0)
        
        # Create DataFrame from report
        metrics_df = pd.DataFrame(report)
        
        # Print overall metrics
        print(f"\n{'-'*50}")
        print(f"{model_name} Results:")
        print(f"{'-'*50}")
        print(f"Overall Accuracy: {accuracy:.1f}%")
        print(f"Macro F1-Score: {f1_macro:.1f}%")
        print(f"Weighted F1-Score: {f1_weighted:.1f}%")
        
        # Convert metrics to percentages (except support)
        metrics_formatted = metrics_df.copy()
        for col in metrics_formatted.columns:
            if col != 'support':
                metrics_formatted[col] = metrics_formatted[col] * 100
        
        # Create a more readable version of the metrics
        display_metrics = pd.DataFrame(
            index=pd.Index(['Precision', 'Recall', 'F1-Score', 'Support']),
            columns=self.class_names + ['Macro Avg', 'Weighted Avg']
        )
        
        # Fill in the class metrics
        for class_name in self.class_names:
            if class_name in metrics_formatted.columns:
                display_metrics.loc['Precision', class_name] = metrics_formatted.loc['precision', class_name]
                display_metrics.loc['Recall', class_name] = metrics_formatted.loc['recall', class_name]
                display_metrics.loc['F1-Score', class_name] = metrics_formatted.loc['f1-score', class_name]
                display_metrics.loc['Support', class_name] = metrics_formatted.loc['support', class_name]
        
        # Fill in the averages
        for avg in ['macro avg', 'weighted avg']:
            col_name = 'Macro Avg' if avg == 'macro avg' else 'Weighted Avg'
            display_metrics.loc['Precision', col_name] = metrics_formatted.loc['precision', avg]
            display_metrics.loc['Recall', col_name] = metrics_formatted.loc['recall', avg]
            display_metrics.loc['F1-Score', col_name] = metrics_formatted.loc['f1-score', avg]
            display_metrics.loc['Support', col_name] = metrics_formatted.loc['support', avg]
        
        # Create styled version for display
        styled_metrics = display_metrics.style\
            .format(lambda x: f'{x:.1f}%' if pd.notnull(x) and isinstance(x, (int, float)) and x != int(x) else f'{int(x)}' if pd.notnull(x) else '', na_rep='-')\
            .background_gradient(cmap='RdYlGn', subset=pd.IndexSlice[['Precision', 'Recall', 'F1-Score'], :])\
            .set_caption(f"{model_name} Detailed Metrics")
        
        # Display the styled metrics
        display(styled_metrics)
        
        # Save metrics to CSV
        if self.save_outputs:
            display_metrics.to_csv(os.path.join(self.model_dir, 'test_results', f'{model_name}_metrics.csv'))
        
        return metrics_df
    
    def evaluate_model(self, model, X, X_spectral, y, model_name):
        """Evaluate a single model with comprehensive metrics and visualizations"""
        print(f"\nEvaluating {model_name}...")
        model.eval()
        
        try:
            with torch.no_grad():
                # Generate predictions
                X = X.to(self.device)
                X_spectral = X_spectral.to(self.device)
                outputs = model(X, X_spectral)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                true_labels = y.cpu().numpy()
                
                # Log unique classes
                unique_true = np.unique(true_labels)
                unique_pred = np.unique(predictions)
                logging.info(f"\nUnique classes in true labels: {unique_true}")
                logging.info(f"Unique classes in predictions: {unique_pred}")
                
                # Plot confusion matrices
                cm_absolute, cm_percentage = self.plot_confusion_matrices(
                    true_labels, predictions, model_name
                )
                
                # Display and save metrics
                metrics_df = self.display_metrics(true_labels, predictions, model_name)
                
                return {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'confusion_matrix_absolute': cm_absolute,
                    'confusion_matrix_percentage': cm_percentage,
                    'metrics': metrics_df
                }
                
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            raise
    
        
        try:
            with torch.no_grad():
                # Generate predictions
                X = X.to(self.device)
                X_spectral = X_spectral.to(self.device)
                outputs = model(X, X_spectral)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                true_labels = y.cpu().numpy()
                
                # Plot confusion matrices
                cm_absolute, cm_percentage = self.plot_confusion_matrices(
                    true_labels, predictions, model_name
                )
                
                # Display and save metrics
                metrics_df = self.display_metrics(true_labels, predictions, model_name)
                
                return {
                    'predictions': predictions,
                    'true_labels': true_labels,
                    'confusion_matrix_absolute': cm_absolute,
                    'confusion_matrix_percentage': cm_percentage,
                    'metrics': metrics_df
                }
                
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            raise  # Add this to see the full error traceback



class ClassAwareBatchSampler(Sampler):
    """
    Batch sampler that ensures class balance within each batch
    """
    def __init__(self, labels, batch_size, drop_last=False):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create class-wise sample indices
        self.class_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }
        
        # Calculate samples per class per batch
        n_classes = len(self.class_indices)
        self.samples_per_class = max(1, batch_size // n_classes)
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.n_batches = min_class_size // self.samples_per_class
        if not self.drop_last and min_class_size % self.samples_per_class != 0:
            self.n_batches += 1
            
        # Create index pools for each class
        self.class_pools = {
            label: indices.copy()
            for label, indices in self.class_indices.items()
        }
    
    def __iter__(self):
        class_pools = {
            label: indices.copy()
            for label, indices in self.class_indices.items()
        }
        
        for _ in range(self.n_batches):
            batch_indices = []
            
            # Sample from each class
            for label in self.class_indices:
                pool = class_pools[label]
                
                # Reshuffle if needed
                if len(pool) < self.samples_per_class:
                    pool = self.class_indices[label].copy()
                    np.random.shuffle(pool)
                    class_pools[label] = pool
                
                # Get samples for this class
                batch_indices.extend(pool[:self.samples_per_class])
                class_pools[label] = pool[self.samples_per_class:]
            
            # Shuffle samples within the batch
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self):
        return self.n_batches
    



class TrainingMonitor:
    """Monitors and logs training metrics with class-wise tracking"""
    
    def __init__(self, num_classes: int, model_dir: str):
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.history = defaultdict(list)
        self.reset_metrics()
        
        # Create monitoring directory
        self.monitor_dir = os.path.join(model_dir, 'monitoring')
        os.makedirs(self.monitor_dir, exist_ok=True)
    
    def reset_metrics(self):
        """Reset all monitoring metrics"""
        self.class_correct = np.zeros(self.num_classes)
        self.class_total = np.zeros(self.num_classes)
        self.class_losses = np.zeros(self.num_classes)
        self.grad_norms = np.zeros(self.num_classes)
        self.batch_count = 0
        self.running_loss = 0.0
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor, 
               loss: torch.Tensor, model: torch.nn.Module):
        """Update metrics after each batch"""
        # Update accuracy metrics
        _, predicted = torch.max(outputs, 1)
        correct = predicted == targets
        
        for i in range(self.num_classes):
            mask = targets == i
            self.class_correct[i] += correct[mask].sum().item()
            self.class_total[i] += mask.sum().item()
            
            if mask.any():
                self.class_losses[i] += loss.item() * mask.sum().item()
        
        # Update gradient metrics if in training mode
        if model.training:
            for i in range(self.num_classes):
                mask = targets == i
                if mask.any():
                    class_loss = F.cross_entropy(outputs[mask], targets[mask])
                    class_loss.backward(retain_graph=True if i < self.num_classes-1 else False)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=float('inf')
                    )
                    self.grad_norms[i] += grad_norm.item()
            
        self.batch_count += 1
        self.running_loss += loss.item()
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return current metrics"""
        metrics = {}
        
        # Overall metrics
        metrics['loss'] = self.running_loss / max(self.batch_count, 1)
        total_correct = sum(self.class_correct)
        total_samples = sum(self.class_total)
        metrics['accuracy'] = total_correct / max(total_samples, 1)
        
        # Class-wise metrics
        for i in range(self.num_classes):
            acc = self.class_correct[i] / max(self.class_total[i], 1)
            loss = self.class_losses[i] / max(self.class_total[i], 1)
            grad = self.grad_norms[i] / max(self.batch_count, 1)
            
            metrics[f'class_{i}_acc'] = acc
            metrics[f'class_{i}_loss'] = loss
            metrics[f'class_{i}_grad'] = grad
        
        return metrics
    
    def update_history(self, metrics: Dict[str, float], phase: str):
        """Update training history with new metrics"""
        for name, value in metrics.items():
            self.history[f'{phase}_{name}'].append(value)
    
    def plot_metrics(self, epoch: int):
        """Plot and save training metrics"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot overall metrics
        epochs = range(1, epoch + 2)
        
        # Loss plot
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.history['train_accuracy'], label='Train')
        axes[0, 1].plot(epochs, self.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Overall Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Class-wise accuracy plot
        for i in range(self.num_classes):
            axes[1, 0].plot(epochs, 
                          [self.history[f'val_class_{i}_acc'][j] for j in range(len(epochs))],
                          label=f'Class {i}')
        axes[1, 0].set_title('Class-wise Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        
        # Gradient norms plot
        for i in range(self.num_classes):
            axes[1, 1].plot(epochs,
                          [self.history[f'train_class_{i}_grad'][j] for j in range(len(epochs))],
                          label=f'Class {i}')
        axes[1, 1].set_title('Class-wise Gradient Norms')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.monitor_dir, f'metrics_epoch_{epoch}.png'))
        plt.close()
    
    def log_metrics(self, metrics: Dict[str, float], phase: str, epoch: int):
        """Log current metrics"""
        logging.info(f"\nEpoch {epoch} - {phase} Metrics:")
        
        # Log overall metrics
        logging.info(f"    Overall Loss: {metrics['loss']:.4f}")
        logging.info(f"    Overall Accuracy: {metrics['accuracy']:.4f}")
        
        # Log class-wise metrics
        logging.info("\n    Class-wise Metrics:")
        for i in range(self.num_classes):
            logging.info(f"    Class {i}:")
            logging.info(f"        Accuracy: {metrics[f'class_{i}_acc']:.4f}")
            logging.info(f"        Loss: {metrics[f'class_{i}_loss']:.4f}")
            if phase == 'train':
                logging.info(f"        Gradient Norm: {metrics[f'class_{i}_grad']:.4f}")

class EarlyStoppingWithClassMetrics:
    """Enhanced early stopping with class-wise metric monitoring"""
    
    def __init__(self, patience: int = 10, min_epochs: int = 20,
                 min_delta: float = 0.001, class_threshold: float = 0.1):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.class_threshold = class_threshold
        self.counter = 0
        self.best_loss = float('inf')
        self.best_class_metrics = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, metrics: Dict[str, float], model_state: Dict[str, torch.Tensor],
                 epoch: int) -> bool:
        """
        Check if training should stop based on validation metrics
        
        Args:
            metrics: Dictionary containing validation metrics
            model_state: Current model state dict
            epoch: Current epoch number
        
        Returns:
            bool: Whether to stop training
        """
        current_loss = metrics['loss']
        class_accuracies = [metrics[f'class_{i}_acc'] for i in range(5)]
        
        # Check if we have improved
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_class_metrics = class_accuracies
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model_state.items()}
        else:
            self.counter += 1
        
        # Check class-wise performance
        if min(class_accuracies) < self.class_threshold and epoch >= self.min_epochs:
            logging.warning(f"\nStopping due to poor class performance:")
            for i, acc in enumerate(class_accuracies):
                logging.warning(f"    Class {i} accuracy: {acc:.4f}")
            return True
        
        # Check patience
        if self.counter >= self.patience and epoch >= self.min_epochs:
            logging.info(f"\nEarly stopping triggered after {epoch} epochs")
            return True
        
        return False

class DynamicBatchSampler(Sampler):
    """
    Batch sampler that ensures balanced class representation with dynamic weighting
    """
    def __init__(self, labels, batch_size, drop_last=False):
        from tools.functions import compute_dynamic_importance_factors
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Create class-wise sample indices
        self.class_indices = {
            label: np.where(self.labels == label)[0]
            for label in np.unique(self.labels)
        }
        
        # Calculate dynamic sampling probabilities
        class_counts = {
            label: len(indices)
            for label, indices in self.class_indices.items()
        }
        
        importance_factors = compute_dynamic_importance_factors(class_counts)
        
        # Compute sampling probabilities
        self.sampling_probs = {
            label: importance_factors[label] / len(indices)
            for label, indices in self.class_indices.items()
        }
        
        # Normalize probabilities
        prob_sum = sum(self.sampling_probs.values())
        self.sampling_probs = {
            label: prob / prob_sum
            for label, prob in self.sampling_probs.items()
        }
        
        logging.info("\nDynamic sampling probabilities:")
        for label in sorted(self.sampling_probs.keys()):
            logging.info(f"    Class {label}: {self.sampling_probs[label]:.4f}")
    
    def __iter__(self):
        class_pools = {
            label: indices.copy()
            for label, indices in self.class_indices.items()
        }
        
        while True:
            batch_indices = []
            
            # Sample classes based on dynamic probabilities
            classes_for_batch = np.random.choice(
                list(self.class_indices.keys()),
                size=self.batch_size,
                p=[self.sampling_probs[c] for c in sorted(self.class_indices.keys())]
            )
            
            # Sample instances from each selected class
            for class_label in classes_for_batch:
                pool = class_pools[class_label]
                
                # Reshuffle if needed
                if len(pool) == 0:
                    pool = self.class_indices[class_label].copy()
                    np.random.shuffle(pool)
                    class_pools[class_label] = pool
                
                # Get index and update pool
                idx = pool[0]
                class_pools[class_label] = pool[1:]
                batch_indices.append(idx)
            
            yield batch_indices
    
    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


class DataAugmenter:
    """Enhanced data augmentation pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {
            'time_warp': {
                'probability': 0.8,
                'sigma_range': [0.1, 0.3],
                'knot_range': [4, 6]
            },
            'noise': {
                'probability': 0.5,
                'amplitude_range': [0.01, 0.05]
            },
            'scaling': {
                'probability': 0.3,
                'range': [0.9, 1.1]
            },
            'channel_dropout': {
                'probability': 0.2,
                'max_channels': 1
            }
        }

        def augment_sequence(self, sequence):
            """Apply augmentation to a sequence"""
            from tools.functions import time_warp
            augmented = sequence.clone()
            
            # Apply time warping
            if random.random() < self.config['time_warp']['probability']:
                sigma = random.uniform(*self.config['time_warp']['sigma_range'])
                knot = random.randint(*self.config['time_warp']['knot_range'])
                augmented = torch.from_numpy(
                    time_warp(augmented.numpy(), sigma=sigma, knot=knot)
                )
            
            # Add random noise
            if random.random() < self.config['noise']['probability']:
                amplitude = random.uniform(*self.config['noise']['amplitude_range'])
                noise = torch.randn_like(augmented) * amplitude
                augmented = augmented + noise
            
            # Apply random scaling
            if random.random() < self.config['scaling']['probability']:
                scale = random.uniform(*self.config['scaling']['range'])
                augmented = augmented * scale
            
            # Channel dropout
            if random.random() < self.config['channel_dropout']['probability']:
                n_channels = random.randint(1, self.config['channel_dropout']['max_channels'])
                channels = random.sample(range(augmented.shape[1]), n_channels)
                augmented[:, channels, :] = 0
            
            return augmented
        
        def verify_augmentation(self, original, augmented):
            """Verify augmented data integrity"""
            # Check for invalid values
            assert torch.isfinite(augmented).all(), "Invalid values in augmented data"
            
            # Check signal properties
            orig_mean = original.mean()
            orig_std = original.std()
            aug_mean = augmented.mean()
            aug_std = augmented.std()
            
            mean_diff = abs(orig_mean - aug_mean) / abs(orig_mean)
            std_diff = abs(orig_std - aug_std) / abs(orig_std)
            
            assert mean_diff < 0.5, f"Large change in mean: {mean_diff:.2f}"
            assert std_diff < 0.5, f"Large change in std: {std_diff:.2f}"
            
            return True


class SleepStageLoss(nn.Module):
    def __init__(self, class_weights, temporal_weight=0.2, smoothing=0.1):
        super().__init__()
        self.class_weights = class_weights
        self.temporal_weight = temporal_weight
        self.smoothing = smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=smoothing
        )
    
    def forward(self, outputs, targets, night_indices=None):
        ce_loss = self.ce_loss(outputs, targets)
        
        if night_indices is not None:
            temporal_loss = 0
            for night in torch.unique(night_indices):
                night_mask = night_indices == night
                night_outputs = outputs[night_mask]
                night_targets = targets[night_mask]
                
                if len(night_outputs) > 1:
                    probs = F.softmax(night_outputs, dim=1)
                    temporal_diff = torch.abs(probs[1:] - probs[:-1])
                    transition_weights = self._get_transition_weights(night_targets)
                    temporal_loss += (temporal_diff * transition_weights).mean()
            
            return ce_loss + self.temporal_weight * temporal_loss
        
        return ce_loss
    
    def _get_transition_weights(self, targets):
        weights = torch.ones_like(targets[:-1], dtype=torch.float32)
        
        for i in range(len(targets) - 1):
            if not self._is_valid_transition(targets[i], targets[i+1]):
                weights[i] = 2.0
        
        return weights.to(targets.device)
    
    def _is_valid_transition(self, stage1, stage2):
        valid_transitions = {
            0: {0, 1},
            1: {0, 1, 2},
            2: {1, 2, 3},
            3: {2, 3, 4},
            4: {2, 3, 4}
        }
        return stage2.item() in valid_transitions[stage1.item()]