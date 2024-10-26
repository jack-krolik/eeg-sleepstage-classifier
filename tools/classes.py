import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import Sampler
import numpy as np
from sleepdetector_new import ImprovedSleepdetector
from tools.utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tools.config import CONFIG, device, cuda_manager


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
    


class SleepStageEvaluator:
    def __init__(self, model_dir=CONFIG['model_dir']):
        self.model_dir = model_dir
        self.device = device
        # Map class indices to sleep stage names
        self.class_mapping = {
            0: 'N3 (Deep)',
            1: 'N2 (Light)',
            2: 'N1 (Light)',
            3: 'REM',
            4: 'Wake'
        }
        self.class_names = [self.class_mapping[i] for i in range(5)]
        ensure_dir(os.path.join(model_dir, 'test_results'))
        self.save_outputs = True
        
    def plot_confusion_matrices(self, y_true, y_pred, model_name):
        """Plot both absolute and percentage confusion matrices side by side"""
        # Calculate confusion matrices
        cm_absolute = confusion_matrix(y_true, y_pred)
        cm_percentage = (cm_absolute.astype('float') / 
                        cm_absolute.sum(axis=1)[:, np.newaxis] * 100)
        
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
        # Calculate overall metrics
        accuracy = (y_pred == y_true).mean() * 100
        f1_macro = f1_score(y_true, y_pred, average='macro') * 100
        f1_weighted = f1_score(y_true, y_pred, average='weighted') * 100
        
        # Generate detailed classification report
        report = classification_report(y_true, y_pred, 
                                    target_names=self.class_names, 
                                    output_dict=True)
        
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