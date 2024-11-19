import torch
import torch.nn as nn
from typing import List, Dict, Optional
from torch.utils.data import Sampler
import numpy as np
from sleepdetector_new import ImprovedSleepdetector
# from sleepdetector_dup import ImprovedSleepdetector
# from sleepdetector_clean import ImprovedSleepdetector
# from sleepdetector import ImprovedSleepdetector
from tools_clean.utils_clean import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tools_clean.config_clean import CONFIG, device, cuda_manager
import scipy.io as sio
from scipy.signal import welch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
import torch.nn.functional as F
from tqdm import tqdm

class EnsembleModel(nn.Module):
    """Ensemble of sleep detection models"""
    def __init__(self, model_params: Dict, n_models: int = 3):
        super().__init__()
        self.models = nn.ModuleList([
            ImprovedSleepdetector(**model_params) 
            for _ in range(n_models)
        ])
        self.apply(self._init_weights)
    
    def update_normalization_stats(self, dataset_medians, dataset_iqrs):
        """Update normalization statistics for all models in ensemble"""
        for model in self.models:
            model.update_normalization_stats(dataset_medians, dataset_iqrs)
    
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

    # In EnsembleModel.forward
    def forward(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging predictions from all models"""
        if spectral_features.shape[1] != 16:
            raise ValueError(f"Expected 16 spectral features, got {spectral_features.shape[1]}")
            
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

# class BalancedBatchSampler(Sampler):
#     """Sampler that ensures balanced class representation in each batch"""
#     def __init__(self, labels: np.ndarray, batch_size: int):
#         self.labels = labels
#         self.batch_size = batch_size
#         self.label_to_indices = {
#             label: np.where(labels == label)[0] 
#             for label in set(labels)
#         }
#         self.used_label_indices_count = {
#             label: 0 for label in set(labels)
#         }
#         self.count = 0
#         self.n_classes = len(set(labels))
#         self.n_samples = len(labels)
        
#     def __iter__(self):
#         self.count = 0
#         while self.count + self.batch_size < self.n_samples:
#             classes = list(self.label_to_indices.keys())
#             indices = []
#             for class_ in classes:
#                 class_indices = self.label_to_indices[class_]
#                 start_idx = self.used_label_indices_count[class_]
#                 samples_per_class = self.batch_size // self.n_classes
                
#                 # Get indices for this class
#                 if start_idx + samples_per_class > len(class_indices):
#                     np.random.shuffle(class_indices)
#                     self.used_label_indices_count[class_] = 0
#                     start_idx = 0
                    
#                 selected_indices = class_indices[
#                     start_idx:start_idx + samples_per_class
#                 ]
#                 indices.extend(selected_indices)
#                 self.used_label_indices_count[class_] += samples_per_class
                
#             yield indices
#             self.count += self.batch_size

#     def __len__(self) -> int:
#         return self.n_samples // self.batch_size

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
    
    def evaluate_model(self, model, X, X_spectral, y, model_name, batch_size=32):
        """Evaluate a single model with comprehensive metrics and visualizations"""
        print(f"\nEvaluating {model_name}...")
        model.eval()
        
        try:
            with torch.no_grad():
                if not torch.isfinite(X_spectral).all():
                    logging.warning("Invalid values found in spectral features. Cleaning...")
                    X_spectral = torch.nan_to_num(X_spectral, nan=0.0, posinf=1e6, neginf=-1e6)

                # Process in batches
                all_predictions = []
                n_samples = len(y)
                
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    
                    # Move batch to device
                    batch_X = X[i:batch_end].to(self.device)
                    batch_X_spectral = X_spectral[i:batch_end].to(self.device)
                    
                    # Generate predictions
                    outputs = model(batch_X, batch_X_spectral)
                    predictions = outputs.argmax(dim=1).cpu().numpy()
                    all_predictions.extend(predictions)
                    
                    # Clear GPU memory
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                
                # Convert to numpy array
                predictions = np.array(all_predictions)
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
            raise


class SleepDataManager:
    """Integrated class for sleep data management and cross-validation"""
    def __init__(self, data_files, val_ratio=0.2, seed=42):
        self.data_files = data_files
        self.val_ratio = val_ratio
        self.seed = seed
        self.class_weights = None
        self.scaler = None
        
    def load_and_preprocess(self):
        """Load and preprocess all data files"""
        all_data = {
            'x': [], 'x_spectral': [], 
            'y': [], 'night_idx': []
        }
        
        total_files = len(self.data_files)
        successful_loads = 0
        
        # logging.info(f"Starting to load {total_files} data files...")
        with tqdm(total=total_files, desc="Loading Data Files", unit="file") as pbar:

            for night_idx, file_path in enumerate(self.data_files):
                try:
                    # Load data
                    logging.info(f"Loading file {night_idx + 1}/{total_files}: {os.path.basename(file_path)}")
                    mat_data = sio.loadmat(file_path)
                    
                    # Stack the signals
                    x = np.stack((
                        mat_data['sig1'], mat_data['sig2'],
                        mat_data['sig3'], mat_data['sig4']
                    ), axis=1)
                    y = mat_data['labels'].flatten()
                    
                    # Log shapes before filtering
                    # logging.info(f"Initial shapes - X: {x.shape}, y: {y.shape}")
                    
                    # Filter valid data
                    valid_mask = y != -1
                    x = x[valid_mask]
                    y = y[valid_mask]
                    
                    # logging.info(f"After filtering - X: {x.shape}, y: {y.shape}")
                    
                    # Extract spectral features
                    # logging.info("Extracting spectral features...")
                    try:
                        x_spectral = np.array([
                            self._extract_spectral_features(epoch) 
                            for epoch in x
                        ])
                        logging.info(f"Spectral features shape: {x_spectral.shape}")
                        
                    except Exception as e:
                        logging.error(f"Error in spectral feature extraction: {str(e)}")
                        raise
                    
                    # Store processed data
                    all_data['x'].append(torch.FloatTensor(x))
                    all_data['x_spectral'].append(torch.FloatTensor(x_spectral))
                    all_data['y'].append(torch.LongTensor(y))
                    all_data['night_idx'].extend([night_idx] * len(y))
                    
                    successful_loads += 1
                    logging.info(f"Successfully loaded night {night_idx + 1}: {len(y)} samples")
                    
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {str(e)}")
                    logging.error("Skipping this file and continuing...")
                    continue

                pbar.update(1)
            
        if successful_loads == 0:
            raise RuntimeError("No files were successfully loaded!")
        
        logging.info(f"\nSuccessfully loaded {successful_loads} out of {total_files} files")
        
        try:
            # Combine all nights
            logging.info("Combining data from all nights...")
            self.data = {
                'x': torch.cat(all_data['x']),
                'x_spectral': torch.cat(all_data['x_spectral']),
                'y': torch.cat(all_data['y']),
                'night_idx': torch.tensor(all_data['night_idx'])
            }
            
            # Log final data shapes
            logging.info(f"\nFinal data shapes:")
            logging.info(f"X: {self.data['x'].shape}")
            logging.info(f"X_spectral: {self.data['x_spectral'].shape}")
            logging.info(f"y: {self.data['y'].shape}")
            logging.info(f"night_indices: {self.data['night_idx'].shape}")
            
            # Update class weights
            self._update_class_weights()
            return self
            
        except Exception as e:
            logging.error(f"Error combining data: {str(e)}")
            raise

    def _extract_spectral_features(self, x):
        """Extract spectral features to match model input size (4 features per channel)"""
        features = []
        for channel in range(x.shape[0]):
            channel_data = x[channel]
            f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
            
            # Calculate only the basic frequency bands (4 per channel)
            delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
            theta = np.sum(psd[(f > 4) & (f <= 8)])
            alpha = np.sum(psd[(f > 8) & (f <= 13)])
            beta = np.sum(psd[(f > 13) & (f <= 30)])
            
            # Add only the 4 basic features
            features.extend([delta, theta, alpha, beta])
        
        features = np.array(features)
        assert len(features) == 16, f"Expected 16 spectral features, got {len(features)}"
        return features
    
    def _update_class_weights(self):
        """Calculate class weights with improved scaling"""
        class_counts = torch.bincount(self.data['y'])
        total = class_counts.sum()
        
        # Use effective number of samples
        beta = 0.9999
        effective_num = (1.0 - torch.pow(beta, class_counts.float())) / (1.0 - beta)
        weights = (1.0 - beta) / effective_num
        
        # Add smoothing and normalization
        weights = weights + 1e-6  # Smoothing factor
        weights = weights / weights.sum()
        
        # Store both raw and normalized weights
        self.class_weights_raw = weights
        self.class_weights = weights

        # Log class weights for monitoring
        logging.info("Updated class weights:")
        for i, weight in enumerate(weights):
            logging.info(f"Class {i} ({SLEEP_STAGES[i]}): count={class_counts[i]}, weight={weight:.4f}")
    
    def create_cross_validator(self, n_folds=5):
        """Create a night-aware cross-validator"""
        return NightBasedCrossValidator(
            night_indices=self.data['night_idx'],
            n_folds=min(n_folds, len(self.data_files)),
            val_ratio=self.val_ratio,
            seed=self.seed
        )
    def get_loader(self, indices, batch_size=32, is_training=True, verify_balance=False):
        """Memory-efficient data loader creation"""
        class DatasetWithIndices(TensorDataset):
            def __init__(self, data_dict, indices):
                self.data_dict = data_dict.copy()  # Create a copy of the dictionary
                self.indices = indices
                # Calculate fold-specific class weights
                fold_labels = self.data_dict['y'][indices]
                class_counts = torch.bincount(fold_labels)
                beta = 0.9999
                effective_num = (1.0 - torch.pow(beta, class_counts.float())) / (1.0 - beta)
                fold_weights = (1.0 - beta) / effective_num
                self.data_dict['class_weights'] = fold_weights / fold_weights.sum()
                # self.data_dict['class_weights'] = data_dict.class_weights  # Add class weights
                
                # Validate dimensions of first sample
                x = self.data_dict['x'][indices[0]]
                x_spectral = self.data_dict['x_spectral'][indices[0]]
                if x_spectral.shape[0] != 16:
                    logging.error(f"Incorrect spectral features dimension: {x_spectral.shape[0]}, expected 16")
                
            def __getitem__(self, idx):
                true_idx = self.indices[idx]
                return (
                    self.data_dict['x'][true_idx],
                    self.data_dict['x_spectral'][true_idx],
                    self.data_dict['y'][true_idx]
                )
                
            def __len__(self):
                return len(self.indices)
        
        # Create dataset
        dataset = DatasetWithIndices(self.data, indices)
        
        if is_training:
            weights = self.class_weights[self.data['y'][indices]]
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )

        # Add optional balance verification
        if verify_balance and is_training:
            self.verify_loader_balance(loader)

        return loader
    def _augment_minorities(self, x, x_spectral, y, threshold_ratio=0.5):
        """Augment minority classes using time warping"""
        from tools_clean.functions_clean import time_warp
        class_counts = torch.bincount(y)
        max_count = class_counts.max()
        augmented_data = {'x': [x], 'x_spectral': [x_spectral], 'y': [y]}
        
        for class_idx, count in enumerate(class_counts):
            if count / max_count < threshold_ratio:
                class_mask = y == class_idx
                class_x = x[class_mask]
                class_x_spectral = x_spectral[class_mask]
                
                # Calculate number of augmentations needed
                n_aug = int(max_count * threshold_ratio) - count
                if n_aug <= 0:
                    continue
                
                # Apply time warping augmentation
                aug_x = torch.from_numpy(
                    time_warp(
                        class_x.numpy(),
                        sigma=0.2,
                        knot=4
                    )
                )
                
                # Add augmented data
                augmented_data['x'].append(aug_x)
                augmented_data['x_spectral'].append(class_x_spectral.repeat(n_aug, 1))
                augmented_data['y'].append(torch.full((n_aug,), class_idx))
        
        return (torch.cat(augmented_data['x']),
                torch.cat(augmented_data['x_spectral']),
                torch.cat(augmented_data['y']))
    def verify_loader_balance(self, loader, num_batches=5):
        """Verify class balance in data loader without disrupting existing flow"""
        if not hasattr(self, 'data'):
            logging.warning("No data loaded yet, cannot verify balance")
            return
        
        logging.info("\nVerifying data loader balance:")
        
        # Log overall distribution first
        total_dist = Counter(self.data['y'].numpy())
        total = sum(total_dist.values())
        logging.info("\nOverall dataset distribution:")
        for class_idx in sorted(SLEEP_STAGES.keys()):
            count = total_dist.get(class_idx, 0)
            pct = (count / total) * 100
            logging.info(f"    {SLEEP_STAGES[class_idx]}: {count:5d} ({pct:5.1f}%)")

        # Check a few batches
        batch_distributions = []
        for i, (_, _, y) in enumerate(loader):
            if i >= num_batches:
                break
            batch_distributions.append(Counter(y.numpy()))
        
        # Calculate and log average batch distribution
        avg_dist = Counter()
        for dist in batch_distributions:
            avg_dist.update(dist)
        
        for k in avg_dist:
            avg_dist[k] /= len(batch_distributions)
        
        logging.info(f"\nAverage batch distribution ({num_batches} batches):")
        for class_idx in sorted(SLEEP_STAGES.keys()):
            avg = avg_dist.get(class_idx, 0)
            batch_size = sum(avg_dist.values())
            pct = (avg / batch_size) * 100
            logging.info(f"    {SLEEP_STAGES[class_idx]}: {avg:.1f} ({pct:.1f}%)")

        return avg_dist
    
class NightBasedCrossValidator:
    """Cross-validator that respects night boundaries and maintains class distribution"""
    def __init__(self, night_indices, n_folds=5, val_ratio=0.2, seed=42):
        self.night_indices = night_indices
        self.n_folds = n_folds
        self.val_ratio = val_ratio
        self.rng = np.random.RandomState(seed)
        # Define minimum ratios per class based on actual data distribution
        self.min_class_ratios = None

    def calculate_dynamic_class_ratios(self, y):
        night_unique = torch.unique(self.night_indices)
        class_ratios_per_night = []
        
        for night in night_unique:
            night_mask = self.night_indices == night
            night_labels = y[night_mask]
            class_counts = torch.bincount(night_labels)
            total = len(night_labels)
            # Convert to ratio but ensure at least 1/total ratio for each class
            min_ratio = 1.0 / total
            class_ratios = torch.maximum(class_counts.float() / total, torch.tensor(min_ratio))
            class_ratios_per_night.append(class_ratios)
        
        # Get minimum ratios across nights
        min_ratios = torch.stack(class_ratios_per_night).min(dim=0)[0]
        min_ratios = min_ratios * 0.8  # Safety buffer
        
        return min_ratios.tolist()
    
    def split(self, y):
        """Generate train/val splits while maintaining night boundaries"""
        unique_nights = torch.unique(self.night_indices)
        n_nights = len(unique_nights)
        
        self.min_class_ratios = self.calculate_dynamic_class_ratios(y)
        logging.info(f"Calculated dynamic class ratios: {self.min_class_ratios}")
        
        logging.info(f"Total number of nights: {n_nights}")
        splits = []
        max_attempts = 10  # Maximum number of shuffle attempts per fold
        
        for fold in range(self.n_folds):
            fold_valid = False
            for attempt in range(max_attempts):
                # Shuffle nights
                night_order = self.rng.permutation(n_nights)
                
                # Select validation nights
                val_size = max(1, int(n_nights * self.val_ratio))
                start_idx = (fold * val_size) % n_nights
                val_nights = night_order[start_idx:start_idx + val_size]
                
                logging.info(f"Fold {fold + 1}, Attempt {attempt + 1}: Validation nights: {val_nights}")
                
                # Create masks
                val_mask = torch.tensor([
                    night in val_nights for night in self.night_indices
                ])
                train_mask = ~val_mask
                
                # Verify class distribution
                if self._verify_split(y[train_mask], y[val_mask]):
                    splits.append((
                        torch.where(train_mask)[0],
                        torch.where(val_mask)[0]
                    ))
                    logging.info(f"Split {fold + 1} verified successfully after {attempt + 1} attempts")
                    fold_valid = True
                    break
            
            if not fold_valid:
                logging.warning(f"Could not create valid split for fold {fold + 1} after {max_attempts} attempts")
        
        return splits
    
    def _verify_split(self, y_train, y_val):
        """Verify that split maintains reasonable class distribution"""
        train_counts = torch.bincount(y_train)
        val_counts = torch.bincount(y_val)
        
        # Check if all classes are represented
        if len(train_counts) != len(val_counts):
            logging.warning(f"Mismatched number of classes: Train has {len(train_counts)}, Val has {len(val_counts)}")
            return False
            
        # Check class ratios
        train_ratios = train_counts.float() / len(y_train)
        val_ratios = val_counts.float() / len(y_val)
        
        # Log ratios
        for i in range(len(train_ratios)):
            logging.info(f"Class {i} - Train ratio: {train_ratios[i]:.3f}, Val ratio: {val_ratios[i]:.3f}")
            min_ratio = self.min_class_ratios[i]
            logging.info(f"  Required min ratio: {min_ratio:.3f}")
        
        # Verify minimum representation using class-specific thresholds
        train_valid = all(
            train_ratios[i] >= self.min_class_ratios[i] 
            for i in range(len(train_ratios))
        )
        val_valid = all(
            val_ratios[i] >= self.min_class_ratios[i]
            for i in range(len(val_ratios))
        )
        
        if not train_valid:
            logging.warning("Train set failed minimum ratio check")
            # Log which classes failed
            for i in range(len(train_ratios)):
                if train_ratios[i] < self.min_class_ratios[i]:
                    logging.warning(f"  Class {i} train ratio too low: {train_ratios[i]:.3f} < {self.min_class_ratios[i]:.3f}")
        
        if not val_valid:
            logging.warning("Validation set failed minimum ratio check")
            # Log which classes failed
            for i in range(len(val_ratios)):
                if val_ratios[i] < self.min_class_ratios[i]:
                    logging.warning(f"  Class {i} val ratio too low: {val_ratios[i]:.3f} < {self.min_class_ratios[i]:.3f}")
        
        return train_valid and val_valid

class EarlyStoppingWithMetrics:
    """Enhanced early stopping with multiple metric monitoring"""
    def __init__(self, patience=7, min_epochs=20, min_delta=0.001):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.counter = 0
        self.best_metrics = {
            'val_f1': float('-inf'),
            'val_loss': float('inf')
        }
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, metrics, model_state, epoch):
        improved = False
        
        # Check F1 score improvement
        if metrics['f1_macro'] > self.best_metrics['val_f1'] + self.min_delta:
            self.best_metrics['val_f1'] = metrics['f1_macro']
            improved = True
        
        # Check loss improvement
        if metrics['loss'] < self.best_metrics['val_loss'] - self.min_delta:
            self.best_metrics['val_loss'] = metrics['loss']
            improved = True
        
        if improved:
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
        else:
            self.counter += 1
        
        # Only stop after minimum epochs
        if self.counter >= self.patience and epoch >= self.min_epochs:
            self.early_stop = True
        
        return self.early_stop


class BalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



class SleepPaperDataManager:
    """Integrated class for sleep data management and cross-validation"""
    
    def __init__(self, x_data, y_data, val_ratio=0.2, seed=42):
        """
        Initialize with preloaded data.
        
        Args:
            x_data (numpy array): Input features of shape (num_samples, channels, length).
            y_data (numpy array): Target labels of shape (num_samples,).
            val_ratio (float): Validation split ratio.
            seed (int): Seed for reproducibility.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.val_ratio = val_ratio
        self.seed = seed
        self.class_weights = None

    def load_and_preprocess(self):
        """Load and preprocess provided data"""
        all_data = {
            'x': [],
            'y': [],
            'night_idx': []
        }

        # Verify data dimensions
        if len(self.x_data) != len(self.y_data):
            raise ValueError("Mismatch between x_data and y_data lengths")

        # Process data
        logging.info("Starting data preprocessing...")
        x = torch.FloatTensor(self.x_data)
        y = torch.LongTensor(self.y_data)

        # Filter valid data (exclude entries where y == -1)
        valid_mask = y != -1
        x = x[valid_mask]
        y = y[valid_mask]

        try:
            x_spectral = np.array([
                self._extract_spectral_features(epoch) 
                for epoch in x
            ])
            logging.info(f"Spectral features shape: {x_spectral.shape}")
            
        except Exception as e:
            logging.error(f"Error in spectral feature extraction: {str(e)}")
            raise
        # Store processed data in the structure
        all_data['x'].append(x)
        all_data['x_spectral'].append(torch.FloatTensor(x_spectral))
        all_data['y'].append(y)
        all_data['night_idx'].extend([0] * len(y))  # Using a single night index if all data is from one set

        # Combine processed data
        logging.info("Combining processed data...")
        self.data = {
            'x': torch.cat(all_data['x']),
            'y': torch.cat(all_data['y']),
            'night_idx': torch.tensor(all_data['night_idx'])
        }

        # Log data shapes
        logging.info(f"\nData shapes:")
        logging.info(f"X: {self.data['x'].shape}")
        logging.info(f"y: {self.data['y'].shape}")
        logging.info(f"night_indices: {self.data['night_idx'].shape}")
        
        # Update class weights
        self._update_class_weights()
        return self
    
    def _update_class_weights(self):
        """Calculate class weights based on class distribution"""
        class_counts = torch.bincount(self.data['y'])
        total = class_counts.sum()

        # Use effective number of samples for weighting
        beta = 0.9999
        effective_num = (1.0 - torch.pow(beta, class_counts.float())) / (1.0 - beta)
        weights = (1.0 - beta) / effective_num

        # Normalize weights
        weights = weights + 1e-6  # Smoothing factor
        weights = weights / weights.sum()

        # Store weights
        self.class_weights = weights
        logging.info("Updated class weights:")
        for i, weight in enumerate(weights):
            logging.info(f"Class {i}: count={class_counts[i]}, weight={weight:.4f}")

    def _extract_spectral_features(self, x):
        """Extract spectral features to match model input size (4 features per channel)"""
        features = []
        for channel in range(x.shape[0]):
            channel_data = x[channel]
            f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
            
            # Calculate only the basic frequency bands (4 per channel)
            delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
            theta = np.sum(psd[(f > 4) & (f <= 8)])
            alpha = np.sum(psd[(f > 8) & (f <= 13)])
            beta = np.sum(psd[(f > 13) & (f <= 30)])
            
            # Add only the 4 basic features
            features.extend([delta, theta, alpha, beta])

    def get_loader(self, indices, batch_size=32, is_training=True, verify_balance=False):
        """Memory-efficient data loader creation"""
        class DatasetWithIndices(TensorDataset):
            def __init__(self, data_dict, indices):
                self.data_dict = data_dict.copy()  # Create a copy of the dictionary
                self.indices = indices
                # Calculate fold-specific class weights
                fold_labels = self.data_dict['y'][indices]
                class_counts = torch.bincount(fold_labels)
                beta = 0.9999
                effective_num = (1.0 - torch.pow(beta, class_counts.float())) / (1.0 - beta)
                fold_weights = (1.0 - beta) / effective_num
                self.data_dict['class_weights'] = fold_weights / fold_weights.sum()
                # self.data_dict['class_weights'] = data_dict.class_weights  # Add class weights
                
                # Validate dimensions of first sample
                x = self.data_dict['x'][indices[0]]
                x_spectral = self.data_dict['x_spectral'][indices[0]]
                if x_spectral.shape[0] != 16:
                    logging.error(f"Incorrect spectral features dimension: {x_spectral.shape[0]}, expected 16")
                
            def __getitem__(self, idx):
                true_idx = self.indices[idx]
                return (
                    self.data_dict['x'][true_idx],
                    self.data_dict['x_spectral'][true_idx],
                    self.data_dict['y'][true_idx]
                )
                
            def __len__(self):
                return len(self.indices)
        
        # Create dataset
        dataset = DatasetWithIndices(self.data, indices)
        
        if is_training:
            weights = self.class_weights[self.data['y'][indices]]
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )

        # Add optional balance verification
        if verify_balance and is_training:
            self.verify_loader_balance(loader)

        return loader

    def _augment_minorities(self, x, x_spectral, y, threshold_ratio=0.5):
        """Augment minority classes using time warping"""
        from tools_clean.functions_clean import time_warp
        class_counts = torch.bincount(y)
        max_count = class_counts.max()
        augmented_data = {'x': [x], 'x_spectral': [x_spectral], 'y': [y]}
        
        for class_idx, count in enumerate(class_counts):
            if count / max_count < threshold_ratio:
                class_mask = y == class_idx
                class_x = x[class_mask]
                class_x_spectral = x_spectral[class_mask]
                
                # Calculate number of augmentations needed
                n_aug = int(max_count * threshold_ratio) - count
                if n_aug <= 0:
                    continue
                
                # Apply time warping augmentation
                aug_x = torch.from_numpy(
                    time_warp(
                        class_x.numpy(),
                        sigma=0.2,
                        knot=4
                    )
                )
                
                # Add augmented data
                augmented_data['x'].append(aug_x)
                augmented_data['x_spectral'].append(class_x_spectral.repeat(n_aug, 1))
                augmented_data['y'].append(torch.full((n_aug,), class_idx))
        
        return (torch.cat(augmented_data['x']),
                torch.cat(augmented_data['x_spectral']),
                torch.cat(augmented_data['y']))
    def verify_loader_balance(self, loader, num_batches=5):
        """Verify class balance in data loader without disrupting existing flow"""
        if not hasattr(self, 'data'):
            logging.warning("No data loaded yet, cannot verify balance")
            return
        
        logging.info("\nVerifying data loader balance:")
        
        # Log overall distribution first
        total_dist = Counter(self.data['y'].numpy())
        total = sum(total_dist.values())
        logging.info("\nOverall dataset distribution:")
        for class_idx in sorted(SLEEP_STAGES.keys()):
            count = total_dist.get(class_idx, 0)
            pct = (count / total) * 100
            logging.info(f"    {SLEEP_STAGES[class_idx]}: {count:5d} ({pct:5.1f}%)")

        # Check a few batches
        batch_distributions = []
        for i, (_, _, y) in enumerate(loader):
            if i >= num_batches:
                break
            batch_distributions.append(Counter(y.numpy()))
        
        # Calculate and log average batch distribution
        avg_dist = Counter()
        for dist in batch_distributions:
            avg_dist.update(dist)
        
        for k in avg_dist:
            avg_dist[k] /= len(batch_distributions)
        
        logging.info(f"\nAverage batch distribution ({num_batches} batches):")
        for class_idx in sorted(SLEEP_STAGES.keys()):
            avg = avg_dist.get(class_idx, 0)
            batch_size = sum(avg_dist.values())
            pct = (avg / batch_size) * 100
            logging.info(f"    {SLEEP_STAGES[class_idx]}: {avg:.1f} ({pct:.1f}%)")

        return avg_dist