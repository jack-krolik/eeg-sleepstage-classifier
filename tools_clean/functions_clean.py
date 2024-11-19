import torch
import numpy as np
import scipy.io as sio
import logging
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tools_clean.classes_clean import *
from tools_clean.config_clean import *
# from tools_clean.utils_clean import *
from tools_clean.config_clean import get_cuda_device
import random
import math
import os
from optuna import create_study, trial
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import defaultdict
import json
import gc

def set_seed(seed=42):
    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device.type == 'cuda':
        try:
            torch.cuda.manual_seed(seed)
            print(f"CUDA seed set on device {torch.cuda.get_device_name(device)}")
        except RuntimeError as e:
            print(f"Failed to set CUDA seed: {e}")
    else:
        print("CUDA not available, seed set only for CPU")


# def generate_random_params():
#     return {
#         'model_params': {
#             'n_filters': [random.choice(CONFIG['model_params']['tuning_ranges']['n_filters'])],
#             'lstm_hidden': random.choice(CONFIG['model_params']['tuning_ranges']['lstm_hidden']),
#             'lstm_layers': random.choice(CONFIG['model_params']['tuning_ranges']['lstm_layers']),
#             'dropout': random.uniform(*CONFIG['model_params']['tuning_ranges']['dropout'])
#         },
#         'train_params': {
#             'lr': random.uniform(*CONFIG['train_params']['tuning_ranges']['lr']),
#             'batch_size': random.choice(CONFIG['train_params']['tuning_ranges']['batch_size']),
#             'num_epochs': random.choice(CONFIG['train_params']['tuning_ranges']['num_epochs']),
#             'patience': random.choice(CONFIG['train_params']['tuning_ranges']['patience'])
#         }
#     }


def initialize_model_with_gpu_check(model_params, device):
    try:
        if device.type == 'cuda':
            # Check memory availability before model initialization
            with torch.cuda.device(device):
                torch.cuda.empty_cache()  # Clear the cache before measuring memory
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                free_memory = total_memory - allocated_memory

                # Log memory before initialization
                logging.info(f"Free GPU memory before initialization: {free_memory / 1e6} MB")

                estimated_size = sum(p.numel() * p.element_size() for p in EnsembleModel(model_params).parameters())

                if estimated_size * 2 > free_memory:  # Less aggressive factor for safety
                    logging.warning("Insufficient GPU memory, retrying with reduced batch size")
                    raise MemoryError("GPU out of memory")

        # Initialize model on device
        model = EnsembleModel(model_params).to(device)
        model.apply(model._init_weights)
        return model, device

    except MemoryError as e:
        logging.error(f"MemoryError: {str(e)} - Trying to recover")
        torch.cuda.empty_cache()  # Clear memory
        return model.cpu(), torch.device('cpu')

    except RuntimeError as e:
        logging.error(f"RuntimeError: {str(e)}")
        device = torch.device('cpu')
        model = EnsembleModel(model_params).to(device)
        model.apply(model._init_weights)
        return model, device


def create_data_loaders(X, X_spectral, y, batch_size, is_train=True):
    dataset = TensorDataset(X, X_spectral, y)
    
    if is_train:
        # Calculate sample weights for balanced sampling
        class_counts = torch.bincount(y)
        class_weights = 1. / class_counts.float()
        class_weights = torch.sqrt(class_weights)  # Reduce extreme weights
        class_weights = class_weights / class_weights.sum()
        
        # Create sample weights
        sample_weights = class_weights[y]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y),
            replacement=True
        )
        
        return DataLoader(dataset, 
                         batch_size=batch_size,
                         sampler=sampler,
                         num_workers=4,
                         pin_memory=True)
    else:
        return DataLoader(dataset, 
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True)


# def extract_spectral_features(x):
#     """Enhanced spectral feature extraction"""
#     features = []
#     for channel in range(x.shape[0]):
#         channel_data = x[channel].cpu().numpy()
#         f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
        
#         # Traditional frequency bands
#         delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
#         theta = np.sum(psd[(f > 4) & (f <= 8)])
#         alpha = np.sum(psd[(f > 8) & (f <= 13)])
#         beta = np.sum(psd[(f > 13) & (f <= 30)])
#         gamma = np.sum(psd[(f > 30) & (f <= 50)])  # Added gamma band
        
#         # Band ratios (useful for sleep stage classification)
#         theta_alpha_ratio = theta / (alpha + 1e-6)
#         delta_beta_ratio = delta / (beta + 1e-6)
        
#         # Spectral edge frequency (frequency below which 95% of power exists)
#         cumsum = np.cumsum(psd)
#         spectral_edge = f[np.where(cumsum >= 0.95 * cumsum[-1])[0][0]]
        
#         features.extend([
#             delta, theta, alpha, beta, gamma,
#             theta_alpha_ratio, delta_beta_ratio,
#             spectral_edge
#         ])
#     return np.array(features)


def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1), pat[:, dim]).T
    return ret



def improved_oversample(X, X_spectral, y, max_ratio=3.0, min_samples=50):
    """
    Enhanced oversampling strategy with controlled class ratios
    """
    class_counts = Counter(y)
    median_count = np.median(list(class_counts.values()))
    
    # Calculate target counts more intelligently
    target_counts = {}
    total_samples = len(y)
    min_class_count = min(class_counts.values())
    
    for class_label, count in class_counts.items():
        if count < median_count:
            # More sophisticated target calculation
            ratio = min(max_ratio, np.sqrt(median_count/count))  # Square root scaling
            target_count = min(
                int(count * ratio),
                int(total_samples * 0.4)  # Prevent any class from dominating
            )
            target_count = max(target_count, min_samples)
        else:
            target_count = count
        target_counts[class_label] = target_count

    # Enhanced time warping parameters for minority classes
    time_warp_params = {
        'sigma': lambda count: 0.4 * (min_class_count / count),  # Adaptive warping
        'knot': lambda count: max(4, min(8, int(np.log2(count))))  # Adaptive complexity
    }
    
    oversampled_X = []
    oversampled_X_spectral = []
    oversampled_y = []
    
    for class_label in class_counts:
        class_indices = np.where(y == class_label)[0]
        current_count = len(class_indices)
        target_count = target_counts[class_label]
        
        # Add original samples
        oversampled_X.append(X[class_indices])
        oversampled_X_spectral.append(X_spectral[class_indices])
        oversampled_y.extend([class_label] * current_count)
        
        if current_count < target_count:
            n_oversample = target_count - current_count
            
            if current_count < min_samples:
                # Use more aggressive augmentation for very small classes
                sigma = time_warp_params['sigma'](current_count)
                knot = time_warp_params['knot'](current_count)
                
                aug_samples = []
                aug_spectral = []
                
                for _ in range(n_oversample):
                    # Randomly combine features from same class
                    idx1, idx2 = np.random.choice(class_indices, 2)
                    alpha = np.random.beta(0.4, 0.4)  # Bimodal distribution
                    
                    # Interpolate between samples
                    mixed_sample = alpha * X[idx1] + (1 - alpha) * X[idx2]
                    warped = time_warp(
                        mixed_sample.unsqueeze(0),
                        sigma=sigma,
                        knot=knot
                    )
                    aug_samples.append(torch.from_numpy(warped).squeeze(0))
                    
                    # Interpolate spectral features
                    mixed_spectral = alpha * X_spectral[idx1] + (1 - alpha) * X_spectral[idx2]
                    aug_spectral.append(mixed_spectral)
                
                oversampled_X.append(torch.stack(aug_samples))
                oversampled_X_spectral.append(torch.stack(aug_spectral))
                oversampled_y.extend([class_label] * n_oversample)
            else:
                # Use simpler oversampling for larger minority classes
                oversample_idx = np.random.choice(class_indices, size=n_oversample, replace=True)
                oversampled_X.append(X[oversample_idx])
                oversampled_X_spectral.append(X_spectral[oversample_idx])
                oversampled_y.extend([class_label] * n_oversample)
    
    return (torch.cat(oversampled_X), 
            torch.cat(oversampled_X_spectral), 
            torch.tensor(oversampled_y))

def get_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_lr(model, train_loader, val_loader, optimizer, criterion, device, num_iter=100, start_lr=None, end_lr=1):
    """
    Find the optimal learning rate using the learning rate range test.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: The optimizer to use
        criterion: The loss function
        device: The device to run on
        num_iter: Number of iterations for the test
        start_lr: Starting learning rate (if None, uses 1e-8)
        end_lr: Ending learning rate
        
    Returns:
        float: The suggested learning rate
    """
    logging.info("Starting learning rate finder...")
    
    # Initialize
    if start_lr is None:
        start_lr = 1e-8
    
    model.train()
    update_step = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    optimizer.param_groups[0]["lr"] = lr
    
    # Initialize tracking variables
    smoothed_loss = 0
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []
    
    progress_bar = tqdm(range(num_iter), desc="Finding best LR")
    
    try:
        for i in progress_bar:
            # Get batch of data
            try:
                inputs, spectral_features, targets = next(iter(train_loader))
            except StopIteration:
                train_loader = iter(train_loader)
                inputs, spectral_features, targets = next(train_loader)
            
            # Move data to device
            inputs = inputs.to(device)
            spectral_features = spectral_features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, spectral_features)

            # Add logging for focal loss components
            with torch.no_grad():
                ce_loss = F.cross_entropy(outputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** criterion.gamma
                
                logging.debug(f"LR: {lr:.2e}")
                logging.debug(f"CE Loss: {ce_loss.mean():.4f}")
                logging.debug(f"Focal Weight Mean: {focal_weight.mean():.4f}")
                
            loss = criterion(outputs, targets)
            
            # Compute smoothed loss
            if batch_num == 0:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss.item()
            
            # Record best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Stop if loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                logging.info(f"Loss is exploding, stopping early at lr={lr:.2e}")
                break
            
            # Stop if loss has been increasing
            if len(losses) > 5 and smoothed_loss > min(losses[-5:]):
                logging.info(f"Loss has been increasing, stopping at lr={lr:.2e}")
                break
            
            # Store values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            lr *= update_step
            optimizer.param_groups[0]["lr"] = lr
            
            progress_bar.set_postfix({
                'loss': f'{smoothed_loss:.4f}',
                'lr': f'{lr:.2e}'
            })
            batch_num += 1

        # Validate model at different learning rates
        val_losses = []
        model.eval()
        with torch.no_grad():
            for lr_idx, (lr, loss) in enumerate(zip(log_lrs, losses)):
                optimizer.param_groups[0]["lr"] = 10**lr
                val_loss = 0
                for inputs, spectral_features, targets in val_loader:
                    inputs = inputs.to(device)
                    spectral_features = spectral_features.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs, spectral_features)
                    val_loss += criterion(outputs, targets).item()
                val_losses.append(val_loss / len(val_loader))

        # Create and save the learning rate plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(log_lrs, losses)
        plt.xlabel("Log Learning Rate")
        plt.ylabel("Training Loss")
        plt.title("Learning Rate vs. Training Loss")
        
        plt.subplot(1, 2, 2)
        plt.plot(log_lrs, val_losses)
        plt.xlabel("Log Learning Rate")
        plt.ylabel("Validation Loss")
        plt.title("Learning Rate vs. Validation Loss")
        
        plt.tight_layout()
        
        # Save plot using new config structure
        lr_plot_path = os.path.join(CONFIG['model_dir'], 'lr_finder_plot.png')
        plt.savefig(lr_plot_path)
        plt.close()
        
        # Find the learning rate with steepest negative gradient
        gradients = np.gradient(val_losses, log_lrs)
        steepest_point = np.argmin(gradients)
        best_lr = 10 ** log_lrs[steepest_point]
        
        # Log results
        logging.info(f"Learning rate finder completed. Suggested Learning Rate: {best_lr:.2e}")
        logging.info(f"Learning rate vs. loss plot saved as '{lr_plot_path}'")
        
        return best_lr

    except Exception as e:
        logging.error(f"Error during learning rate finding: {str(e)}")
        plt.close()  # Ensure plot is closed even if error occurs
        raise
    
    finally:
        # Cleanup
        progress_bar.close()
        if 'plt' in locals():
            plt.close()


def get_class_weights(y):
    class_counts = torch.bincount(y)
    class_weights = 1. / class_counts.float()
    # Apply sqrt to reduce extreme weights
    class_weights = torch.sqrt(class_weights)
    class_weights = class_weights / class_weights.sum()
    return class_weights


def get_training_parameters(data_manager):
    """Get training parameters based on configuration mode"""
    if CONFIG['training_mode']['hyperparameter_tuning']:
        logging.info("Starting hyperparameter tuning...")
        # Create a small validation set for tuning
        train_idx, val_idx = data_manager.create_cross_validator(n_splits=1).split(data_manager.data['y'])[0]
        
        # Get tuned parameters
        best_params = run_hyperparameter_tuning(
            X=data_manager.data['x'][train_idx],
            X_spectral=data_manager.data['x_spectral'][train_idx],
            y=data_manager.data['y'][train_idx],
            device=device,
            start_with_config=True  # Start from CONFIG values
        )
        
        model_params = {k: v for k, v in best_params.items() 
                       if k in ['n_filters', 'lstm_hidden', 'lstm_layers', 'dropout']}
        train_params = {
            'lr': best_params['lr'],
            'batch_size': best_params['batch_size'],
            'num_epochs': CONFIG['train_params']['initial']['num_epochs'],
            'patience': CONFIG['train_params']['initial']['patience']
        }
        
    else:
        logging.info("Using parameters from CONFIG...")
        model_params = CONFIG['model_params']['initial']
        train_params = CONFIG['train_params']['initial']
        
        # Optionally find best learning rate
        if CONFIG['training_mode']['find_lr']:
            logging.info("Finding optimal learning rate...")
            train_idx, val_idx = data_manager.create_cross_validator(n_splits=1).split(data_manager.data['y'])[0]
            
            # Initialize temporary model for LR finding
            temp_model, _ = initialize_model_with_gpu_check(model_params, device)
            temp_optimizer = optim.AdamW(temp_model.parameters(), 
                                       lr=train_params['lr'], 
                                       weight_decay=1e-5)
            
            # Create temporary loaders
            train_loader = data_manager.get_loader(train_idx, 
                                                 batch_size=train_params['batch_size'], 
                                                 is_training=True)
            val_loader = data_manager.get_loader(val_idx, 
                                               batch_size=train_params['batch_size'], 
                                               is_training=False)
            
            # UPDATED: Use BalancedFocalLoss instead of CrossEntropyLoss
            criterion = BalancedFocalLoss(
                gamma=2.0,
                alpha=data_manager.class_weights.to(device)
            )
            
            best_lr = find_lr(
                temp_model, train_loader, val_loader,
                temp_optimizer, criterion, device,
                start_lr=train_params['lr']
            )
            
            train_params['lr'] = best_lr
            logging.info(f"Found optimal learning rate: {best_lr:.2e}")
            
            # Clean up
            del temp_model
            torch.cuda.empty_cache()
    
    logging.info("Training parameters:")
    logging.info(f"Model parameters: {model_params}")
    logging.info(f"Training parameters: {train_params}")
    
    return model_params, train_params

def train_with_cv(data_manager, cv, model_params, train_params):
    """Train model using cross-validation"""
    results = []

    device = get_cuda_device()

    # Log initial class distribution once
    log_class_statistics(data_manager.data['y'], phase="Overall")
    
    # Get splits while maintaining class distribution
    logging.info("Creating cross-validation splits...")
    splits = cv.split(data_manager.data['y'])
    
    # Check if splits are being created
    splits = list(splits)  # Convert generator to list
    logging.info(f"Number of splits created: {len(splits)}")
    
    # Add final distribution logging
    final_distributions = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        train_dist = Counter(data_manager.data['y'][train_idx].numpy())
        val_dist = Counter(data_manager.data['y'][val_idx].numpy())
        final_distributions.append({
            'fold': fold,
            'train': train_dist,
            'val': val_dist
        })
    
    # Log final distributions
    logging.info("\nFinal Split Distributions:")
    for dist in final_distributions:
        logging.info(f"\nFold {dist['fold']}:")
        logging.info("Training Distribution:")
        logging.info(format_class_distribution(dist['train']))
        logging.info("Validation Distribution:")
        logging.info(format_class_distribution(dist['val']))

    if not splits:
        logging.error("No valid splits were created!")
        return results

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        logging.info(f"\nTraining Fold {fold}")
        logging.info(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")

        # Verify balance only for first fold
        verify_this_fold = (fold == 1)
        
        train_loader = data_manager.get_loader(
            train_idx,
            batch_size=train_params['batch_size'],
            is_training=True,
            verify_balance=verify_this_fold  # Only verify first fold
        )
        
        val_loader = data_manager.get_loader(
            val_idx,
            batch_size=train_params['batch_size'],
            is_training=False
        )
        
        # Initialize model
        # global device
        model, device = initialize_model_with_gpu_check(model_params, device)
        logging.info(f"Model initialized on {device}")
        
        # Calculate and update dataset statistics for this fold
        dataset_medians, dataset_iqrs = calculate_dataset_stats(train_loader)
        model.update_normalization_stats(dataset_medians, dataset_iqrs)

        # Setup training components
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_params['lr'],
            weight_decay=1e-5
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        criterion = BalancedFocalLoss(
        gamma=2.0, 
        alpha=data_manager.class_weights.to(device)
    )
        
        # Train fold
        best_model_state, metrics = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_params=train_params,
            fold=fold
        )
        
        results.append({
            'fold': fold,
            'best_model_state': best_model_state,
            'metrics': metrics,
            'val_indices': val_idx
        })
        
        # Log fold results
        log_fold_results(fold, metrics)
    
    return results

def train_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, train_params, fold):
    """Train a single fold"""
    early_stopping = EarlyStoppingWithMetrics(
        patience=train_params['patience'],
        min_epochs=train_params.get('min_epochs', 20),
        min_delta=train_params.get('min_delta', 0.001)
    )
    
    best_metrics = None
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(train_params['num_epochs']), 
                     desc=f'Fold {fold}',
                     position=0,
                     leave=True)
    
    for epoch in epoch_pbar:
        # Training phase
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion
        )
        
        # Validation phase
        val_metrics = validate_epoch(
            model=model,
            val_loader=val_loader,
            criterion=criterion
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Format per-class F1 scores
        per_class_f1 = {
            f"{SLEEP_STAGES[i]}": f"{val_metrics['per_class_f1'][i]:.3f}"
            for i in range(len(SLEEP_STAGES))
        }
        
        # Update progress bar with metrics
        postfix = {
            'tr_loss': f"{train_metrics['loss']:.3f}",
            'val_loss': f"{val_metrics['loss']:.3f}",
            'val_f1': f"{val_metrics['f1_macro']:.3f}",
            **per_class_f1,
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        }
        epoch_pbar.set_postfix(postfix)
        
        # Log progress (keeping your original logging)
        log_epoch_metrics(
            epoch=epoch,
            total_epochs=train_params['num_epochs'],
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            current_lr=optimizer.param_groups[0]['lr']
        )
        
        # Early stopping check
        if early_stopping(val_metrics, model.state_dict(), epoch):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Update best metrics (keeping your original logic)
        if best_metrics is None or val_metrics['f1_macro'] > best_metrics['f1_macro']:
            best_metrics = val_metrics
    
    # Log fold results (keeping your original logging)
    log_fold_results(fold, best_metrics)
    
    return early_stopping.best_state, best_metrics

# def train_epoch(model, train_loader, optimizer, criterion):
#     """Train for one epoch"""
#     model.train()
#     metrics = defaultdict(float)
#     all_predictions = []
#     all_targets = []
    
#     # Add progress bar for batches
#     batch_pbar = tqdm(enumerate(train_loader), 
#                      total=len(train_loader),
#                      desc='Training',
#                      position=1,
#                      leave=False)
    
#     for batch_idx, (batch_x, batch_x_spectral, batch_y) in batch_pbar:
#         batch_x = batch_x.to(device)
#         batch_x_spectral = batch_x_spectral.to(device)
#         batch_y = batch_y.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(batch_x, batch_x_spectral)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         metrics['loss'] += loss.item()
#         predictions = outputs.argmax(dim=1)
#         all_predictions.extend(predictions.cpu().numpy())
#         all_targets.extend(batch_y.cpu().numpy())
        
#         # Update batch progress bar
#         batch_pbar.set_postfix({
#             'loss': f"{loss.item():.3f}"
#         })
    
#     # Calculate epoch metrics (keeping your original logic)
#     metrics['loss'] /= len(train_loader)
#     metrics.update(calculate_metrics(all_predictions, all_targets))
    
#     return metrics

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    metrics = defaultdict(float)
    all_predictions = []
    all_targets = []
    
    # Updated GradScaler initialization
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = 4
    
    batch_pbar = tqdm(enumerate(train_loader), 
                     total=len(train_loader),
                     desc='Training',
                     position=1,
                     leave=False)
    
    for batch_idx, (batch_x, batch_x_spectral, batch_y) in batch_pbar:
        # Split batch into smaller mini-batches
        mini_batch_size = batch_x.size(0) // 2
        for i in range(0, batch_x.size(0), mini_batch_size):
            mini_x = batch_x[i:i+mini_batch_size].to(device)
            mini_spectral = batch_x_spectral[i:i+mini_batch_size].to(device)
            mini_y = batch_y[i:i+mini_batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Updated autocast
            with torch.amp.autocast('cuda'):
                outputs = model(mini_x, mini_spectral)
                loss = criterion(outputs, mini_y) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            metrics['loss'] += loss.item() * accumulation_steps
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(mini_y.cpu().numpy())
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.3f}"
            })
            
            # Clear cache
            del outputs, loss
            torch.cuda.empty_cache()
    
    # Calculate epoch metrics (keeping your original logic)
    metrics['loss'] /= len(train_loader)
    metrics.update(calculate_metrics(all_predictions, all_targets))
    
    return metrics

def get_dynamic_cv_splits(night_indices, min_folds=3, max_folds=10):
    """
    Determine optimal number of folds based on available nights
    
    Args:
        night_indices: Array of night indices for each sample
        min_folds: Minimum number of folds to use
        max_folds: Maximum number of folds to use
        
    Returns:
        int: Optimal number of folds
    """
    n_nights = len(np.unique(night_indices))
    n_folds = min(max(min_folds, n_nights // 2), max_folds)
    return n_folds

def objective(trial, X, X_spectral, y, device, n_folds=5, start_with_config=False):
    """
    Objective function for hyperparameter optimization using Optuna.
    Includes early stopping and improved error handling.
    
    Args:
        trial: Optuna trial object
        X: Input data
        X_spectral: Spectral features
        y: Target labels
        device: Device to run on
        n_folds: Number of cross-validation folds
        start_with_config: Whether to start with config parameters
    
    Returns:
        float: Mean validation accuracy across folds
    """
    try:
        # Get parameter ranges based on config
        model_ranges = CONFIG['model_params']['tuning_ranges']
        train_ranges = CONFIG['train_params']['tuning_ranges']
        initial_params = CONFIG['model_params']['initial']
        initial_train_params = CONFIG['train_params']['initial']
        
        # Update early stopping config to use actual config values
        early_stop_config = CONFIG['train_params']['tuning_ranges']['early_stopping']

        # Sample parameters based on configuration
        if start_with_config:
            model_params = {
                'n_filters': trial.suggest_categorical('n_filters', [initial_params['n_filters']]),
                'lstm_hidden': trial.suggest_int('lstm_hidden', 
                                               initial_params['lstm_hidden'], 
                                               initial_params['lstm_hidden']),
                'lstm_layers': trial.suggest_int('lstm_layers', 
                                               initial_params['lstm_layers'], 
                                               initial_params['lstm_layers']),
                'dropout': trial.suggest_float('dropout', 
                                            initial_params['dropout'], 
                                            initial_params['dropout'])
            }
            batch_size = trial.suggest_categorical('batch_size', [initial_train_params['batch_size']])
            lr_base = initial_train_params['lr']
        else:
            model_params = {
                'n_filters': trial.suggest_categorical('n_filters', model_ranges['n_filters']),
                'lstm_hidden': trial.suggest_categorical('lstm_hidden', model_ranges['lstm_hidden']),
                'lstm_layers': trial.suggest_int('lstm_layers', 
                                               min(model_ranges['lstm_layers']), 
                                               max(model_ranges['lstm_layers'])),
                'dropout': trial.suggest_float('dropout', 
                                            min(model_ranges['dropout']), 
                                            max(model_ranges['dropout']))
            }
            batch_size = trial.suggest_categorical('batch_size', train_ranges['batch_size'])
            lr_base = initial_train_params['lr']

        # Set learning rate range
        lr = trial.suggest_float('lr', lr_base * 0.1, lr_base * 10, log=True)

        # Get night indices from the data
        night_indices = torch.arange(len(X)) // (len(X) // len(DATA_FILES))
        
        # Determine optimal number of folds
        n_folds = get_dynamic_cv_splits(night_indices)
        unique_nights = np.unique(night_indices)
        
        # Create night-based splits
        night_splits = np.array_split(unique_nights, n_folds)
        cv_scores = []
        fold_metrics = []
        
        fold_progress = tqdm(enumerate(night_splits), 
                           total=n_folds, 
                           desc="Cross-validation folds",
                           leave=False)

        for fold, val_nights in fold_progress:
            try:
                # Get indices for validation nights
                val_mask = np.isin(night_indices, val_nights)
                train_mask = ~val_mask
                
                # Split data
                X_train_fold = X[train_mask]
                X_val_fold = X[val_mask]
                X_train_spectral_fold = X_spectral[train_mask]
                X_val_spectral_fold = X_spectral[val_mask]
                y_train_fold = y[train_mask]
                y_val_fold = y[val_mask]

                # Initialize model and move to device with error handling
                try:
                    model = EnsembleModel(model_params).to(device)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logging.warning(f"CUDA error in model initialization, falling back to CPU")
                        device = torch.device('cpu')
                        model = EnsembleModel(model_params).to(device)
                    else:
                        raise e

                # Create data loaders
                train_loader = create_data_loaders(
                    X_train_fold, X_train_spectral_fold, y_train_fold,
                    batch_size=batch_size, is_train=True
                )
                val_loader = create_data_loaders(
                    X_val_fold, X_val_spectral_fold, y_val_fold,
                    batch_size=batch_size, is_train=False
                )

                # Set up loss function with class weights
                class_weights = get_class_weights(y_train_fold).to(device)
                
                # Add gamma as a tunable parameter
                gamma = trial.suggest_float('focal_loss_gamma', 0.5, 3.0, log=True)

                # Then in the training loop:
                criterion = BalancedFocalLoss(
                    gamma=gamma,
                    alpha=class_weights
                )

                # Set up optimizer and scheduler
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=False
                )

                # Initialize early stopping
                early_stopping = EarlyStopping(
                    patience=early_stop_config['patience'],
                    min_epochs=early_stop_config['min_epochs'],
                    min_delta=early_stop_config['min_delta'],
                    monitor=early_stop_config['monitor']
                )

                fold_history = {
                    'train_loss': [],
                    'val_loss': [],
                    'val_accuracy': []
                }

                # Training loop
                epoch_progress = tqdm(range(train_ranges['num_epochs']), 
                                    desc=f"Fold {fold+1}/{n_folds}", 
                                    leave=False)

                for epoch in epoch_progress:
                    try:
                        # Training phase
                        model.train()
                        running_loss = 0.0
                        for batch_x, batch_x_spectral, batch_y in train_loader:
                            try:
                                batch_x = batch_x.to(device)
                                batch_x_spectral = batch_x_spectral.to(device)
                                batch_y = batch_y.to(device)

                                optimizer.zero_grad()
                                outputs = model(batch_x, batch_x_spectral)
                                loss = criterion(outputs, batch_y)
                                loss.backward()
                                
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()
                                
                                running_loss += loss.item()

                            except RuntimeError as e:
                                if "CUDA" in str(e):
                                    logging.warning("CUDA error in training batch, skipping...")
                                    continue
                                raise e

                        # Validation phase
                        val_loss, val_accuracy, _ = evaluate_model(
                            model, 
                            (X_val_fold, X_val_spectral_fold, y_val_fold), 
                            criterion, 
                            device
                        )
                        
                        # Update scheduler
                        scheduler.step(val_loss)

                        # Store metrics
                        fold_history['train_loss'].append(running_loss/len(train_loader))
                        fold_history['val_loss'].append(val_loss)
                        fold_history['val_accuracy'].append(val_accuracy)

                        # Update progress bar
                        epoch_progress.set_postfix({
                            'val_acc': f'{val_accuracy:.4f}',
                            'val_loss': f'{val_loss:.4f}',
                            'train_loss': f'{running_loss/len(train_loader):.4f}'
                        })

                        # Check early stopping
                        if early_stopping(
                            metrics={'loss': val_loss, 'accuracy': val_accuracy},
                            epoch=epoch,
                            state_dict=model.state_dict()
                        ):
                            logging.info(f"Early stopping triggered in fold {fold+1} at epoch {epoch+1}")
                            break

                    except Exception as e:
                        logging.error(f"Error in epoch {epoch}: {str(e)}")
                        continue

                # Store best score for this fold
                best_val_accuracy = max(fold_history['val_accuracy'])
                cv_scores.append(best_val_accuracy)
                fold_metrics.append(fold_history)
                
                fold_progress.set_postfix({
                    'mean_acc': f'{np.mean(cv_scores):.4f}',
                    'best_fold_acc': f'{best_val_accuracy:.4f}'
                })

            except Exception as e:
                logging.error(f"Error in fold {fold}: {str(e)}")
                cv_scores.append(0.0)
                continue

        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        # Log detailed results
        logging.info(f"""Trial completed:
        Parameters: {model_params}
        Batch size: {batch_size}
        Learning rate: {lr:.2e}
        Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}
        Best fold accuracy: {max(cv_scores):.4f}
        Worst fold accuracy: {min(cv_scores):.4f}
        """)

        return mean_accuracy

    except Exception as e:
        logging.error(f"Error in objective function: {str(e)}")
        return 0.0

def run_hyperparameter_tuning(X, X_spectral, y, device, n_trials=50, start_with_config=False):
    n_trials = CONFIG['train_params']['tuning_ranges'].get('n_trials', n_trials)

    try:
        study = create_study(direction='maximize', sampler=TPESampler())

        # Wrap the objective function to include error handling
        def objective_wrapper(trial):
            try:
                # Clear memory and cache before each trial
                torch.cuda.empty_cache()
                gc.collect()  # Free Python memory

                return objective(trial, X, X_spectral, y, device, start_with_config=start_with_config)
            except Exception as e:
                logging.error(f"Error in trial: {str(e)}")
                return 0.0

        study.optimize(objective_wrapper, n_trials=n_trials)
        best_params = study.best_params
        if not best_params:  # If no successful trials
            logging.warning("No successful trials. Using initial parameters.")
            best_params = CONFIG['model_params']['initial']
            best_params.update({
                'lr': CONFIG['train_params']['initial']['lr'],
                'batch_size': CONFIG['train_params']['initial']['batch_size']
            })
            
        logging.info(f"Best hyperparameters: {best_params}")
        return best_params
        
    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {str(e)}")
        # Return default parameters if tuning fails
        default_params = CONFIG['model_params']['initial']
        default_params.update({
            'lr': CONFIG['train_params']['initial']['lr'],
            'batch_size': CONFIG['train_params']['initial']['batch_size']
        })
        return default_params


def validate_epoch(model, val_loader, criterion):
    """Validate for one epoch"""
    model.eval()
    metrics = defaultdict(float)
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_x_spectral, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_x_spectral = batch_x_spectral.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x, batch_x_spectral)
            loss = criterion(outputs, batch_y)
            
            metrics['loss'] += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    metrics['loss'] /= len(val_loader)
    metrics.update(calculate_metrics(all_predictions, all_targets))
    
    return metrics



def save_best_model_results(results, data_manager, model_params, train_params, save_dir=CONFIG['model_dir']):
    """Save only the best model and its results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(save_dir, exist_ok=True)
    
    # Find best fold based on validation F1-macro score
    best_fold = max(results, key=lambda x: x['metrics']['f1_macro'])
    best_fold_num = best_fold['fold']
    
    logging.info(f"\nBest performance achieved in fold {best_fold_num}")
    
    # Save model weights
    model_path = os.path.join(save_dir, 'best_model.pt')
    torch.save(best_fold['best_model_state'], model_path)
    logging.info(f"Saved best model weights to: {model_path}")
    
    # Convert metrics to serializable format
    serializable_metrics = convert_to_serializable(best_fold['metrics'])
    
    # Save configuration and metrics
    config = {
        'model_params': convert_to_serializable(model_params),
        'train_params': convert_to_serializable(train_params),
        'best_metrics': serializable_metrics,
        'fold_number': int(best_fold_num),
        'data_info': {
            'n_nights': int(len(torch.unique(data_manager.data['night_idx']))),
            'n_samples': int(len(data_manager.data['y'])),
            'class_distribution': convert_to_serializable(
                Counter(data_manager.data['y'].numpy()).most_common()
            ),
            'validation_indices': convert_to_serializable(best_fold['val_indices'])
        },
        'timestamp': timestamp
    }

    # Save configuration
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"Saved model configuration to: {config_path}")
    
    # Evaluate best model
    model = EnsembleModel(model_params).to(device)
    model.load_state_dict(best_fold['best_model_state'])

    # Get validation data for best fold
    val_indices = best_fold['val_indices']
    X_val = data_manager.data['x'][val_indices]
    X_spectral_val = data_manager.data['x_spectral'][val_indices]
    y_val = data_manager.data['y'][val_indices]
    
    evaluator = SleepStageEvaluator(model_dir=save_dir)
    batch_size = train_params.get('batch_size', 32)
    evaluation_results = evaluator.evaluate_model(
        model=model,
        X=X_val, # Use the validation data
        X_spectral=X_spectral_val, # Use the validation data
        y=y_val, # Use the validation data
        model_name="Best Model",
        batch_size=batch_size
    )

    # Save evaluation results separately
    if isinstance(evaluation_results.get('metrics'), pd.DataFrame):
        metrics_csv_path = os.path.join(save_dir, 'evaluation_metrics.csv')
        evaluation_results['metrics'].to_csv(metrics_csv_path)
        evaluation_results['metrics'] = evaluation_results['metrics'].to_dict(orient='records')
    
    # Save other evaluation results
    eval_results = {
        'predictions': evaluation_results['predictions'].tolist(),
        'true_labels': evaluation_results['true_labels'].tolist(),
        'confusion_matrix_absolute': evaluation_results['confusion_matrix_absolute'].tolist(),
        'confusion_matrix_percentage': evaluation_results['confusion_matrix_percentage'].tolist()
    }
    
    eval_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=4)
    
    # Log final performance summary
    logging.info("\nBest Model Performance (Validation Set):")
    logging.info(f"Validation Metrics:")
    logging.info(f"    Loss: {best_fold['metrics']['loss']:.4f}")
    logging.info(f"    Accuracy: {best_fold['metrics']['accuracy']:.4f}")
    logging.info(f"    F1 Macro: {best_fold['metrics']['f1_macro']:.4f}")
    logging.info(f"    F1 Weighted: {best_fold['metrics']['f1_weighted']:.4f}")
    
    return save_dir, model

# def calculate_dataset_stats(data_loader):
#     """Calculate median and IQR statistics for each channel across the dataset using chunked processing"""
#     logging.info("Calculating dataset statistics...")
    
#     # Initialize running statistics
#     n_channels = 4
#     running_values = {i: [] for i in range(n_channels)}
#     chunk_size = 10000  # Adjust this based on your memory constraints
    
#     with torch.no_grad():
#         for batch_x, _, _ in tqdm(data_loader, desc="Processing batches"):
#             # If last dimension is 1, squeeze it
#             if batch_x.shape[-1] == 1:
#                 batch_x = batch_x.squeeze(-1)
            
#             # Process each channel separately
#             for i in range(n_channels):
#                 channel_data = batch_x[:, i].cpu()
#                 # Store sorted samples for later quantile calculation
#                 running_values[i].append(channel_data)
                
#                 # If we've accumulated enough samples, sort and subsample
#                 if len(running_values[i]) * batch_x.shape[0] > chunk_size:
#                     combined = torch.cat(running_values[i])
#                     # Sort and subsample to keep memory usage bounded
#                     sorted_values, _ = torch.sort(combined)
#                     stride = max(len(sorted_values) // chunk_size, 1)
#                     running_values[i] = [sorted_values[::stride]]
    
#     # Calculate final statistics
#     medians = torch.zeros(n_channels)
#     iqrs = torch.zeros(n_channels)
    
#     for i in range(n_channels):
#         # Combine all values for this channel
#         combined = torch.cat(running_values[i])
#         sorted_values, _ = torch.sort(combined)
        
#         # Calculate statistics
#         medians[i] = sorted_values[len(sorted_values)//2]
#         q1 = sorted_values[len(sorted_values)//4]
#         q3 = sorted_values[3*len(sorted_values)//4]
#         iqrs[i] = q3 - q1
        
#         # Clear memory
#         del sorted_values
#         torch.cuda.empty_cache()
    
#     logging.info(f"Calculated medians: {medians}")
#     logging.info(f"Calculated IQRs: {iqrs}")
    
#     return medians, iqrs

def calculate_dataset_stats(data_loader):
    """Calculate median and IQR statistics for each channel across the dataset using chunked processing"""
    logging.info("Calculating dataset statistics...")
    
    # Initialize running statistics
    n_channels = 4
    running_values = {i: [] for i in range(n_channels)}
    chunk_size = 10000  # Adjust this based on your memory constraints
    
    with torch.no_grad():
        for batch_x, _, _ in tqdm(data_loader, desc="Processing batches"):
            # If last dimension is 1, squeeze it
            if batch_x.shape[-1] == 1:
                batch_x = batch_x.squeeze(-1)
            
            # Process each channel separately
            for i in range(n_channels):
                # Flatten the channel data
                channel_data = batch_x[:, i].reshape(-1).cpu()
                # Store samples for later quantile calculation
                running_values[i].append(channel_data)
                
                # If we've accumulated enough samples, sort and subsample
                if len(running_values[i]) * batch_x.shape[0] > chunk_size:
                    combined = torch.cat(running_values[i])
                    # Sort and subsample to keep memory usage bounded
                    sorted_values, _ = torch.sort(combined)
                    stride = max(len(sorted_values) // chunk_size, 1)
                    running_values[i] = [sorted_values[::stride]]
    
    # Calculate final statistics
    medians = torch.zeros(n_channels)
    iqrs = torch.zeros(n_channels)
    
    for i in range(n_channels):
        # Combine all values for this channel
        combined = torch.cat(running_values[i])
        sorted_values, _ = torch.sort(combined.flatten())  # Ensure values are flattened
        
        # Calculate statistics
        total_len = len(sorted_values)
        medians[i] = sorted_values[total_len//2].item()  # Convert to scalar
        q1 = sorted_values[total_len//4].item()
        q3 = sorted_values[3*total_len//4].item()
        iqrs[i] = q3 - q1
        
        # Clear memory
        del sorted_values
        torch.cuda.empty_cache()
    
    logging.info(f"Calculated medians: {medians}")
    logging.info(f"Calculated IQRs: {iqrs}")
    
    return medians, iqrs

'''
---------------------------------------------------------------------------------------------------------
                                                CODE FOR LATER
---------------------------------------------------------------------------------------------------------

'''



def distill_knowledge(teacher_model, student_model, train_loader, val_data, device, num_epochs=50, log_interval=5):
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = get_scheduler(optimizer, num_warmup_steps=len(train_loader) * 5, num_training_steps=len(train_loader) * num_epochs)
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    temperature = 2.0
    alpha = 0.5  # Balance between hard and soft targets

    teacher_model.eval()
    overall_progress = tqdm(total=num_epochs, desc="Overall Distillation Progress", position=0)
    
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        
        epoch_progress = tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        for batch_x, batch_x_spectral, batch_y in epoch_progress:
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(batch_x, batch_x_spectral)
            
            student_outputs = student_model(batch_x, batch_x_spectral)
            
            # Soft targets
            soft_loss = kl_criterion(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature ** 2)
            
            # Hard targets
            hard_loss = ce_criterion(student_outputs, batch_y)
            
            # Combined loss
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            epoch_progress.set_postfix({'loss': f'{running_loss/(epoch_progress.n+1):.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        # Evaluate and log every log_interval epochs
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            _, accuracy, _ = evaluate_model(student_model, val_data, ce_criterion, device)
            logging.info(f"Distillation Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        overall_progress.update(1)
    
    overall_progress.close()
    return student_model


