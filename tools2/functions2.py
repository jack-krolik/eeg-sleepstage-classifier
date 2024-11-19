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
from tools2.classes2 import *
from tools2.config2 import *
from tools2.utils2 import *
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

def load_data(data_paths):
    """
    Load and combine any number of nights of sleep data
    
    Args:
        data_paths: List of paths to .mat files containing sleep data
        
    Returns:
        tuple: Combined features, labels, and night indices
    """
    combined_x = []
    combined_y = []
    night_indices = []  # Track which night each sample comes from
    
    for night_idx, data_path in enumerate(data_paths):
        try:
            mat_file = sio.loadmat(data_path)
            
            # Stack the signals for current night
            x = np.stack((mat_file['sig1'], mat_file['sig2'], 
                         mat_file['sig3'], mat_file['sig4']), axis=1)
            x = torch.from_numpy(x).float()
            
            # Get labels for current night
            y = torch.from_numpy(mat_file['labels'].flatten()).long()
            
            # Filter valid indices
            valid_indices = y != -1
            x = x[valid_indices]
            y = y[valid_indices]
            
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            # Add data from this night
            combined_x.append(x)
            combined_y.append(y)
            night_indices.extend([night_idx] * len(y))
            
            logging.info(f"Loaded data from {data_path}")
            logging.info(f"Night {night_idx + 1} data shape: {x.shape}, Labels shape: {y.shape}")
            class_distribution = format_class_distribution(Counter(combined_y.numpy())).replace('\n', '\n    ')
            logging.info(f"Class distribution for night {night_idx + 1}:\n    {class_distribution}")            
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {e}")
            continue
    
    try:
        combined_x = torch.cat(combined_x, dim=0)
        combined_y = torch.cat(combined_y, dim=0)
        night_indices = torch.tensor(night_indices)
        
        logging.info(f"Combined data shape: {combined_x.shape}")
        logging.info(f"Combined labels shape: {combined_y.shape}")
        logging.info(f"Overall class distribution: {Counter(combined_y.numpy())}")
        
        return combined_x, combined_y, night_indices
        
    except Exception as e:
        logging.error(f"Error combining data: {e}")
        raise

def prepare_data_multi_night(x, y, night_indices, test_size=0.2, val_size=0.1):
    """
    Prepare data for any number of nights with robust handling of rare classes
    
    Args:
        x: Combined input features from multiple nights
        y: Combined labels from multiple nights
        night_indices: Tensor indicating which night each sample belongs to
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
    """
    # Convert to numpy for sklearn
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    night_indices_np = night_indices.cpu().numpy()
    
    # Analyze class distribution
    class_counts = Counter(y_np)
    logging.info(f"Class distribution before split: {format_class_distribution(class_counts)}")
    
    try:
        # First split: separate test set
        # If some classes have too few samples, we'll fall back to night-based stratification only
        if min(class_counts.values()) >= 5:
            # Create composite stratification labels
            stratify_labels = np.stack([night_indices_np, y_np], axis=1)
            X_train_val, X_test, y_train_val, y_test, night_train_val, night_test = train_test_split(
                x_np, y_np, night_indices_np,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=42
            )
        else:
            logging.warning("Some classes have very few samples. Falling back to night-based stratification only.")
            X_train_val, X_test, y_train_val, y_test, night_train_val, night_test = train_test_split(
                x_np, y_np, night_indices_np,
                test_size=test_size,
                stratify=night_indices_np,
                random_state=42
            )

        # log split statistics
        logging.info("\nData split statistics:")
        logging.info(f"Training set shape: {X_train_val.shape}")
        logging.info(f"Training set nights: {Counter(night_train_val)}")
        logging.info(f"Training set classes: {Counter(y_train_val)}")
        logging.info(f"Training set class distribution: {format_class_distribution(Counter(y_train_val))}")
        logging.info(f"Test set shape: {X_test.shape}")
        logging.info(f"Test set nights: {Counter(night_test)}")
        logging.info(f"Test set classes: {Counter(y_test)}")
        logging.info(f"Test set class distribution: {format_class_distribution(Counter(y_test))}")
        
        # Second split: separate validation set
        # Again, check if we can stratify by both night and class
        if min(Counter(y_train_val).values()) >= 5:
            stratify_labels_train = np.stack([night_train_val, y_train_val], axis=1)
            X_train, X_val, y_train, y_val, night_train, night_val = train_test_split(
                X_train_val, y_train_val, night_train_val,
                test_size=val_size/(1-test_size),
                stratify=stratify_labels_train,
                random_state=42
            )
        else:
            logging.warning("Falling back to night-based stratification for validation split.")
            X_train, X_val, y_train, y_val, night_train, night_val = train_test_split(
                X_train_val, y_train_val, night_train_val,
                test_size=val_size/(1-test_size),
                stratify=night_train_val,
                random_state=42
            )
        
        # Convert to PyTorch tensors
        X_train_torch = torch.from_numpy(X_train).float()
        X_val_torch = torch.from_numpy(X_val).float()
        X_test_torch = torch.from_numpy(X_test).float()
        
        # Extract spectral features
        X_train_spectral = np.array([extract_spectral_features(x) for x in X_train_torch])
        X_val_spectral = np.array([extract_spectral_features(x) for x in X_val_torch])
        X_test_spectral = np.array([extract_spectral_features(x) for x in X_test_torch])
        
        # Log split statistics
        logging.info("\nData split statistics:")
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Training set nights: {Counter(night_train)}")
        logging.info(f"Training set classes: {Counter(y_train)}")
        logging.info(f"Validation set shape: {X_val.shape}")
        logging.info(f"Validation set nights: {Counter(night_val)}")
        logging.info(f"Validation set classes: {Counter(y_val)}")
        logging.info(f"Test set shape: {X_test.shape}")
        logging.info(f"Test set nights: {Counter(night_test)}")
        logging.info(f"Test set classes: {Counter(y_test)}")
        
        # Apply improved oversampling only to training set
        X_train_resampled, X_train_spectral_resampled, y_resampled = improved_oversample(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(X_train_spectral).float(),
            y_train,
            max_ratio=3.0,
            min_samples=50
        )
        
        logging.info("\nAfter oversampling:")
        logging.info(f"Training set classes: {Counter(y_resampled.numpy())}")
        
        return (X_train_resampled,
                X_train_spectral_resampled,
                y_resampled,
                torch.from_numpy(X_val).float(),
                torch.from_numpy(X_val_spectral).float(),
                torch.from_numpy(y_val).long(),
                torch.from_numpy(X_test).float(),
                torch.from_numpy(X_test_spectral).float(),
                torch.from_numpy(y_test).long())
                
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise




def preprocess_data(X, X_spectral):
    # Remove outliers (e.g., clip values beyond 3 standard deviations)
    X = torch.clamp(X, -3, 3)
    
    # Ensure X_spectral is non-negative (if it represents power spectral density)
    X_spectral = torch.clamp(X_spectral, min=0)
    
    return X, X_spectral


def generate_random_params():
    return {
        'model_params': {
            'n_filters': [random.choice(CONFIG['model_params']['tuning_ranges']['n_filters'])],
            'lstm_hidden': random.choice(CONFIG['model_params']['tuning_ranges']['lstm_hidden']),
            'lstm_layers': random.choice(CONFIG['model_params']['tuning_ranges']['lstm_layers']),
            'dropout': random.uniform(*CONFIG['model_params']['tuning_ranges']['dropout'])
        },
        'train_params': {
            'lr': random.uniform(*CONFIG['train_params']['tuning_ranges']['lr']),
            'batch_size': random.choice(CONFIG['train_params']['tuning_ranges']['batch_size']),
            'num_epochs': random.choice(CONFIG['train_params']['tuning_ranges']['num_epochs']),
            'patience': random.choice(CONFIG['train_params']['tuning_ranges']['patience'])
        }
    }


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


def extract_spectral_features(x):
    """Enhanced spectral feature extraction"""
    features = []
    for channel in range(x.shape[0]):
        channel_data = x[channel].cpu().numpy()
        f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
        
        # Traditional frequency bands
        delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
        theta = np.sum(psd[(f > 4) & (f <= 8)])
        alpha = np.sum(psd[(f > 8) & (f <= 13)])
        beta = np.sum(psd[(f > 13) & (f <= 30)])
        gamma = np.sum(psd[(f > 30) & (f <= 50)])  # Added gamma band
        
        # Band ratios (useful for sleep stage classification)
        theta_alpha_ratio = theta / (alpha + 1e-6)
        delta_beta_ratio = delta / (beta + 1e-6)
        
        # Spectral edge frequency (frequency below which 95% of power exists)
        cumsum = np.cumsum(psd)
        spectral_edge = f[np.where(cumsum >= 0.95 * cumsum[-1])[0][0]]
        
        features.extend([
            delta, theta, alpha, beta, gamma,
            theta_alpha_ratio, delta_beta_ratio,
            spectral_edge
        ])
    return np.array(features)


# @torch.jit.script  # Use TorchScript for speed
# def time_warp_batch(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
#     """GPU-optimized batch time warping"""
#     batch_size, n_channels, n_samples = x.shape
#     device = x.device
    
#     # Create time steps
#     orig_steps = torch.arange(n_samples, device=device).float()
    
#     # Create warping points
#     warp_steps = torch.linspace(0, n_samples-1, knot+2, device=device)
    
#     # Generate random warps for the batch
#     random_warps = torch.normal(
#         mean=1.0,
#         std=sigma,
#         size=(batch_size, knot+2),
#         device=device
#     )
    
#     # Ensure positive values
#     random_warps = torch.abs(random_warps)
    
#     # Fix endpoints
#     random_warps[:, 0] = 1.0
#     random_warps[:, -1] = 1.0
    
#     # Create output tensor
#     out = torch.zeros_like(x)
    
#     # Process each sample in the batch
#     for i in range(batch_size):
#         # Create cumulative warps
#         warped = torch.cumsum(random_warps[i], dim=0)
#         warped = warped * (n_samples - 1) / warped[-1]
        
#         # Interpolate for each channel
#         for c in range(n_channels):
#             out[i, c] = torch.interp(
#                 orig_steps,
#                 warped,
#                 x[i, c],
#                 left=x[i, c, 0],
#                 right=x[i, c, -1]
#             )
    
#     return out

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


def augment_minority_classes(x, x_spectral, y, minority_classes):
    augmented_x = []
    augmented_x_spectral = []
    augmented_y = []
    for i in range(len(y)):
        augmented_x.append(x[i])
        augmented_x_spectral.append(x_spectral[i])
        augmented_y.append(y[i])
        if y[i] in minority_classes:
            # Apply time_warp augmentation
            augmented = torch.from_numpy(time_warp(x[i].unsqueeze(0).numpy(), sigma=0.3, knot=5)).squeeze(0)
            augmented_x.append(augmented)
            augmented_x_spectral.append(x_spectral[i])  # Duplicate spectral features for augmented data
            augmented_y.append(y[i])
           
    return torch.stack(augmented_x), torch.stack(augmented_x_spectral), torch.tensor(augmented_y)

def _augment_minorities(self, x, x_spectral, y, threshold_ratio=0.5):
    """GPU-optimized minority class augmentation"""
    device = x.device
    class_counts = torch.bincount(y)
    max_count = class_counts.max()
    
    # Pre-allocate lists for augmented data
    augmented_x = [x]
    augmented_x_spectral = [x_spectral]
    augmented_y = [y]
    
    # Process each minority class
    with tqdm(total=len(class_counts), desc="Augmenting minority classes") as pbar:
        for class_idx, count in enumerate(class_counts):
            if count / max_count < threshold_ratio:
                class_mask = y == class_idx
                class_x = x[class_mask]
                class_x_spectral = x_spectral[class_mask]
                
                n_aug = int(max_count * threshold_ratio) - count
                if n_aug <= 0:
                    continue
                
                # Batch the augmentation
                batch_size = min(n_aug, 128)  # Process in batches of 128
                remaining = n_aug
                
                while remaining > 0:
                    curr_batch = min(batch_size, remaining)
                    
                    # Apply time warping on GPU
                    aug_x = time_warp(
                        class_x[:curr_batch].to(device),
                        sigma=0.2,
                        knot=4
                    )
                    
                    augmented_x.append(aug_x)
                    augmented_x_spectral.append(class_x_spectral[:curr_batch].repeat(1, 1))
                    augmented_y.append(torch.full((curr_batch,), class_idx, device=device))
                    
                    remaining -= curr_batch
            pbar.update(1)
    
    # Concatenate all augmented data
    return (
        torch.cat(augmented_x, dim=0),
        torch.cat(augmented_x_spectral, dim=0),
        torch.cat(augmented_y, dim=0)
    )

def simple_oversample(X, X_spectral, y):
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    oversampled_X = []
    oversampled_X_spectral = []
    oversampled_y = []
    
    for class_label in class_counts:
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)
        n_oversample = max_count - n_samples
        
        oversampled_X.append(X[class_indices])
        oversampled_X_spectral.append(X_spectral[class_indices])
        oversampled_y.extend([class_label] * n_samples)
        
        if n_oversample > 0:
            oversampled_indices = np.random.choice(class_indices, size=n_oversample, replace=True)
            oversampled_X.append(X[oversampled_indices])
            oversampled_X_spectral.append(X_spectral[oversampled_indices])
            oversampled_y.extend([class_label] * n_oversample)
    
    return np.concatenate(oversampled_X), np.concatenate(oversampled_X_spectral), np.array(oversampled_y)

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
                criterion = nn.CrossEntropyLoss(weight=class_weights + 1e-6, label_smoothing=0.1)

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


def plot_training_history(metrics_history, save_dir):
    """Enhanced training history visualization"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Losses
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(metrics_history['train_loss'], label='Train Loss')
    ax1.plot(metrics_history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot 2: Accuracies
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(metrics_history['train_accuracy'], label='Train Accuracy')
    ax2.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Plot 3: F1 Scores
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(metrics_history['train_f1_macro'], label='Train F1-Macro')
    ax3.plot(metrics_history['val_f1_macro'], label='Validation F1-Macro')
    ax3.set_title('F1 Score History')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.legend()
    
    # Plot 4: Learning Rate
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(metrics_history['learning_rates'])
    ax4.set_title('Learning Rate History')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Plot per-class metrics
    plot_per_class_metrics(metrics_history['per_class_metrics'], save_dir)

def plot_per_class_metrics(per_class_metrics, save_dir):
    """Plot per-class performance metrics"""
    n_epochs = len(per_class_metrics)
    n_classes = len(per_class_metrics[0]['train']['f1'])
    
    class_names = [SLEEP_STAGES[i] for i in range(n_classes)]
    
    plt.figure(figsize=(15, 8))
    for i in range(n_classes):
        train_f1 = [epoch['train']['f1'][i] for epoch in per_class_metrics]
        val_f1 = [epoch['val']['f1'][i] for epoch in per_class_metrics]
        
        plt.plot(train_f1, linestyle='--', label=f'{class_names[i]} (Train)')
        plt.plot(val_f1, label=f'{class_names[i]} (Val)')
    
    plt.title('Per-Class F1 Score History')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'))
    plt.close()

def calculate_epoch_metrics(predictions, true_labels, running_loss, num_batches):
    """Calculate comprehensive metrics for an epoch"""
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    return {
        'loss': running_loss / num_batches,
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'per_class_f1': f1_score(true_labels, predictions, average=None),
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }

def validate_model(model, val_data, criterion, device):
    """Enhanced validation function"""
    model.eval()
    X, X_spectral, y = val_data
    predictions = []
    true_labels = y.cpu().numpy()
    total_loss = 0
    
    with torch.no_grad():
        batch_size = 128  # Adjust based on available memory
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size].to(device)
            batch_x_spectral = X_spectral[i:i+batch_size].to(device)
            batch_y = y[i:i+batch_size].to(device)
            
            outputs = model(batch_x, batch_x_spectral)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * len(batch_y)
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
    
    predictions = np.array(predictions)
    return calculate_epoch_metrics(predictions, true_labels, total_loss, len(X))

def update_metrics_history(history, train_metrics, val_metrics, lr):
    """Update training history with new metrics"""
    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    history['val_accuracy'].append(val_metrics['accuracy'])
    history['train_f1_macro'].append(train_metrics['f1_macro'])
    history['val_f1_macro'].append(val_metrics['f1_macro'])
    history['per_class_metrics'].append({
        'train': {'f1': train_metrics['per_class_f1']},
        'val': {'f1': val_metrics['per_class_f1']}
    })
    history['learning_rates'].append(lr)


def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, 
                device, epochs=100, accumulation_steps=4, verbose=True):
    """Enhanced training loop with improved monitoring and stability"""
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=CONFIG['train_params']['initial']['early_stopping']['patience'],
        min_epochs=CONFIG['train_params']['initial']['early_stopping']['min_epochs'],
        min_delta=CONFIG['train_params']['initial']['early_stopping']['min_delta'],
        monitor=['loss', 'accuracy', 'f1_macro'],  # Added F1 macro monitoring
        mode='auto'
    )
    
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1_macro': [], 'val_f1_macro': [],
        'per_class_metrics': [],
        'learning_rates': []
    }
    
    # Track best metrics
    best_metrics = {'val_f1_macro': 0.0, 'epoch': 0}
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
        model.train()
        running_stats = {
            'loss': 0.0,
            'predictions': [],
            'true_labels': [],
            'batch_sizes': []
        }
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, (batch_x, batch_x_spectral, batch_y) in enumerate(train_loader):
            try:
                batch_x = batch_x.to(device)
                batch_x_spectral = batch_x_spectral.to(device)
                batch_y = batch_y.to(device)
                
                # Gradient accumulation setup
                is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
                with autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
                    outputs = model(batch_x, batch_x_spectral)
                    loss = criterion(outputs, batch_y) / accumulation_steps
                
                # Handle gradient accumulation
                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    if not is_accumulation_step:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if not is_accumulation_step:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Update running statistics
                running_stats['loss'] += loss.item() * accumulation_steps
                running_stats['predictions'].extend(outputs.argmax(dim=1).cpu().numpy())
                running_stats['true_labels'].extend(batch_y.cpu().numpy())
                running_stats['batch_sizes'].append(len(batch_y))
                
                # Memory management for CUDA
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logging.error(f"CUDA error in batch {batch_idx}: {str(e)}")
                    # Attempt recovery
                    torch.cuda.empty_cache()
                    continue
                raise e
        
        # Calculate training metrics
        train_metrics = calculate_epoch_metrics(
            running_stats['predictions'],
            running_stats['true_labels'],
            running_stats['loss'],
            len(train_loader)
        )
        
        # Validation phase with early stopping check
        val_metrics = validate_model(
            model, val_data, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        update_metrics_history(
            metrics_history,
            train_metrics,
            val_metrics,
            current_lr
        )
        
        # Early stopping check
        stop_training = early_stopping(
            metrics={
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'f1_macro': val_metrics['f1_macro']
            },
            epoch=epoch,
            state_dict=model.state_dict()
        )
        
        # Update best metrics if improved
        if val_metrics['f1_macro'] > best_metrics['val_f1_macro']:
            best_metrics.update({
                'val_f1_macro': val_metrics['f1_macro'],
                'epoch': epoch,
                'state_dict': model.state_dict().copy()
            })
        
        if verbose:
            log_epoch_metrics(epoch, epochs, train_metrics, val_metrics, current_lr)
        
        if stop_training:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Plot and save training history
    plot_training_history(metrics_history, save_dir=CONFIG['model_dir'])
    
    return best_metrics['state_dict'], best_metrics['val_f1_macro']


def evaluate_model(model, data, criterion, device):
    model.eval()
    X, X_spectral, y = data
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    
    with torch.no_grad():
        outputs = model(X.to(device), X_spectral.to(device))
        loss = criterion(outputs, y.to(device))
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        total += y.size(0)
        correct += (predicted == y.to(device)).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    return avg_loss, accuracy, np.array(all_predictions)

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


def convert_to_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj