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
from tools.classes import *
from tools.config import *
from tools.utils import *
import random
import math
import os
from optuna import create_study, trial
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from torch.utils.data import WeightedRandomSampler



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

# def load_data(data_path):
#     try:
#         mat_file = sio.loadmat(data_path)
        
#         x = np.stack((mat_file['sig1'], mat_file['sig2'], mat_file['sig3'], mat_file['sig4']), axis=1)
#         x = torch.from_numpy(x).float()
        
#         y = torch.from_numpy(mat_file['labels'].flatten()).long()
        
#         valid_indices = y != -1
#         x = x[valid_indices]
#         y = y[valid_indices]
        
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
        
#         print(f"Loaded data shape: {x.shape}, Labels shape: {y.shape}")
        
#         return x, y

#     except Exception as e:
#         logging.error(f"Error loading data from {data_path}: {e}")
#         raise

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
            logging.info(f"Class distribution for night {night_idx + 1}: {Counter(y.numpy())}")
            
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {e}")
            raise
    
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
    logging.info(f"Class distribution before split: {class_counts}")
    
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


# def create_data_loaders(X, X_spectral, y, batch_size, is_train=True):
#     dataset = TensorDataset(X, X_spectral, y)
#     if is_train:
#         sampler = BalancedBatchSampler(y.numpy(), batch_size=batch_size)
#         return DataLoader(dataset, batch_sampler=sampler)
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
    features = []
    for channel in range(x.shape[0]):  # Iterate over channels
        # Convert to NumPy array for scipy.signal.welch
        channel_data = x[channel].cpu().numpy()
        f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
        delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
        theta = np.sum(psd[(f > 4) & (f <= 8)])
        alpha = np.sum(psd[(f > 8) & (f <= 13)])
        beta = np.sum(psd[(f > 13) & (f <= 30)])
        features.extend([delta, theta, alpha, beta])
    return np.array(features)





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
    Improved oversampling with controlled ratios and augmentation
    
    Args:
        X: Input features
        X_spectral: Spectral features
        y: Labels
        max_ratio: Maximum ratio between largest and smallest class
        min_samples: Minimum samples per class after oversampling
    """
    class_counts = Counter(y)
    median_count = np.median(list(class_counts.values()))
    target_counts = {}
    
    # Calculate target count for each class
    for class_label, count in class_counts.items():
        if count < median_count:
            # Don't oversample minority classes too aggressively
            target_count = min(
                median_count,  # Don't exceed median
                max(
                    min_samples,  # Ensure minimum samples
                    count * max_ratio  # Limit oversampling ratio
                )
            )
        else:
            target_count = count  # Don't modify majority classes
        target_counts[class_label] = int(target_count)
    
    oversampled_X = []
    oversampled_X_spectral = []
    oversampled_y = []
    
    for class_label in class_counts:
        class_indices = np.where(y == class_label)[0]
        current_count = len(class_indices)
        target_count = target_counts[class_label]
        
        # First add all original samples
        oversampled_X.append(X[class_indices])
        oversampled_X_spectral.append(X_spectral[class_indices])
        oversampled_y.extend([class_label] * current_count)
        
        if current_count < target_count:
            n_oversample = target_count - current_count
            
            # For very small classes, use augmentation
            if current_count < 10:
                aug_samples = []
                aug_spectral = []
                for _ in range(n_oversample):
                    idx = np.random.choice(class_indices)
                    # Apply time warping with random parameters
                    augmented = time_warp(
                        X[idx:idx+1], 
                        sigma=np.random.uniform(0.1, 0.4),
                        knot=np.random.randint(4, 7)
                    )
                    aug_samples.append(torch.from_numpy(augmented).squeeze(0))
                    aug_spectral.append(X_spectral[idx])
                
                oversampled_X.append(torch.stack(aug_samples))
                oversampled_X_spectral.append(torch.stack(aug_spectral))
                oversampled_y.extend([class_label] * n_oversample)
            else:
                # For larger minority classes, use random oversampling
                oversample_idx = np.random.choice(class_indices, size=n_oversample, replace=True)
                oversampled_X.append(X[oversample_idx])
                oversampled_X_spectral.append(X_spectral[oversample_idx])
                oversampled_y.extend([class_label] * n_oversample)
    
    return (torch.cat(oversampled_X), 
            torch.cat(oversampled_X_spectral), 
            torch.tensor(oversampled_y))

    

# def prepare_data(x, y, test_size=0.2, val_size=0.1):
#     # Convert PyTorch tensors to NumPy arrays for scikit-learn
#     x_np = x.cpu().numpy()
#     y_np = y.cpu().numpy()

#     # First split: separate test set
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         x_np, y_np, 
#         test_size=test_size, 
#         stratify=y_np, 
#         random_state=42
#     )
    
#     # Second split: separate validation set
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, 
#         y_train_val, 
#         test_size=val_size/(1-test_size), 
#         stratify=y_train_val, 
#         random_state=42
#     )
    
#     # Convert back to PyTorch tensors for feature extraction
#     X_train_torch = torch.from_numpy(X_train).float()
#     X_val_torch = torch.from_numpy(X_val).float()
#     X_test_torch = torch.from_numpy(X_test).float()

#     # Extract spectral features
#     X_train_spectral = np.array([extract_spectral_features(x) for x in X_train_torch])
#     X_val_spectral = np.array([extract_spectral_features(x) for x in X_val_torch])
#     X_test_spectral = np.array([extract_spectral_features(x) for x in X_test_torch])
    
#     print("Original train set class distribution:")
#     print(Counter(y_train))
    
#     # Apply improved oversampling
#     X_train_resampled, X_train_spectral_resampled, y_resampled = improved_oversample(
#         torch.from_numpy(X_train).float(),
#         torch.from_numpy(X_train_spectral).float(),
#         y_train,
#         max_ratio=3.0,  # Adjust this based on your needs
#         min_samples=50  # Adjust this based on your needs
#     )
    
#     print("After improved oversampling train set class distribution:")
#     print(Counter(y_resampled.numpy()))
    
#     # Convert everything to PyTorch tensors
#     return (X_train_resampled, 
#             X_train_spectral_resampled, 
#             torch.tensor(y_resampled), 
#             # y_resampled.clone().detach(),
#             torch.from_numpy(X_val).float(),
#             torch.from_numpy(X_val_spectral).float(), 
#             torch.from_numpy(y_val).long(),
#             torch.from_numpy(X_test).float(),
#             torch.from_numpy(X_test_spectral).float(), 
#             torch.from_numpy(y_test).long())



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


# def get_class_weights(y):
#     class_counts = torch.bincount(y)
#     class_weights = 1. / class_counts.float()
#     class_weights = class_weights / class_weights.sum()
#     return class_weights
def get_class_weights(y):
    class_counts = torch.bincount(y)
    class_weights = 1. / class_counts.float()
    # Apply sqrt to reduce extreme weights
    class_weights = torch.sqrt(class_weights)
    class_weights = class_weights / class_weights.sum()
    return class_weights









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

        # Cross-validation setup
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_metrics = []
        
        fold_progress = tqdm(enumerate(kf.split(X)), 
                           total=n_folds, 
                           desc="Cross-validation folds",
                           leave=False)

        for fold, (train_index, val_index) in fold_progress:
            try:
                # Split data
                X_train_fold = X[train_index]
                X_val_fold = X[val_index]
                X_train_spectral_fold = X_spectral[train_index]
                X_val_spectral_fold = X_spectral[val_index]
                y_train_fold = y[train_index]
                y_val_fold = y[val_index]

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


def plot_training_history(metrics_history):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['model_dir'], 'training_history.png'))
    plt.close()

# def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, 
#                 device, epochs=100, accumulation_steps=4, verbose=True):
#     scaler = GradScaler()
#     early_stopping = EarlyStopping(
#         patience=CONFIG['train_params']['initial']['early_stopping']['patience'],
#         min_epochs=CONFIG['train_params']['initial']['early_stopping']['min_epochs'],
#         min_delta=CONFIG['train_params']['initial']['early_stopping']['min_delta'],
#         monitor=CONFIG['train_params']['initial']['early_stopping']['monitor'],
#         mode='auto'
#     )
    
#     metrics_history = {
#         'train_loss': [],
#         'val_loss': [],
#         'val_accuracy': []
#     }
    
#     for epoch in tqdm(range(epochs), desc="Training Progress"):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (batch_x, batch_x_spectral, batch_y) in enumerate(train_loader):
#             try:
#                 batch_x = batch_x.to(device)
#                 batch_x_spectral = batch_x_spectral.to(device)
#                 batch_y = batch_y.to(device)
                
#                 # Perform mixed precision training with autocast
#                 with autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
#                     outputs = model(batch_x, batch_x_spectral)
#                     loss = criterion(outputs, batch_y)
#                     loss = loss / accumulation_steps
                
#                 if device.type == 'cpu':
#                     loss.backward()
#                     if (batch_idx + 1) % accumulation_steps == 0:
#                         optimizer.step()
#                         optimizer.zero_grad()
#                 else:
#                     scaler.scale(loss).backward()
#                     if (batch_idx + 1) % accumulation_steps == 0:
#                         scaler.unscale_(optimizer)
#                         optimizer.step()
#                         scaler.update()
#                         optimizer.zero_grad()

#                 running_loss += loss.item() * accumulation_steps
                
#                 # ** Add this memory check every 50 batches **
#                 if device.type == 'cuda' and batch_idx % 50 == 0:
#                     allocated_memory = torch.cuda.memory_allocated(device) / 1e6
#                     logging.info(f"Allocated GPU memory after {batch_idx} batches: {allocated_memory} MB")

#             except RuntimeError as e:
#                 if "CUDA" in str(e):
#                     logging.error(f"CUDA error during training: {str(e)}")
#                     device = torch.device('cpu')
#                     model = model.cpu()
#                     batch_x = batch_x.cpu()
#                     batch_x_spectral = batch_x_spectral.cpu()
#                     batch_y = batch_y.cpu()
#                     continue
#                 else:
#                     raise e

#         # Validation phase
#         model.eval()
#         val_loss, val_accuracy, _ = evaluate_model(model, val_data, criterion, device)
        
#         # Update learning rate
#         scheduler.step(val_loss)
        
#         # Store metrics
#         metrics_history['train_loss'].append(running_loss/len(train_loader))
#         metrics_history['val_loss'].append(val_loss)
#         metrics_history['val_accuracy'].append(val_accuracy)
        
#         # Check early stopping
#         should_stop = early_stopping(
#             metrics={'loss': val_loss, 'accuracy': val_accuracy},
#             epoch=epoch,
#             state_dict=model.state_dict()
#         )
        
#         if verbose:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch+1}/{epochs} - "
#                   f"Loss: {running_loss/len(train_loader):.4f}, "
#                   f"Val Loss: {val_loss:.4f}, "
#                   f"Val Accuracy: {val_accuracy:.4f}, "
#                   f"LR: {current_lr:.6f}")
        
#         if should_stop:
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break
            
#     # Plot training history
#     plot_training_history(metrics_history)
    
#     return early_stopping.best_state, metrics_history['val_accuracy'][early_stopping.best_epoch]

def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, 
                device, epochs=100, accumulation_steps=4, verbose=True):
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=CONFIG['train_params']['initial']['early_stopping']['patience'],
        min_epochs=CONFIG['train_params']['initial']['early_stopping']['min_epochs'],
        min_delta=CONFIG['train_params']['initial']['early_stopping']['min_delta'],
        monitor=CONFIG['train_params']['initial']['early_stopping']['monitor'],
        mode='auto'
    )
    
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'class_accuracies': []  # Track per-class performance
    }
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Training phase
        model.train()
        running_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch_idx, (batch_x, batch_x_spectral, batch_y) in enumerate(train_loader):
            try:
                batch_x = batch_x.to(device)
                batch_x_spectral = batch_x_spectral.to(device)
                batch_y = batch_y.to(device)
                
                # Perform mixed precision training with autocast
                with autocast(device_type=device.type if device.type != 'cpu' else 'cpu'):
                    outputs = model(batch_x, batch_x_spectral)
                    loss = criterion(outputs, batch_y)
                    loss = loss / accumulation_steps
                
                if device.type == 'cpu':
                    loss.backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                running_loss += loss.item() * accumulation_steps
                
                # Store predictions and true labels for per-class metrics
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                true_labels.extend(batch_y.cpu().numpy())
                
                if device.type == 'cuda' and batch_idx % 50 == 0:
                    allocated_memory = torch.cuda.memory_allocated(device) / 1e6
                    logging.info(f"Allocated GPU memory after {batch_idx} batches: {allocated_memory} MB")

            except RuntimeError as e:
                if "CUDA" in str(e):
                    logging.error(f"CUDA error during training: {str(e)}")
                    device = torch.device('cpu')
                    model = model.cpu()
                    batch_x = batch_x.cpu()
                    batch_x_spectral = batch_x_spectral.cpu()
                    batch_y = batch_y.cpu()
                    continue
                else:
                    raise e

        # Calculate per-class metrics for training
        train_class_accuracies = {}
        for class_idx in range(len(torch.unique(batch_y))):
            mask = np.array(true_labels) == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(np.array(predictions)[mask] == class_idx)
                train_class_accuracies[f'class_{class_idx}'] = class_acc

        # Validation phase
        model.eval()
        val_loss, val_accuracy, val_predictions = evaluate_model(
            model, val_data, criterion, device
        )
        
        # Calculate per-class metrics for validation
        val_class_accuracies = {}
        for class_idx in range(len(torch.unique(val_data[2]))):
            mask = val_data[2].cpu().numpy() == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(val_predictions[mask] == class_idx)
                val_class_accuracies[f'class_{class_idx}'] = class_acc
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        metrics_history['train_loss'].append(running_loss/len(train_loader))
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_accuracy'].append(val_accuracy)
        metrics_history['class_accuracies'].append(val_class_accuracies)
        
        # Check early stopping
        should_stop = early_stopping(
            metrics={'loss': val_loss, 'accuracy': val_accuracy},
            epoch=epoch,
            state_dict=model.state_dict()
        )
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Loss: {running_loss/len(train_loader):.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print("Per-class validation accuracies:")
            for class_name, acc in val_class_accuracies.items():
                print(f"{class_name}: {acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Plot training history with per-class metrics
    plot_training_history(metrics_history)
    
    return early_stopping.best_state, metrics_history['val_accuracy'][early_stopping.best_epoch]

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