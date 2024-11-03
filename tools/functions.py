import torch
import numpy as np
import scipy.io as sio
import logging
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tools.classes import ( EnsembleModel, EarlyStopping, SleepStageLoss, TrainingMonitor, 
EarlyStoppingWithClassMetrics, SleepStageEvaluator, DynamicBatchSampler )
from tools.config import *
from tools.utils import *
import random
import math
import os
import optuna
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


# Add these helper functions at the beginning:

def validate_dimensions(X, X_spectral, y):
    """Validate input dimensions and consistency"""
    if X.ndim not in [2, 3]:
        raise ValueError(f"Expected X to have 2 or 3 dimensions, got {X.ndim}")
    if X_spectral.ndim != 2:
        raise ValueError(f"Expected X_spectral to have 2 dimensions, got {X_spectral.ndim}")
    if len(X) != len(y) or len(X_spectral) != len(y):
        raise ValueError("Length mismatch between X, X_spectral, and y")

def ensure_3d(x, name="array"):
    """Ensure array is 3D with shape (samples, channels, features)"""
    if x.ndim == 2:
        logging.info(f"Reshaping {name} from 2D to 3D")
        return x.reshape(x.shape[0], 1, -1)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f"Cannot convert {name} with {x.ndim} dimensions to 3D")

# def load_data(data_paths, verbose=False):
#     """
#     Load and combine any number of nights of sleep data
    
#     Args:
#         data_paths: List of paths to .mat files containing sleep data
        
#     Returns:
#         tuple: Combined features, labels, and night indices
#     """
#     combined_x = []
#     combined_y = []
#     night_indices = []  # Track which night each sample comes from
    
#     for night_idx, data_path in enumerate(data_paths):
#         try:
#             mat_file = sio.loadmat(data_path)
            
#             # Stack the signals for current night
#             x = np.stack((mat_file['sig1'], mat_file['sig2'], 
#                          mat_file['sig3'], mat_file['sig4']), axis=1)
#             x = torch.from_numpy(x).float()
            
#             # Get labels for current night
#             y = torch.from_numpy(mat_file['labels'].flatten()).long()
            
#             # Filter valid indices
#             valid_indices = y != -1
#             x = x[valid_indices]
#             y = y[valid_indices]
            
#             if x.dim() == 2:
#                 x = x.unsqueeze(1)
            
#             # Add data from this night
#             combined_x.append(x)
#             combined_y.append(y)
#             night_indices.extend([night_idx] * len(y))
            
#             if verbose:
#                 logging.info(f"Loaded data from {data_path}")
#                 logging.info(f"Night {night_idx + 1} data shape: {x.shape}, Labels shape: {y.shape}")
#                 logging.info(f"Class distribution for night {night_idx + 1}:\n    {format_class_distribution(Counter(y.numpy()))}")
        
#         except OSError as e:
#             logging.error(f"Error loading data from {data_path}: {e}")
#             continue  # Skip this file and move on to the next
#         except Exception as e:
#             logging.error(f"Error loading data from {data_path}: {e}")
#             raise
    
#     try:
#         combined_x = torch.cat(combined_x, dim=0)
#         combined_y = torch.cat(combined_y, dim=0)
#         night_indices = torch.tensor(night_indices)
        
#         logging.info(f"Combined data shape: {combined_x.shape}")
#         logging.info(f"Combined labels shape: {combined_y.shape}")
#         logging.info(f"Overall class distribution:\n    {format_class_distribution(Counter(combined_y.numpy()))}")
        
#         return combined_x, combined_y, night_indices
        
#     except Exception as e:
#         logging.error(f"Error combining data: {e}")
#         raise

def load_data(data_paths, verbose=False):
    """
    Load and combine multiple nights of sleep data with variable lengths
    
    Args:
        data_paths: List of paths to .mat files containing sleep data
        verbose: Whether to print detailed loading information
        
    Returns:
        tuple: (combined_x, combined_y, night_indices, night_lengths)
            - combined_x: Tensor containing all EEG signals
            - combined_y: Tensor containing all sleep stage labels
            - night_indices: Tensor mapping each epoch to its night
            - night_lengths: Dictionary mapping night indices to number of epochs
    """
    combined_x = []
    combined_y = []
    night_indices = []
    night_lengths = {}
    total_epochs = 0
    valid_nights = 0
    
    logging.info(f"Loading data from {len(data_paths)} nights...")
    
    for night_idx, data_path in enumerate(data_paths):
    # for night_idx, path in enumerate(tqdm(data_paths, desc="Loading files", disable=not verbose)):
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
            
            # Record night length and update counters
            n_epochs = len(y)
            night_lengths[night_idx] = n_epochs
            total_epochs += n_epochs
            valid_nights += 1
            
            # Add data from this night
            combined_x.append(x)
            combined_y.append(y)
            night_indices.extend([night_idx] * n_epochs)
        
        #     if not verbose:
        #                 continue  # Skip detailed logging
        # except Exception as e:
        #     logging.error(f"Error loading {path}: {e}")
        #     continue
            
            if verbose:
                logging.info(f"\nNight {night_idx + 1}:")
                logging.info(f"    Source: {os.path.basename(data_path)}")
                logging.info(f"    Epochs: {n_epochs}")
                logging.info(f"    Data shape: {x.shape}")
                logging.info(f"    Class distribution:\n        " + 
                           format_class_distribution(Counter(y.numpy())).replace('\n', '\n        '))
        
        except OSError as e:
            logging.error(f"Error loading data from {data_path}: {e}")
            continue  # Skip this file and move on to the next
        except Exception as e:
            logging.error(f"Error loading data from {data_path}: {e}")
            raise
    
    try:
        # Combine all data
        combined_x = torch.cat(combined_x, dim=0)
        combined_y = torch.cat(combined_y, dim=0)
        night_indices = torch.tensor(night_indices)
        
        # Calculate statistics
        avg_epochs = total_epochs / valid_nights if valid_nights > 0 else 0
        epoch_std = np.std([length for length in night_lengths.values()])
        min_epochs = min(night_lengths.values())
        max_epochs = max(night_lengths.values())
        
        # Log overall statistics
        logging.info("\nData Loading Summary:")
        logging.info(f"    Total nights processed: {valid_nights}")
        logging.info(f"    Total epochs: {total_epochs}")
        logging.info(f"    Average epochs per night: {avg_epochs:.1f} ± {epoch_std:.1f}")
        logging.info(f"    Range: {min_epochs} to {max_epochs} epochs")

        if verbose:
            logging.info("\nNight lengths (epochs):")
            for night_idx, length in sorted(night_lengths.items()):
                logging.info(f"    Night {night_idx + 1}: {length}")
        
        logging.info("\nOverall Data Shapes:")
        logging.info(f"    Combined data: {combined_x.shape}")
        logging.info(f"    Combined labels: {combined_y.shape}")
        
        logging.info("\nOverall Class Distribution:")
        logging.info("    " + format_class_distribution(Counter(combined_y.numpy())).replace('\n', '\n    '))
        
        # Verify data integrity
        if len(combined_y) != total_epochs:
            raise ValueError(f"Mismatch in total epochs: {len(combined_y)} vs {total_epochs}")
        if len(night_indices) != total_epochs:
            raise ValueError(f"Mismatch in night indices: {len(night_indices)} vs {total_epochs}")
        
        
        return combined_x, combined_y, night_indices, night_lengths
        
    except Exception as e:
        logging.error(f"Error combining data: {e}")
        raise




def calculate_segment_weights(segments, target_samples):
    """Helper to calculate weights for segment distribution"""
    weights = []
    for segment in segments:
        # Consider both segment length and target samples
        segment_length = len(segment)
        weight = segment_length / target_samples if target_samples > 0 else 0
        weights.append(1 - weight)  # Inverse weight - prefer smaller segments for better distribution
    return np.array(weights)

def softmax(x):
    """Compute softmax values for array x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def calculate_oversampling_targets(class_counts, strategy='proportional'):
    """
    Calculate target counts for oversampling based on different strategies
    
    Args:
        class_counts: Dictionary of current class counts
        strategy: 'balanced' or 'proportional'
    
    Returns:
        Dictionary of target counts per class
    """
    total_samples = sum(class_counts.values())
    max_class_count = max(class_counts.values())
    
    if strategy == 'balanced':
        # All classes will have the same count as the majority class
        return {cls: max_class_count for cls in class_counts.keys()}
    else:
        # Maintain relative proportions while increasing minority classes
        desired_total = total_samples * 1.5  # Increase total samples by 50%
        class_ratios = {
            cls: count / total_samples 
            for cls, count in class_counts.items()
        }
        return {
            cls: max(count, int(desired_total * ratio))
            for cls, (count, ratio) in zip(
                class_counts.keys(), 
                class_ratios.items()
            )
        }

def prepare_data_multi_night(x, y, night_indices, night_lengths, train_size=0.7, val_size=0.15, test_size=0.15):
    """Enhanced data preparation with proper tensor and integer handling"""
    # Validate split ratios
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    # Initial class distribution logging
    initial_dist = Counter(y.numpy())
    logging.info("\nInitial class distribution:")
    for cls in sorted(initial_dist.keys()):
        count = initial_dist[cls]
        percentage = count / len(y) * 100
        logging.info(f"    {SLEEP_STAGES[cls]}: {count} ({percentage:.1f}%)")
    
    try:
        # Ensure all inputs are proper tensors
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        night_indices = torch.as_tensor(night_indices, dtype=torch.long)
        
        # Get unique nights as integers
        unique_nights = night_indices.unique().cpu().numpy()
        n_nights = len(unique_nights)
        
        # Calculate split sizes
        n_train = int(n_nights * train_size)
        n_val = int(n_nights * val_size)
        
        # Build night distribution dictionary
        night_class_dist = {}
        for night in unique_nights:
            night_mask = night_indices == night
            night_class_dist[night] = Counter(y[night_mask].cpu().numpy())
        
        # Calculate distribution similarity
        overall_dist = Counter(y.cpu().numpy())
        total_samples = len(y)
        
        def get_dist_similarity(night):
            night_dist = night_class_dist[night]
            similarity = 0
            for cls in overall_dist:
                expected_prop = overall_dist[cls] / total_samples
                night_prop = night_dist.get(cls, 0) / sum(night_dist.values())
                similarity += abs(expected_prop - night_prop)
            return similarity
        
        # Sort and shuffle nights
        sorted_nights = sorted(unique_nights, key=get_dist_similarity)
        np.random.shuffle(sorted_nights)  # In-place shuffle
        
        # Split nights
        train_nights = set(sorted_nights[:n_train])
        val_nights = set(sorted_nights[n_train:n_train+n_val])
        test_nights = set(sorted_nights[n_train+n_val:])
        
        # Create boolean masks
        train_mask = torch.tensor([n.item() in train_nights for n in night_indices])
        val_mask = torch.tensor([n.item() in val_nights for n in night_indices])
        test_mask = torch.tensor([n.item() in test_nights for n in night_indices])
        
        # Split data
        X_train = x[train_mask]
        y_train = y[train_mask]
        X_val = x[val_mask]
        y_val = y[val_mask]
        X_test = x[test_mask]
        y_test = y[test_mask]
        
        # Calculate target count for oversampling
        train_class_counts = Counter(y_train.numpy())


        
        # Calculate targets that maintain relative proportions
        target_counts = calculate_oversampling_targets(
            train_class_counts,
            strategy='proportional'  # or 'balanced' for equal class sizes
        )

        logging.info("\nClass distribution details:")
        total_samples = sum(train_class_counts.values())
        for cls in sorted(train_class_counts.keys()):
            current = train_class_counts[cls]
            target = target_counts[cls]
            orig_ratio = current / total_samples
            new_ratio = target / sum(target_counts.values())
            
            logging.info(f"    {SLEEP_STAGES[cls]}:")
            logging.info(f"        Current: {current} ({orig_ratio:.1%})")
            logging.info(f"        Target:  {target} ({new_ratio:.1%})")
            logging.info(f"        Increase factor: {target/current:.2f}x")
       
        # Extract spectral features
        logging.info("\nExtracting spectral features...")
        X_train_spectral = torch.tensor(extract_spectral_features_batch(X_train), dtype=torch.float32)
        X_val_spectral = torch.tensor(extract_spectral_features_batch(X_val), dtype=torch.float32)
        X_test_spectral = torch.tensor(extract_spectral_features_batch(X_test), dtype=torch.float32)
        
        # Apply oversampling
        logging.info("\nApplying oversampling to training data...")
        X_train_resampled, X_train_spectral_resampled, y_train_resampled = improved_oversample(
            X_train, X_train_spectral, y_train, target_count=None
        )
        
        # Log final distributions
        splits = {
            "Training": (y_train_resampled, len(y_train_resampled)),
            "Validation": (y_val, len(y_val)),
            "Testing": (y_test, len(y_test))
        }
        
        for name, (y_split, total) in splits.items():
            dist = Counter(y_split.cpu().numpy())
            logging.info(f"\n{name} split distribution:")
            for cls in sorted(dist.keys()):
                count = dist[cls]
                percentage = count / total * 100
                logging.info(f"    {SLEEP_STAGES[cls]}: {count} ({percentage:.1f}%)")
        
        return (
            X_train_resampled, X_train_spectral_resampled, y_train_resampled,
            X_val, X_val_spectral, y_val,
            X_test, X_test_spectral, y_test
        )
        
    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise

# def prepare_data_multi_night(x, y, night_indices, night_lengths, train_size=0.7, val_size=0.15, test_size=0.15):
#     """
#     Prepare data with balanced class distribution across splits while maintaining temporal relationships
    
#     Args:
#         x: Input features tensor
#         y: Labels tensor
#         night_indices: Tensor indicating which night each sample belongs to
#         night_lengths: Dictionary mapping night indices to number of epochs
#         train_size: Proportion of data for training set
#         val_size: Proportion of data for validation set
#         test_size: Proportion of data for test set
#     """
#     if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-9):
#         raise ValueError(f"Split ratios must sum to 1.0 (got {train_size + val_size + test_size})")
    
#     x_np = x.cpu().numpy()
#     y_np = y.cpu().numpy()
#     night_indices_np = night_indices.cpu().numpy()
    
#     total_samples = len(y_np)
#     class_counts = Counter(y_np)
    
#     # Verify all classes are present
#     expected_classes = set(range(len(SLEEP_STAGES)))  # 0 to 4
#     found_classes = set(class_counts.keys())
#     if found_classes != expected_classes:
#         logging.warning(f"Missing classes in data: {expected_classes - found_classes}")
    
#     logging.info(f"Class distribution before split:\n    {format_class_distribution(class_counts)}")

#     try:
#         unique_nights = np.unique(night_indices_np)
        
#         # Analyze class distribution per night
#         night_class_stats = {}
#         for night in unique_nights:
#             night_mask = night_indices_np == night
#             night_y = y_np[night_mask]
#             night_class_stats[night] = Counter(night_y)
            
#             logging.info(f"\nNight {night} class distribution:")
#             logging.info("    " + format_class_distribution(night_class_stats[night]))
        
#         # Identify rare classes (less than 1% of total samples)
#         rare_class_threshold = total_samples * 0.01
#         rare_classes = {cls: count for cls, count in class_counts.items() 
#                        if count < rare_class_threshold}
        
#         if rare_classes:
#             logging.info("\nIdentified rare classes:")
#             for cls, count in rare_classes.items():
#                 logging.info(f"    {SLEEP_STAGES[cls]}: {count} samples ({count/total_samples:.1%})")
        
#         # Initialize containers for class-wise segments
#         class_segments = {cls: [] for cls in SLEEP_STAGES.keys()}
        
#         # Find continuous segments for each class
#         for night in unique_nights:
#             night_mask = night_indices_np == night
#             night_indices_local = np.where(night_mask)[0]
#             night_y = y_np[night_indices_local]
            
#             for class_label in SLEEP_STAGES.keys():
#                 class_mask = night_y == class_label
#                 if not np.any(class_mask):
#                     continue
                
#                 # Find continuous segments
#                 segment_starts = np.where(np.diff(np.concatenate(([0], class_mask))) == 1)[0]
#                 segment_ends = np.where(np.diff(np.concatenate((class_mask, [0]))) == -1)[0]
                
#                 for start, end in zip(segment_starts, segment_ends):
#                     segment = night_indices_local[start:end + 1]
#                     min_length = 1 if class_label in rare_classes else 3
#                     if len(segment) >= min_length:
#                         class_segments[class_label].append(segment.tolist())
        
#         # Calculate target samples per class per split
#         target_samples = {
#             'train': {cls: max(int(count * train_size), 1) for cls, count in class_counts.items()},
#             'val': {cls: max(int(count * val_size), 1) for cls, count in class_counts.items()},
#             'test': {cls: max(int(count * test_size), 1) for cls, count in class_counts.items()}
#         }
        
#         # Initialize split indices
#         split_indices = {
#             'train': [],
#             'val': [],
#             'test': []
#         }
        
#         # Process each class
#         for class_label in SLEEP_STAGES.keys():
#             if not class_segments[class_label]:
#                 logging.warning(f"No segments found for class {SLEEP_STAGES[class_label]}")
#                 continue
            
#             segments = class_segments[class_label]
#             np.random.shuffle(segments)  # Shuffle while keeping segments intact
            
#             # Ensure rare classes are represented in training
#             # if class_label in rare_classes:
#             #     # Allocate all segments to training
#             #     split_indices['train'].extend([idx for segment in segments for idx in segment])
#             if class_label in rare_classes:
#                 # Distribute rare classes across all splits while maintaining temporal relationships
#                 n_segments = len(segments)
#                 train_end = int(n_segments * 0.8)  # Bias towards training
#                 val_end = int(n_segments * 0.9)
                
#                 split_indices['train'].extend([idx for seg in segments[:train_end] for idx in seg])
#                 split_indices['val'].extend([idx for seg in segments[train_end:val_end] for idx in seg])
#                 split_indices['test'].extend([idx for seg in segments[val_end:] for idx in seg])
#             else:
#                 # Calculate split points
#                 total_segments = len(segments)
#                 n_train = max(int(total_segments * train_size), 1)
#                 n_val = max(int(total_segments * val_size), 1)
                
#                 # Distribute segments
#                 train_segments = segments[:n_train]
#                 val_segments = segments[n_train:n_train + n_val]
#                 test_segments = segments[n_train + n_val:]
                
#                 # Add to split indices
#                 for segment in train_segments:
#                     split_indices['train'].extend(segment)
#                 for segment in val_segments:
#                     split_indices['val'].extend(segment)
#                 for segment in test_segments:
#                     split_indices['test'].extend(segment)
        
#         # Convert to numpy arrays
#         train_indices = np.array(split_indices['train'], dtype=np.int64)
#         val_indices = np.array(split_indices['val'], dtype=np.int64)
#         test_indices = np.array(split_indices['test'], dtype=np.int64)
        
#         # Handle empty splits
#         if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
#             logging.warning("Empty splits detected, redistributing samples")
#             all_indices = np.concatenate([train_indices, val_indices, test_indices])
#             np.random.shuffle(all_indices)
            
#             n_train = int(len(all_indices) * train_size)
#             n_val = int(len(all_indices) * val_size)
            
#             train_indices = all_indices[:n_train]
#             val_indices = all_indices[n_train:n_train + n_val]
#             test_indices = all_indices[n_train + n_val:]
        
#         # Create final splits
#         X_train = x_np[train_indices]
#         y_train = y_np[train_indices]
#         X_val = x_np[val_indices]
#         y_val = y_np[val_indices]
#         X_test = x_np[test_indices]
#         y_test = y_np[test_indices]
        
#         # Validate class distribution in splits
#         for split_name, y_split in [
#             ('Training', y_train),
#             ('Validation', y_val),
#             ('Testing', y_test)
#         ]:
#             split_dist = Counter(y_split)
#             missing_classes = set(range(len(SLEEP_STAGES))) - set(split_dist.keys())
#             logging.info(f"\n{split_name} split distribution:")
#             logging.info(f"    Samples: {len(y_split)} ({len(y_split)/total_samples:.1%})")
#             logging.info("    " + format_class_distribution(split_dist))
#             if missing_classes:
#                 logging.warning(f"{split_name} split is missing classes: {missing_classes}")
        
#         # Extract spectral features
#         X_train_spectral = np.array([extract_spectral_features(torch.from_numpy(x)) for x in X_train])
#         X_val_spectral = np.array([extract_spectral_features(torch.from_numpy(x)) for x in X_val])
#         X_test_spectral = np.array([extract_spectral_features(torch.from_numpy(x)) for x in X_test])
        
#         # Apply improved oversampling to training set
#         X_train_resampled, X_train_spectral_resampled, y_resampled = improved_oversample(
#             torch.from_numpy(X_train).float(),
#             torch.from_numpy(X_train_spectral).float(),
#             torch.from_numpy(y_train),
#             max_ratio=3.0,
#             min_samples=50
#         )
        
#         return (X_train_resampled,
#                 X_train_spectral_resampled,
#                 y_resampled,
#                 torch.from_numpy(X_val).float(),
#                 torch.from_numpy(X_val_spectral).float(),
#                 torch.from_numpy(y_val).long(),
#                 torch.from_numpy(X_test).float(),
#                 torch.from_numpy(X_test_spectral).float(),
#                 torch.from_numpy(y_test).long())
                
#     except Exception as e:
#         logging.error(f"Error in data preparation: {str(e)}")
#         logging.error("Detailed error information:")
#         import traceback
#         traceback.print_exc()
#         raise


def preprocess_data(X, X_spectral):
    """
    Light preprocessing for already processed data
    
    Args:
        X: Signal data that's already been through batch_preprocessing.py
        X_spectral: Spectral features
    """
    # Just ensure data types and ranges are correct
    X = torch.clamp(X, -5, 5)  # Conservative clipping since data is already scaled
    X_spectral = torch.clamp(X_spectral, min=0)  # Spectral powers are non-negative
    
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
    """
    Initialize model with comprehensive GPU memory checking and fallback handling
    
    Args:
        model_params: Dictionary of model parameters
        device: Target device for model
        
    Returns:
        tuple: (initialized model, device to use)
    """
    try:
        if device.type == 'cuda':
            # Clear GPU cache and log initial memory state
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            logging.info(f"\nInitializing model on {device}:")
            logging.info(f"    Available memory: {(total_memory - initial_memory)/1e9:.2f}GB")
            
            # Estimate model size
            temp_model = EnsembleModel(model_params)
            model_size = sum(p.numel() * p.element_size() for p in temp_model.parameters())
            estimated_memory = model_size * 3  # Account for optimizer and gradients
            
            logging.info(f"    Estimated model memory: {model_size/1e9:.2f}GB")
            logging.info(f"    Estimated total usage: {estimated_memory/1e9:.2f}GB")
            
            if estimated_memory > (total_memory - initial_memory) * 0.9:  # 90% threshold
                logging.warning("Insufficient GPU memory for safe model initialization")
                device = torch.device('cpu')
                logging.info("Falling back to CPU")
            
            del temp_model
            torch.cuda.empty_cache()
        
        # Initialize model with proper error handling
        try:
            model = EnsembleModel(model_params).to(device)
            
            # Initialize weights
            model.apply(model._init_weights)
            
            # Verify model initialization
            try:
                with torch.no_grad():
                    # Test forward pass with small batch
                    test_input = torch.randn(2, 4, 3000).to(device)
                    test_spectral = torch.randn(2, 16).to(device)
                    _ = model(test_input, test_spectral)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                        
                logging.info("Model initialization successful")
                    
            except Exception as e:
                logging.error(f"Model verification failed: {str(e)}")
                raise
            
            return model, device
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("GPU OOM during model initialization")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                device = torch.device('cpu')
                model = EnsembleModel(model_params).to(device)
                model.apply(model._init_weights)
                return model, device
            raise
            
    except Exception as e:
        logging.error(f"Error in model initialization: {str(e)}")
        raise

def verify_model_integrity(model, device):
    """
    Verify model integrity with comprehensive checks
    
    Args:
        model: The initialized model
        device: Device the model is on
    """
    try:
        model.eval()
        with torch.no_grad():
            # Test various input sizes
            batch_sizes = [1, 2, 4]
            for batch_size in batch_sizes:
                test_input = torch.randn(batch_size, 4, 3000).to(device)
                test_spectral = torch.randn(batch_size, 16).to(device)
                outputs = model(test_input, test_spectral)
                
                # Verify output shape
                expected_shape = (batch_size, len(SLEEP_STAGES))
                assert outputs.shape == expected_shape, \
                    f"Invalid output shape: {outputs.shape}, expected {expected_shape}"
                
                # Verify output values
                assert not torch.isnan(outputs).any(), "Model produced NaN values"
                assert not torch.isinf(outputs).any(), "Model produced infinite values"
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
        logging.info("Model integrity verification passed")
        
    except Exception as e:
        logging.error(f"Model integrity verification failed: {str(e)}")
        raise

def setup_training_components(model, train_loader, config):
    """
    Set up training components with proper initialization
    
    Args:
        model: The initialized model
        train_loader: Training data loader
        config: Configuration dictionary
        
    Returns:
        tuple: (optimizer, scheduler, criterion)
    """
    try:
        # Get class weights for loss function
        labels = torch.tensor([y for _, _, y in train_loader.dataset])
        class_weights = get_class_weights(labels).to(model.device)
        
        # Initialize criterion
        criterion = nn.CrossEntropyLoss(
            weight=class_weights + 1e-6,
            label_smoothing=0.1
        )
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['train_params']['initial']['lr'],
            weight_decay=1e-5,
            eps=1e-8
        )
        
        # Initialize scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['train_params']['scheduler']['factor'],
            patience=config['train_params']['scheduler']['patience'],
            min_lr=config['train_params']['scheduler']['min_lr'],
            verbose=config['train_params']['scheduler']['verbose']
        )
        
        return optimizer, scheduler, criterion
        
    except Exception as e:
        logging.error(f"Error setting up training components: {str(e)}")
        raise


def extract_spectral_features(x):
    """
    Extract spectral features optimized for your preprocessed data
    
    Args:
        x: Input signal of shape (channels, timepoints) that has been:
           - Bandpass filtered (0.5-50 Hz)
           - Downsampled to 100 Hz
           - Bipolar referenced
           - Artifact cleaned
           - Scaled to target IQR and median
    """
    features = []
    sample_rate = 100  # From your preprocessing
    
    # Define frequency bands matching your preprocessing
    freq_bands = {
        'delta': (0.5, 4),    # Lower bound matches your bandpass
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)     # Upper bound matches your bandpass
    }
    
    for channel in range(x.shape[0]):  # 4 bipolar channels
        channel_data = x[channel].cpu().numpy()
        
        # Use window length that matches your epoch duration
        nperseg = min(3000, len(channel_data))  # 30s * 100Hz = 3000 samples
        
        # Compute PSD with parameters matching your preprocessing
        f, psd = welch(channel_data, fs=sample_rate, 
                      nperseg=nperseg,
                      noverlap=nperseg//2)
        
        # Extract powers in your filtered bands
        for band_name, (low, high) in freq_bands.items():
            mask = (f >= low) & (f <= high)
            if np.any(mask):
                band_power = np.sum(psd[mask])
                features.append(band_power)
    
    return np.array(features)

def extract_spectral_features_batch(X):
    """
    Batched version of spectral feature extraction
    
    Args:
        X: Input data of shape (batch_size, channels, samples)
        
    Returns:
        Array of spectral features for the batch
    """
    if isinstance(X, np.ndarray):  # Convert each sample to tensor if it’s numpy
            X = torch.from_numpy(X)

    batch_size = len(X)
    features_list = []
    
    # Process in smaller batches to manage memory
    batch_size_proc = 100  # Process 100 samples at a time
    
    
    for i in range(0, batch_size, batch_size_proc):
        
        end_idx = min(i + batch_size_proc, batch_size)
        batch = X[i:end_idx]
        
        # Process each sample in the mini-batch
        batch_features = []
        for sample in batch:
            features = extract_spectral_features(sample)
            batch_features.append(features)
        
        features_list.extend(batch_features)
    
    return np.array(features_list)



def time_warp(x, sigma=0.2, knot=4):
    """
    Apply time warping to EEG signals while preserving signal characteristics
    
    Args:
        x: Input signal of shape (batch_size, channels, samples) or (channels, samples)
        sigma: Warping intensity (smaller values = less distortion)
        knot: Number of control points for warping
    
    Returns:
        Warped signal with same shape as input
    """
    # Convert input to numpy if it's a tensor
    if torch.is_tensor(x):
        is_tensor = True
        x = x.cpu().numpy()
    else:
        is_tensor = False
    
    # Ensure input is 3D: (batch_size, channels, samples)
    if x.ndim == 2:
        x = x.reshape(1, *x.shape)
    
    batch_size, n_channels, n_samples = x.shape
    ret = np.zeros_like(x)
    
    # Create base time steps
    orig_steps = np.arange(n_samples, dtype=np.float32)
    
    try:
        for i in range(batch_size):
            for ch in range(n_channels):
                # Generate knot points with better spacing
                knot_points = np.linspace(0, n_samples - 1, num=knot + 2)
                
                # Generate smoother warping pattern
                random_warps = np.random.normal(loc=1.0, scale=sigma, size=knot + 2)
                random_warps = np.abs(random_warps)  # Ensure positive values
                random_warps[0] = 1.0  # Fix endpoints
                random_warps[-1] = 1.0
                
                # Apply smoothing to warps
                random_warps = np.convolve(random_warps, [0.2, 0.6, 0.2], mode='same')
                
                # Create cumulative warps
                random_warps = np.cumsum(random_warps)
                random_warps = random_warps / random_warps[-1]
                
                # Generate warped knots
                warped_knots = knot_points * random_warps
                
                # Ensure strict monotonicity
                eps = 1e-6
                for j in range(1, len(warped_knots)):
                    if warped_knots[j] <= warped_knots[j-1]:
                        warped_knots[j] = warped_knots[j-1] + eps
                
                try:
                    # Create smoother interpolation
                    warp_func = CubicSpline(
                        knot_points, 
                        warped_knots,
                        bc_type='clamped'
                    )
                    
                    # Generate warped time points
                    time_warp = warp_func(orig_steps)
                    time_warp = np.maximum.accumulate(time_warp)
                    time_warp = time_warp * (n_samples - 1) / time_warp[-1]
                    
                    # Preserve signal characteristics
                    channel_data = x[i, ch]
                    warped_data = np.interp(
                        orig_steps,
                        np.clip(time_warp, 0, n_samples - 1),
                        channel_data
                    )
                    
                    # Normalize to maintain signal amplitude
                    orig_std = np.std(channel_data)
                    warped_std = np.std(warped_data)
                    if warped_std > 0:
                        warped_data = warped_data * (orig_std / warped_std)
                    
                    ret[i, ch] = warped_data
                    
                except ValueError as e:
                    logging.warning(f"Interpolation failed: {str(e)}. Using original signal.")
                    ret[i, ch] = x[i, ch]
                    
    except Exception as e:
        logging.error(f"Time warping failed: {str(e)}")
        return torch.from_numpy(x) if is_tensor else x
    
    # Return tensor if input was tensor
    return torch.from_numpy(ret) if is_tensor else ret


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
            augmented = time_warp(x[i].unsqueeze(0), sigma=0.3, knot=5).squeeze(0)
            augmented_x.append(augmented)
            augmented_x_spectral.append(x_spectral[i])  # Duplicate spectral features for augmented data
            augmented_y.append(y[i])
    return torch.stack(augmented_x), torch.stack(augmented_x_spectral), torch.tensor(augmented_y)

def compute_dynamic_importance_factors(class_counts):
    """
    Compute class importance factors based on distribution
    
    Args:
        class_counts: Dictionary of class frequencies
        
    Returns:
        Dictionary of importance factors per class
    """
    total_samples = sum(class_counts.values())
    class_ratios = {
        cls: count/total_samples 
        for cls, count in class_counts.items()
    }
    
    # Compute relative scarcity
    median_ratio = np.median(list(class_ratios.values()))
    importance_factors = {
        cls: np.log1p(median_ratio/ratio)
        for cls, ratio in class_ratios.items()
    }
    
    # Normalize factors
    max_factor = max(importance_factors.values())
    importance_factors = {
        cls: factor/max_factor + 1.0
        for cls, factor in importance_factors.items()
    }
    
    return importance_factors



def create_balanced_data_loaders(X, X_spectral, y, batch_size, is_train=True):
    """
    Create data loaders with balanced class sampling
    """
    dataset = TensorDataset(X, X_spectral, y)
    
    if is_train:
        sampler = DynamicBatchSampler(y, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

# Add these to functions.py

def get_class_weights(y):
    """
    Compute class weights using effective numbers approach
    
    Args:
        y: Labels tensor
        
    Returns:
        Tensor of class weights
    """
    class_counts = torch.bincount(y)
    total = class_counts.sum()
    
    # Use effective numbers
    beta = 0.9999
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    
    # Normalize weights
    weights = weights / weights.sum() * len(class_counts)
    
    logging.info("\nComputed class weights:")
    for i, weight in enumerate(weights):
        logging.info(f"    Class {i}: {weight:.4f}")
    
    return weights

def create_data_loaders(X, X_spectral, y, batch_size, is_train=True):
    """
    Create data loaders with dynamic batch sampling for training
    """
    dataset = TensorDataset(X, X_spectral, y)
    
    if is_train:
        # Use dynamic batch sampling for training
        sampler = DynamicBatchSampler(
            labels=y,
            batch_size=batch_size,
            drop_last=False
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

def improved_oversample(X, X_spectral, y, target_count=None):
    """
    Enhanced oversampling with proper tensor handling
    """
    # Ensure all inputs are tensors
    X = torch.as_tensor(X, dtype=torch.float32)
    X_spectral = torch.as_tensor(X_spectral, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    
    class_counts = Counter(y.numpy())
    
    if target_count is None:
        target_count = max(class_counts.values())
    
    logging.info("\nOversampling targets:")
    for cls in sorted(class_counts.keys()):
        current = class_counts[cls]
        logging.info(f"    {SLEEP_STAGES[cls]}: {current} → {target_count}")
    
    oversampled_data = []
    
    for cls in sorted(class_counts.keys()):
        class_mask = y == cls
        X_class = X[class_mask]
        X_spectral_class = X_spectral[class_mask]
        current_count = len(X_class)
        
        # Add original samples
        oversampled_data.append((X_class, X_spectral_class, torch.full((current_count,), cls, dtype=torch.long)))
        
        if current_count < target_count:
            n_synthetic = target_count - current_count
            
            synthetic_X = []
            synthetic_X_spectral = []
            
            while len(synthetic_X) < n_synthetic:
                try:
                    # Sample with replacement
                    idx = torch.randint(0, current_count, (min(n_synthetic - len(synthetic_X), 100),))
                    samples = X_class[idx]
                    
                    # Apply augmentation
                    augmented = time_warp(samples)
                    
                    # Add noise
                    noise_mask = torch.rand(len(augmented)) < 0.5
                    if noise_mask.any():
                        noise = torch.randn_like(augmented[noise_mask]) * 0.02
                        augmented[noise_mask] += noise
                    
                    synthetic_X.append(augmented)
                    synthetic_X_spectral.append(X_spectral_class[idx])
                    
                except Exception as e:
                    logging.warning(f"Augmentation failed for class {cls}: {str(e)}")
                    continue
            
            if synthetic_X:
                synthetic_X = torch.cat(synthetic_X, dim=0)
                synthetic_X_spectral = torch.cat(synthetic_X_spectral, dim=0)
                
                # Trim to exact size needed
                if len(synthetic_X) > n_synthetic:
                    synthetic_X = synthetic_X[:n_synthetic]
                    synthetic_X_spectral = synthetic_X_spectral[:n_synthetic]
                
                oversampled_data.append((
                    synthetic_X,
                    synthetic_X_spectral,
                    torch.full((len(synthetic_X),), cls, dtype=torch.long)
                ))
    
    # Combine all data
    X_combined = torch.cat([x for x, _, _ in oversampled_data])
    X_spectral_combined = torch.cat([x_s for _, x_s, _ in oversampled_data])
    y_combined = torch.cat([y_cls for _, _, y_cls in oversampled_data])
    
    # Shuffle
    perm = torch.randperm(len(y_combined))
    X_combined = X_combined[perm]
    X_spectral_combined = X_spectral_combined[perm]
    y_combined = y_combined[perm]
    
    return X_combined, X_spectral_combined, y_combined


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

        # mean_accuracy = np.mean(cv_scores)
        # std_accuracy = np.std(cv_scores)

        # # Log detailed results
        # logging.info(f"""Trial completed:
        # Parameters: {model_params}
        # Batch size: {batch_size}
        # Learning rate: {lr:.2e}
        # Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}
        # Best fold accuracy: {max(cv_scores):.4f}
        # Worst fold accuracy: {min(cv_scores):.4f}
        # """)

        # return mean_accuracy

    # except Exception as e:
    # logging.error(f"Error in objective function: {str(e)}")
    # return 0.0

        # After calculating all fold scores
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        # Calculate and store per-class metrics from the best fold
        best_fold_idx = np.argmax(cv_scores)
        best_fold_metrics = fold_metrics[best_fold_idx]
        
        # Store metrics in trial user attributes
        trial.set_user_attr('metrics', best_fold_metrics['val_class_accuracies'][-1])

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
    """
    Run hyperparameter optimization with memory management and cross-validation
    
    Args:
        X: Input data
        X_spectral: Spectral features
        y: Target labels
        device: Device to run on
        n_trials: Number of optimization trials
        start_with_config: Whether to start with config parameters
    """
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=CONFIG['settings']['seed'])
    )
    
    # Wrap the objective function to include memory management
    def objective_wrapper(trial):
        try:
            # Clear memory before each trial
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log memory state at start of trial
            if device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                logging.info(f"\nStarting trial with memory state:")
                logging.info(f"    Allocated: {memory_allocated:.2f}GB")
                logging.info(f"    Reserved: {memory_reserved:.2f}GB")
            
            # Run the original objective function
            result = objective(
                trial=trial,
                X=X,
                X_spectral=X_spectral,
                y=y,
                device=device,
                n_folds=5,
                start_with_config=start_with_config
            )
            
            # Log trial results
            logging.info(f"\nTrial finished with accuracy: {result:.4f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Trial failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()
        
        finally:
            # Cleanup after trial
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
    
    # Run optimization with proper exception handling
    try:
        study.optimize(
            objective_wrapper,
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: logging.info(f"\nBest value so far: {study.best_value:.4f}")
            ]
        )
        
        # Log results
        logging.info("\nHyperparameter optimization completed:")
        logging.info(f"Best trial accuracy: {study.best_value:.4f}")
        logging.info("Best hyperparameters:")
        for param, value in study.best_params.items():
            logging.info(f"    {param}: {value}")
        
        # Create a complete parameter set
        best_params = {}
        if start_with_config:
            # Start with initial configuration
            best_params.update(CONFIG['model_params']['initial'])
            best_params.update(CONFIG['train_params']['initial'])
        
        # Update with optimized parameters
        best_params.update(study.best_params)
        
        return best_params
        
    except Exception as e:
        logging.error(f"Hyperparameter optimization failed: {str(e)}")
        logging.info("Falling back to default parameters")
        return {
            **CONFIG['model_params']['initial'],
            **CONFIG['train_params']['initial']
        }


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
#         'val_accuracy': [],
#         'class_accuracies': []  # Track per-class performance
#     }
    
#     for epoch in tqdm(range(epochs), desc="Training Progress"):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         predictions = []
#         true_labels = []
        
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
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                         scaler.step(optimizer)
#                         scaler.update()
#                         optimizer.zero_grad()

#                 running_loss += loss.item() * accumulation_steps
                
#                 # Store predictions and true labels for per-class metrics
#                 predictions.extend(outputs.argmax(dim=1).cpu().numpy())
#                 true_labels.extend(batch_y.cpu().numpy())
                
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

#         # Calculate per-class metrics for training
#         train_class_accuracies = {}
#         for class_idx in range(len(torch.unique(batch_y))):
#             mask = np.array(true_labels) == class_idx
#             if np.sum(mask) > 0:
#                 class_acc = np.mean(np.array(predictions)[mask] == class_idx)
#                 train_class_accuracies[f'class_{class_idx}'] = class_acc

#         # Validation phase
#         model.eval()
#         val_loss, val_accuracy, val_predictions = evaluate_model(
#             model, val_data, criterion, device
#         )
        
#         # Calculate per-class metrics for validation
#         val_class_accuracies = {}
#         for class_idx in range(len(torch.unique(val_data[2]))):
#             mask = val_data[2].cpu().numpy() == class_idx
#             if np.sum(mask) > 0:
#                 class_acc = np.mean(val_predictions[mask] == class_idx)
#                 val_class_accuracies[f'class_{class_idx}'] = class_acc
        
#         # Update learning rate
#         scheduler.step(val_loss)
        
#         # Store metrics
#         metrics_history['train_loss'].append(running_loss/len(train_loader))
#         metrics_history['val_loss'].append(val_loss)
#         metrics_history['val_accuracy'].append(val_accuracy)
#         metrics_history['class_accuracies'].append(val_class_accuracies)
        
#         # Check early stopping
#         should_stop = early_stopping(
#             metrics={'loss': val_loss, 'accuracy': val_accuracy},
#             epoch=epoch,
#             state_dict=model.state_dict()
#         )
        
#         if verbose:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch+1}/{epochs}")
#             print(f"Loss: {running_loss/len(train_loader):.4f}")
#             print(f"Val Loss: {val_loss:.4f}")
#             print(f"Val Accuracy: {val_accuracy:.4f}")
#             print("Per-class validation accuracies:")
#             # for class_name, acc in val_class_accuracies.items():
#             #     print(f"{class_name}: {acc:.4f}")
#             log_class_metrics(val_class_accuracies, prefix="    ")
#             print(f"Learning Rate: {current_lr:.6f}")
        
#         if should_stop:
#             print(f"Early stopping triggered at epoch {epoch+1}")
#             break
            
#     # Plot training history with per-class metrics
#     plot_training_history(metrics_history)
    
#     return early_stopping.best_state, metrics_history['val_accuracy'][early_stopping.best_epoch]

def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, 
                device, monitor, early_stopping, epochs, writer):
    """
    Train the model with comprehensive monitoring and memory management
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_data: Tuple of (X, X_spectral, y, night_indices) for validation
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to train on
        monitor: TrainingMonitor instance
        early_stopping: EarlyStoppingWithClassMetrics instance
        epochs: Number of epochs to train
        writer: TensorBoard writer
    """
    scaler = torch.cuda.amp.GradScaler()
    best_model = None
    best_metrics = None
    
    logging.info("\nStarting training:")
    logging.info(f"    Total epochs: {epochs}")
    logging.info(f"    Batch size: {next(iter(train_loader))[0].shape[0]}")
    logging.info(f"    Device: {device}")
    
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            monitor.reset_metrics()
            epoch_loss = 0.0
            
            # Create progress bar
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (batch_x, batch_x_spectral, batch_y, batch_nights) in enumerate(train_iter):
                try:
                    # Move data to device
                    batch_x = batch_x.to(device)
                    batch_x_spectral = batch_x_spectral.to(device)
                    batch_y = batch_y.to(device)
                    batch_nights = batch_nights.to(device)
                    
                    # Forward pass with mixed precision
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_spectral)
                        loss = criterion(outputs, batch_y, batch_nights)
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    monitor.update(outputs, batch_y, loss, model)
                    
                    # Update progress bar
                    train_iter.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Check memory usage
                    if device.type == 'cuda' and (batch_idx + 1) % 10 == 0:
                        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                        if memory_allocated > CONFIG['runtime']['memory_management']['max_batch_memory']:
                            logging.warning(f"High memory usage: {memory_allocated:.2f}GB allocated")
                            torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.error("GPU OOM in training batch. Attempting recovery...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    raise e
            
            # Get and log training metrics
            train_metrics = monitor.get_metrics()
            monitor.update_history(train_metrics, 'train')
            monitor.log_metrics(train_metrics, 'train', epoch)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics = defaultdict(float)
            
            X_val, X_val_spectral, y_val, night_indices_val = val_data
            
            try:
                with torch.no_grad():
                    for i in range(0, len(X_val), CONFIG['runtime']['evaluation']['batch_size']):
                        end_idx = min(i + CONFIG['runtime']['evaluation']['batch_size'], len(X_val))
                        val_batch_x = X_val[i:end_idx].to(device)
                        val_batch_spectral = X_val_spectral[i:end_idx].to(device)
                        val_batch_y = y_val[i:end_idx].to(device)
                        val_batch_nights = night_indices_val[i:end_idx].to(device)
                        
                        outputs = model(val_batch_x, val_batch_spectral)
                        loss = criterion(outputs, val_batch_y, val_batch_nights)
                        
                        val_loss += loss.item() * len(val_batch_y)
                        monitor.update(outputs, val_batch_y, loss, model)
                        
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("GPU OOM during validation. Reducing batch size...")
                    CONFIG['runtime']['evaluation']['batch_size'] //= 2
                    continue
                raise e
            
            # Get validation metrics
            val_metrics = monitor.get_metrics()
            monitor.update_history(val_metrics, 'val')
            monitor.log_metrics(val_metrics, 'val', epoch)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/validation', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/validation', val_metrics['accuracy'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            # Plot training progress
            monitor.plot_metrics(epoch)
            
            # Check early stopping
            if early_stopping(val_metrics, model.state_dict(), epoch):
                logging.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                best_model = early_stopping.best_weights
                best_metrics = val_metrics
                break
            
            if val_metrics['accuracy'] > (best_metrics['accuracy'] if best_metrics else 0):
                best_model = model.state_dict()
                best_metrics = val_metrics
        
        return best_model, best_metrics
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# def evaluate_model(model, data, criterion, device, batch_size=32):
#     """
#     Evaluate model with batched processing to prevent memory overflow
    
#     Args:
#         model: The model to evaluate
#         data: Tuple of (X, X_spectral, y)
#         criterion: Loss function
#         device: Device to run evaluation on
#         batch_size: Batch size for evaluation
#     """
#     model.eval()
#     X, X_spectral, y = data
#     total_loss = 0
#     all_predictions = []
#     total_samples = len(y)
    
#     # Create batches
#     num_batches = (total_samples + batch_size - 1) // batch_size
    
#     with torch.no_grad():
#         for i in range(num_batches):
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, total_samples)
            
#             # Get batch
#             batch_X = X[start_idx:end_idx].to(device)
#             batch_X_spectral = X_spectral[start_idx:end_idx].to(device)
#             batch_y = y[start_idx:end_idx].to(device)
            
#             try:
#                 # Forward pass
#                 outputs = model(batch_X, batch_X_spectral)
#                 loss = criterion(outputs, batch_y)
                
#                 # Accumulate loss
#                 total_loss += loss.item() * len(batch_y)
                
#                 # Get predictions
#                 _, predicted = torch.max(outputs, 1)
#                 all_predictions.extend(predicted.cpu().numpy())
                
#                 # Clear cache after each batch
#                 if device.type == 'cuda':
#                     torch.cuda.empty_cache()
                    
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     if device.type == 'cuda':
#                         torch.cuda.empty_cache()
#                     logging.error(f"GPU OOM in batch {i+1}/{num_batches}. Try reducing batch_size.")
#                     raise
#                 else:
#                     raise
    
#     # Calculate metrics
#     all_predictions = np.array(all_predictions)
#     accuracy = (all_predictions == y.cpu().numpy()).mean()
#     avg_loss = total_loss / total_samples
    
#     return avg_loss, accuracy, all_predictions

def evaluate_model(model, data, criterion, device, batch_size=32):
    """
    Evaluate model with comprehensive metrics and error handling
    
    Args:
        model: The model to evaluate
        data: Tuple of (X, X_spectral, y)
        criterion: Loss function
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    """
    model.eval()
    X, X_spectral, y = data
    total_loss = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    predictions = []
    total_samples = len(y)
    
    # Calculate number of batches
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    try:
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                
                # Get batch
                batch_X = X[start_idx:end_idx].to(device)
                batch_X_spectral = X_spectral[start_idx:end_idx].to(device)
                batch_y = y[start_idx:end_idx].to(device)
                
                # Forward pass
                outputs = model(batch_X, batch_X_spectral)
                loss = criterion(outputs, batch_y)
                
                # Accumulate loss and predictions
                total_loss += loss.item() * len(batch_y)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(batch_preds)
                
                # Update class-wise metrics
                for cls in range(len(SLEEP_STAGES)):
                    mask = batch_y == cls
                    class_total[cls] += mask.sum().item()
                    class_correct[cls] += ((batch_preds == cls) & 
                                         (batch_y.cpu().numpy() == cls)).sum()
                
                # Clear cache if using GPU
                if device.type == 'cuda' and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = y.cpu().numpy()
        
        # Calculate overall metrics
        accuracy = (predictions == true_labels).mean()
        avg_loss = total_loss / total_samples
        
        # Calculate class-wise metrics
        class_accuracies = {}
        for cls in range(len(SLEEP_STAGES)):
            if class_total[cls] > 0:
                class_accuracies[cls] = class_correct[cls] / class_total[cls]
            else:
                class_accuracies[cls] = 0.0
        
        # Log detailed metrics
        logging.info("\nEvaluation Results:")
        logging.info(f"    Overall Accuracy: {accuracy:.4f}")
        logging.info(f"    Average Loss: {avg_loss:.4f}")
        logging.info("\nClass-wise Accuracies:")
        for cls, acc in class_accuracies.items():
            logging.info(f"    {SLEEP_STAGES[cls]}: {acc:.4f} ({class_total[cls]} samples)")
        
        return avg_loss, accuracy, predictions
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            logging.warning("GPU OOM during evaluation. Trying with smaller batch size...")
            return evaluate_model(model, data, criterion, device, batch_size=batch_size//2)
        raise
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise
   

def validate_model(model, val_loader, criterion, device):
    """
    Validate model during training with class-wise metrics
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
    """
    model.eval()
    val_loss = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    try:
        with torch.no_grad():
            for batch_x, batch_x_spectral, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_x_spectral = batch_x_spectral.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x, batch_x_spectral)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct = predicted == batch_y
                
                # Update class-wise metrics
                for cls in range(len(SLEEP_STAGES)):
                    mask = batch_y == cls
                    class_correct[cls] += correct[mask].sum().item()
                    class_total[cls] += mask.sum().item()
        
        # Calculate metrics
        avg_loss = val_loss / len(val_loader)
        class_accuracies = {
            cls: class_correct[cls] / max(class_total[cls], 1)
            for cls in range(len(SLEEP_STAGES))
        }
        overall_accuracy = sum(class_correct.values()) / sum(class_total.values())
        
        metrics = {
            'loss': avg_loss,
            'accuracy': overall_accuracy,
            'class_accuracies': class_accuracies
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        raise

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


# Add these utility functions to functions.py

def verify_data_balance(labels, threshold=0.1):
    """
    Verify if the class distribution is reasonably balanced
    
    Args:
        labels: Tensor of class labels
        threshold: Maximum allowed deviation from perfect balance
    
    Returns:
        bool: Whether the distribution is balanced
        dict: Class distribution statistics
    """
    class_counts = Counter(labels.numpy())
    total_samples = len(labels)
    perfect_share = 1.0 / len(class_counts)
    
    stats = {}
    is_balanced = True
    
    for cls in sorted(class_counts.keys()):
        share = class_counts[cls] / total_samples
        deviation = abs(share - perfect_share)
        stats[cls] = {
            'count': class_counts[cls],
            'share': share,
            'deviation': deviation
        }
        
        if deviation > threshold:
            is_balanced = False
    
    return is_balanced, stats

def format_time_elapsed(seconds):
    """Format elapsed time in a human-readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def log_gpu_memory_stats():
    """Log detailed GPU memory statistics"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            logging.info(f"\nGPU {i} ({props.name}):")
            logging.info(f"    Total memory: {props.total_memory/1e9:.2f}GB")
            logging.info(f"    Allocated memory: {allocated:.2f}GB")
            logging.info(f"    Reserved memory: {reserved:.2f}GB")

def compute_class_statistics(y_true, y_pred):
    """
    Compute detailed per-class statistics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: Dictionary containing per-class metrics
    """
    stats = {}
    unique_classes = sorted(set(y_true))
    
    for cls in unique_classes:
        mask = y_true == cls
        cls_total = mask.sum()
        cls_correct = (y_pred[mask] == cls).sum()
        
        stats[cls] = {
            'total': int(cls_total),
            'correct': int(cls_correct),
            'accuracy': float(cls_correct) / float(cls_total) if cls_total > 0 else 0,
            'share': float(cls_total) / len(y_true)
        }
    
    return stats

def validate_loaded_data(X, y, night_indices):
    """
    Validate loaded data format and dimensions
    
    Args:
        X: Input signal data
        y: Labels
        night_indices: Night indices for each sample
    """
    # Check basic dimensions
    assert X.ndim == 3, f"Expected X to be 3D (n_samples, n_channels, timepoints), got shape {X.shape}"
    assert X.shape[1] == 4, f"Expected 4 channels, got {X.shape[1]}"
    assert X.shape[2] == 3000, f"Expected 3000 timepoints, got {X.shape[2]}"
    
    # Check label dimensions
    assert y.ndim == 1, f"Expected y to be 1D, got shape {y.shape}"
    assert len(X) == len(y), f"Mismatch between X length ({len(X)}) and y length ({len(y)})"
    
    # Check night indices
    assert len(night_indices) == len(y), "Night indices length mismatch"
    
    # Check data types
    assert torch.is_tensor(X), "X should be a PyTorch tensor"
    assert torch.is_tensor(y), "y should be a PyTorch tensor"
    
    # Check value ranges
    assert torch.isfinite(X).all(), "X contains non-finite values"
    assert y.min() >= 0 and y.max() <= 4, f"Invalid label values: min={y.min()}, max={y.max()}"
    
    # Log validation results
    logging.info("\nData validation passed:")
    logging.info(f"    X shape: {X.shape}")
    logging.info(f"    y shape: {y.shape}")
    logging.info(f"    Number of nights: {len(set(night_indices.numpy()))}")
    logging.info(f"    Class distribution: {Counter(y.numpy())}")

    def verify_tensor_types(X, X_spectral, y):
        """Verify that all inputs are proper PyTorch tensors with correct dtypes"""
        assert torch.is_tensor(X) and X.dtype == torch.float32, "X must be a float32 tensor"
        assert torch.is_tensor(X_spectral) and X_spectral.dtype == torch.float32, "X_spectral must be a float32 tensor"
        assert torch.is_tensor(y) and y.dtype == torch.long, "y must be a long tensor"


def create_balanced_data_loaders(X, X_spectral, y, night_indices, batch_size, augment=True):
    """Create data loaders with sophisticated balancing strategy"""
    from tools.classes import BalancedSleepDataset  # Import here to avoid circular imports
    
    # Create dataset
    dataset = BalancedSleepDataset(
        X=X,
        X_spectral=X_spectral,
        y=y,
        night_indices=night_indices,
        augment=augment
    )
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset.weights),
        replacement=True
    )
    
    # Create and return data loader
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )