import argparse

# Add this at the start of training_clean.py, before any other imports
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use')
parser.add_argument('--memory-fraction', type=float, default=0.8, help='Fraction of GPU memory to use')
parser.add_argument('--no-fallback', action='store_true', help='Disable fallback to other GPUs')
args = parser.parse_args()

import os
import logging
import torch  # Import torch before other custom modules
from tools_clean.config_clean import CONFIG, DATA_FILES, get_cuda_device
from tools_clean.classes_clean import SleepDataManager, NightBasedCrossValidator, EnsembleModel
from tools_clean.functions_clean import *
from tools_clean.utils_clean import (
    format_metrics_table, log_epoch_metrics, calculate_metrics, log_fold_results
)
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm

# Initialize device first
device = get_cuda_device(
    gpu_id=args.gpu,
    memory_fraction=args.memory_fraction,
    allow_fallback=not args.no_fallback
)

# Update CONFIG with selected GPU
CONFIG['settings']['gpu_settings']['device_id'] = args.gpu
CONFIG['settings']['gpu_settings']['memory_fraction'] = args.memory_fraction

# Initialize logging
setup_logging()

# Log GPU configuration
logging.info(f"""
GPU Configuration:
    Device: {device}
    Physical GPU ID: {args.gpu if args.gpu is not None else 'Auto-selected'}
    Memory Fraction: {args.memory_fraction}
    Fallback Enabled: {not args.no_fallback}
""")

# set seed
set_seed(CONFIG['settings']['seed'])
logging.info(f"Seed: {CONFIG['settings']['seed']}")

# Print GPU info using cuda_manager for consistency
cuda_manager = CUDAManager.get_instance()
if device.type == 'cuda':
    gpu_info = cuda_manager.get_memory_info()
    if gpu_info:
        logging.info(f"""
GPU Status:
    Selected GPU: {torch.cuda.get_device_name(device)}
    Total Memory: {gpu_info['total']:.2f}GB
    Used Memory: {gpu_info['allocated']:.2f}GB
    Cached Memory: {gpu_info['cached']:.2f}GB
    Free Memory: {gpu_info['free']:.2f}GB
""")
    else:
        logging.warning("Could not get detailed GPU memory information")
else:
    logging.info("Running on CPU")

use_data_files = True

if use_data_files:
    data_files = DATA_FILES[:50]
else:
        
    data_files = [
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_201_N1.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_201_N2.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_202_N1.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_202_N2.mat'),
    ]



# Initialize data manager
logging.info("Initializing data manager...")
data_manager = SleepDataManager(
    data_files=data_files,
    val_ratio=0.2,
    seed=CONFIG['settings']['seed']
)



# Load and preprocess data
data_manager.load_and_preprocess()

# Monitor memory after data loading
if device.type == 'cuda':
    gpu_info = cuda_manager.get_memory_info()
    logging.info(f"Memory usage after data loading: {gpu_info['allocated']:.2f}GB")

# Check data after loading
logging.info(f"Data loaded - Total samples: {len(data_manager.data['y'])}")
# logging.info(f"Class distribution: {Counter(data_manager.data['y'].numpy())}")
logging.info(f"Class distribution: {format_class_distribution(Counter(data_manager.data['y'].numpy()))}")
logging.info(f"Number of nights: {len(torch.unique(data_manager.data['night_idx']))}")



# Get training parameters based on mode
model_params, train_params = get_training_parameters(data_manager)

# Create cross-validator
logging.info("Creating cross-validator...")
cv = data_manager.create_cross_validator(n_folds=5)


# Add memory cleanup before training
torch.cuda.empty_cache()

# Train with cross-validation
results = train_with_cv(data_manager, cv, model_params, train_params)



# Save and log results
save_dir, best_model = save_best_model_results(
    results=results,
    data_manager=data_manager,
    model_params=model_params,
    train_params=train_params
)
