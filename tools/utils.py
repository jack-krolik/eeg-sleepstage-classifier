import os
import json
import torch
import logging
from tools.config import CONFIG
import gc

def save_params(params, filename):
    if not os.path.exists(CONFIG['model_dir']):
        os.makedirs(CONFIG['model_dir'])
    filepath = os.path.join(CONFIG['model_dir'], filename)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def load_pretrained_weights(model, model_name):
    if CONFIG['settings']['use_pretrained_weights']:
        pretrained_path = os.path.join(CONFIG['old_model_path'], CONFIG['model_names'][model_name])
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            logging.info(f"Loaded pretrained weights from {pretrained_path}")
        else:
            logging.warning(f"Pretrained weights file not found: {pretrained_path}")
    return model

def save_model(model, filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    torch.save(model.state_dict(), filepath)

def load_model(model, filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    if os.path.exists(filepath):
        if CONFIG['settings']['force_cpu']:
            model.load_state_dict(torch.load(filepath, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(filepath))
    return model

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Add this to utils.py
SLEEP_STAGES = {
    0: 'N3 (Deep)',
    1: 'N2 (Light)',
    2: 'N1 (Light)',
    3: 'REM',
    4: 'Wake'
}

# def format_class_distribution(counter, class_names=SLEEP_STAGES):
#     """
#     Format class distribution with class names in ordered dictionary
    
#     Args:
#         counter: Counter object with class distributions
#         class_names: Dictionary mapping class indices to names
    
#     Returns:
#         str: Formatted string with class distributions
#     """
#     total = sum(counter.values())
#     formatted_dist = []
    
#     # Sort by class index to ensure consistent order
#     for class_idx in sorted(class_names.keys()):
#         count = counter.get(class_idx, 0)
#         percentage = (count / total) * 100 if total > 0 else 0
#         formatted_dist.append(
#             f"{class_names[class_idx]}: {count} ({percentage:.1f}%)"
#         )
    
#     return "\n    ".join(formatted_dist)

def format_class_distribution(counter, class_names=SLEEP_STAGES):
    """
    Format class distribution with class names in ordered dictionary
    
    Args:
        counter: Counter object or dictionary with class counts
        class_names: Dictionary mapping class indices to names
    
    Returns:
        str: Formatted string with class distributions
    """
    total = sum(counter.values())
    distribution = []
    
    # Always iterate through classes in order (0 through 4)
    for class_idx in sorted(class_names.keys()):
        count = counter.get(class_idx, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        distribution.append(
            f"{class_names[class_idx]}: {count} ({percentage:.1f}%)"
        )
    
    return "\n    ".join(distribution)



def log_class_metrics(metrics_dict, class_names=SLEEP_STAGES, prefix=""):
    """
    Log class-specific metrics with class names
    
    Args:
        metrics_dict: Dictionary of metrics per class
        class_names: Dictionary mapping class indices to names
        prefix: Optional prefix for logging
    """
    for class_idx in sorted(class_names.keys()):
        class_name = class_names[class_idx]
        metric_key = f"class_{class_idx}"
        if metric_key in metrics_dict:
            logging.info(f"{prefix}{class_name}: {metrics_dict[metric_key]:.4f}")