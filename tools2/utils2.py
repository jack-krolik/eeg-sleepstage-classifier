import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter

SLEEP_STAGES = {
    0: 'N3 (Deep)',
    1: 'N2 (Light)',
    2: 'N1 (Light)',
    3: 'REM',
    4: 'Wake'
}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_class_distribution(counter, class_names=SLEEP_STAGES):
    total = sum(counter.values())
    distribution = []
    for class_idx in sorted(class_names.keys()):
        count = counter.get(class_idx, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        distribution.append(
            f"{class_names[class_idx]}: {count} ({percentage:.1f}%)"
        )
    return "\n    ".join(distribution)

def format_time_delta(seconds):
    """Format time delta into a readable string"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def format_metrics_table(metrics, prefix=''):
    """Format metrics into a structured table string"""
    rows = []
    
    # Add basic metrics
    basic_metrics = {
        'Loss': metrics['loss'],
        'Accuracy': metrics['accuracy'],
        'F1 Macro': metrics['f1_macro'],
        'F1 Weighted': metrics['f1_weighted']
    }
    
    for name, value in basic_metrics.items():
        rows.append(f"{prefix}{name}: {value:.4f}")
    
    # Add per-class F1 scores
    for i, f1 in enumerate(metrics['per_class_f1']):
        rows.append(f"{prefix}F1 {SLEEP_STAGES[i]}: {f1:.4f}")
    
    return '\n'.join(rows)

def log_epoch_metrics(epoch, total_epochs, train_metrics, val_metrics, current_lr):
    """Log comprehensive metrics for current epoch"""
    # Create separator line
    separator = "-" * 80
    
    # Format epoch header
    epoch_header = f"Epoch {epoch + 1}/{total_epochs}"
    
    # Format learning rate
    lr_info = f"Learning Rate: {current_lr:.2e}"
    
    # Create metrics tables
    train_table = format_metrics_table(train_metrics, prefix='Train ')
    val_table = format_metrics_table(val_metrics, prefix='Val ')
    
    # Combine all information
    log_message = f"""
{separator}
{epoch_header}
{separator}
{lr_info}

Training Metrics:
{train_table}

Validation Metrics:
{val_table}
{separator}
"""
    
    logging.info(log_message)
    
    # Optional: Create a more compact one-line summary for tqdm
    summary = (f"Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1_macro']:.4f}")
    
    return summary


def log_fold_results(fold, metrics):
    """Log results for a single fold"""
    separator = "-" * 80
    logging.info(f"\n{separator}")
    logging.info(f"Fold {fold} Results:")
    logging.info(f"{separator}")
    
    # Log metrics
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logging.info(f"{metric_name}: {value:.4f}")
        
    # Log class-wise metrics if available
    if 'per_class_f1' in metrics:
        logging.info("\nPer-class F1 Scores:")
        for i, f1 in enumerate(metrics['per_class_f1']):
            logging.info(f"{SLEEP_STAGES[i]}: {f1:.4f}")
    
    logging.info(f"{separator}\n")

def calculate_metrics(predictions, targets):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(targets, predictions),
        'f1_macro': f1_score(targets, predictions, average='macro'),
        'f1_weighted': f1_score(targets, predictions, average='weighted'),
        'per_class_f1': f1_score(targets, predictions, average=None),
        'confusion_matrix': confusion_matrix(targets, predictions)
    }

# In utils2.py - Add these helper functions
def log_class_statistics(y, phase="", detailed=False):
    """Concise logging of class distribution"""
    class_dist = Counter(y.numpy()) if torch.is_tensor(y) else Counter(y)
    total = sum(class_dist.values())
    
    logging.info(f"\n{phase} Class Distribution:")
    for class_idx in sorted(SLEEP_STAGES.keys()):
        count = class_dist.get(class_idx, 0)
        pct = (count / total) * 100 if total > 0 else 0
        logging.info(f"    {SLEEP_STAGES[class_idx]}: {count:5d} ({pct:5.1f}%)")
    
    if detailed:
        return {
            'distribution': class_dist,
            'total': total,
            'ratios': {cls: count/total for cls, count in class_dist.items()}
        }

def monitor_batch_balance(loader, num_batches=5):
    """Monitor class balance in a few batches without excessive logging"""
    batch_distributions = []
    
    for i, (_, _, y) in enumerate(loader):
        if i >= num_batches:
            break
        batch_distributions.append(Counter(y.numpy()))
    
    # Calculate average distribution
    avg_dist = Counter()
    for dist in batch_distributions:
        avg_dist.update(dist)
    for k in avg_dist:
        avg_dist[k] /= len(batch_distributions)
    
    logging.info(f"\nAverage batch distribution ({num_batches} batches):")
    for class_idx in sorted(SLEEP_STAGES.keys()):
        avg = avg_dist.get(class_idx, 0)
        logging.info(f"    {SLEEP_STAGES[class_idx]}: {avg:.1f}")
    
    return avg_dist