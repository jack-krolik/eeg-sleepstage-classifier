import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

def log_confusion_matrix(confusion_mat, epoch, save_dir):
    """Log and visualize confusion matrix"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_mat, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=[SLEEP_STAGES[i] for i in range(len(SLEEP_STAGES))],
                yticklabels=[SLEEP_STAGES[i] for i in range(len(SLEEP_STAGES))])
    plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the confusion matrix plot
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch + 1}.png'),
                bbox_inches='tight')
    plt.close()
    
    # Log confusion matrix details
    logging.info("\nConfusion Matrix:")
    class_names = [SLEEP_STAGES[i] for i in range(len(SLEEP_STAGES))]
    
    # Calculate and log precision, recall for each class
    precisions = confusion_mat.diagonal() / confusion_mat.sum(axis=0)
    recalls = confusion_mat.diagonal() / confusion_mat.sum(axis=1)
    
    for i, (class_name, precision, recall) in enumerate(zip(class_names, precisions, recalls)):
        logging.info(f"{class_name}:")
        logging.info(f"  Precision: {precision:.4f}")
        logging.info(f"  Recall: {recall:.4f}")
        logging.info(f"  Support: {confusion_mat[i].sum()}")

def log_class_distribution(y, title="Class Distribution"):
    """Log distribution of classes in a dataset"""
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    logging.info(f"\n{title}:")
    for class_idx, count in zip(unique, counts):
        percentage = (count / total_samples) * 100
        logging.info(f"{SLEEP_STAGES[class_idx]}: {count} samples ({percentage:.2f}%)")

def log_training_start(model, optimizer, scheduler, criterion):
    """Log training configuration at start of training"""
    logging.info("\nTraining Configuration:")
    logging.info("-" * 80)
    
    # Log model architecture
    logging.info("\nModel Architecture:")
    logging.info(str(model))
    
    # Log optimizer settings
    logging.info("\nOptimizer Settings:")
    logging.info(f"Type: {type(optimizer).__name__}")
    for param_group in optimizer.param_groups:
        logging.info(f"Learning Rate: {param_group['lr']}")
        logging.info(f"Weight Decay: {param_group.get('weight_decay', 'Not set')}")
    
    # Log scheduler settings
    logging.info("\nScheduler Settings:")
    logging.info(f"Type: {type(scheduler).__name__}")
    
    # Log criterion settings
    logging.info("\nLoss Function:")
    logging.info(f"Type: {type(criterion).__name__}")
    
    # Log device information
    if torch.cuda.is_available():
        logging.info("\nGPU Information:")
        logging.info(f"Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        logging.info(f"Memory Cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    else:
        logging.info("\nRunning on CPU")
    
    logging.info("-" * 80 + "\n")


def early_stopping_check(val_metrics, best_metrics, patience):
    """Check if training should be stopped early"""
    if val_metrics['f1_macro'] > best_metrics['val_f1']:
        return False
    return True

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

