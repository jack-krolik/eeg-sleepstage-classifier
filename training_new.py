# %%
import os
import logging
from tools2.config2 import CONFIG, DATA_FILES, device, cuda_manager
from tools2.classes2 import SleepDataManager, NightBasedCrossValidator, EnsembleModel
from tools2.functions2 import *
from tools2.utils2 import (
    format_metrics_table, log_epoch_metrics, log_confusion_matrix,
    log_class_distribution, log_training_start, early_stopping_check,
    log_fold_results, calculate_metrics
)
from datetime import datetime
from collections import defaultdict


def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(CONFIG['model_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )

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
            
            criterion = nn.CrossEntropyLoss(
                weight=data_manager.class_weights.to(device),
                label_smoothing=0.1
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
    
    # Get splits while maintaining class distribution
    logging.info("Creating cross-validation splits...")
    splits = cv.split(data_manager.data['y'])
    
    # Check if splits are being created
    splits = list(splits)  # Convert generator to list
    logging.info(f"Number of splits created: {len(splits)}")
    
    if not splits:
        logging.error("No valid splits were created!")
        return results

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        logging.info(f"\nTraining Fold {fold}")
        logging.info(f"Train set size: {len(train_idx)}, Validation set size: {len(val_idx)}")
        
        # Create data loaders
        train_loader = data_manager.get_loader(
            train_idx,
            batch_size=train_params['batch_size'],
            is_training=True
        )
        
        val_loader = data_manager.get_loader(
            val_idx,
            batch_size=train_params['batch_size'],
            is_training=False
        )
        
        # Initialize model
        global device
        model, device = initialize_model_with_gpu_check(model_params, device)
        logging.info(f"Model initialized on {device}")
        
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
        
        criterion = nn.CrossEntropyLoss(
            weight=data_manager.class_weights.to(device),
            label_smoothing=0.1
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
            'metrics': metrics
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
    
    for epoch in range(train_params['num_epochs']):
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
        
        # Log progress
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
        
        # Update best metrics
        if best_metrics is None or val_metrics['f1_macro'] > best_metrics['f1_macro']:
            best_metrics = val_metrics
    
    # Log fold results
    log_fold_results(fold, best_metrics)
    
    return early_stopping.best_state, best_metrics

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    metrics = defaultdict(float)
    all_predictions = []
    all_targets = []
    
    for batch_x, batch_x_spectral, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_x_spectral = batch_x_spectral.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x, batch_x_spectral)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        metrics['loss'] += loss.item()
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate epoch metrics
    metrics['loss'] /= len(train_loader)
    metrics.update(calculate_metrics(all_predictions, all_targets))
    
    return metrics

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

def save_checkpoint(model, optimizer, metrics, fold):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    path = os.path.join(CONFIG['model_dir'], f'model_fold_{fold}.pt')
    torch.save(checkpoint, path)

def save_results(results):
    """Save training results"""
    path = os.path.join(CONFIG['model_dir'], 'training_results.pt')
    torch.save(results, path)



# %%
# Initialize logging
setup_logging()

use_data_files = False

if use_data_files:
    data_files = DATA_FILES
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



# %%
# Load and preprocess data
data_manager.load_and_preprocess()

# Check data after loading
logging.info(f"Data loaded - Total samples: {len(data_manager.data['y'])}")
logging.info(f"Class distribution: {Counter(data_manager.data['y'].numpy())}")
logging.info(f"Number of nights: {len(torch.unique(data_manager.data['night_idx']))}")



# %%
# Get training parameters based on mode
model_params, train_params = get_training_parameters(data_manager)

# Create cross-validator
logging.info("Creating cross-validator...")
cv = data_manager.create_cross_validator(n_splits=5)


# %%
# Train with cross-validation
results = train_with_cv(data_manager, cv, model_params, train_params)

# %%
# Save and log results
save_results(results)

# %%



