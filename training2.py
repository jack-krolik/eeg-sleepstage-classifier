# Replace the top of training.py with:
import os
import sys
import logging
from tools.config2 import CONFIG, device, cuda_manager  # Import the device directly from config
import torch
from tools.functions2 import set_seed

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["NUMEXPR_MAX_THREADS"] = "16" 

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add logging to track device and memory from the imported device
logger.info(f"Using device: {device}")

# Set seed using the imported CONFIG
set_seed(CONFIG['settings']['seed'])

# Import the rest of your modules
from tools.functions2 import *
from tools.classes2 import *
from tools.utils2 import *
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Ensure the model directory exists
ensure_dir(CONFIG['model_dir'])

# Load Data
try:
    x, y = load_data(CONFIG['data_path'])
    logging.info(f"Loaded data shape: {x.shape}, Labels shape: {y.shape}")
except Exception as e:
    logging.error(f"Error loading data: {str(e)}")
    raise

# Prepare the data (includes SMOTE)
X_train, X_train_spectral, y_train, X_val, X_val_spectral, y_val, X_test, X_test_spectral, y_test = prepare_data(x, y)

# Apply preprocessing
X_train, X_train_spectral = preprocess_data(X_train, X_train_spectral)
X_val, X_val_spectral = preprocess_data(X_val, X_val_spectral)
X_test, X_test_spectral = preprocess_data(X_test, X_test_spectral)

# Identify minority classes for augmentation
class_counts = Counter(y_train.numpy())
minority_classes = [cls for cls, count in class_counts.items() 
                   if count < len(y_train) / len(class_counts) * 0.5]

# Apply augmentation
X_train, X_train_spectral, y_train = augment_minority_classes(
    X_train, X_train_spectral, y_train, minority_classes)

run_tuning = True
start_with_config = True  # Set this to True to start with CONFIG parameters
fine_tune_lr = True  # Set this to True if you want to fine-tune the learning rate

if run_tuning:
    logging.info("Starting hyperparameter tuning...")
    best_params = run_hyperparameter_tuning(
        X_train, X_train_spectral, y_train, 
        device,
        start_with_config=start_with_config
    )
    
    # Initialize model with best parameters
    model_params = {k: v for k, v in best_params.items() 
                   if k in ['n_filters', 'lstm_hidden', 'lstm_layers', 'dropout']}
    # ensemble_model = EnsembleModel(model_params).to(device)
    ensemble_model, device = initialize_model_with_gpu_check(model_params, device)
    

    
    
    # Create data loaders
    train_loader = create_data_loaders(X_train, X_train_spectral, y_train, 
                                     batch_size=best_params['batch_size'], is_train=True)
    val_loader = create_data_loaders(X_val, X_val_spectral, y_val, 
                                   batch_size=best_params['batch_size'], is_train=False)
    
    # Set up loss function
    class_weights = get_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights + 1e-6, label_smoothing=0.1)
    
    # Optionally, find best learning rate
    if fine_tune_lr:
        temp_optimizer = optim.AdamW(ensemble_model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
        best_lr = find_lr(ensemble_model, train_loader, val_loader, temp_optimizer, criterion, device, 
                         start_lr=best_params['lr'])
        logging.info(f"Fine-tuned learning rate: {best_lr}")
    else:
        best_lr = best_params['lr']
    
    params = {
        'model_params': model_params,
        'train_params': {
            'lr': best_lr,
            'batch_size': best_params['batch_size'],
            'num_epochs': CONFIG['train_params']['initial']['num_epochs'],
            'patience': CONFIG['train_params']['initial']['patience']
        }
    }
else:
    params = {
        'model_params': CONFIG['model_params']['initial'],
        'train_params': CONFIG['train_params']['initial']
    }
    # ensemble_model, _ = initialize_model(device)
    ensemble_model, device = initialize_model_with_gpu_check(CONFIG['model_params']['initial'], device)

# Load pretrained weights if specified
if CONFIG['settings']['use_pretrained_weights']:
    try:
        load_pretrained_weights(ensemble_model, 'ensemble')
    except Exception as e:
        logging.warning(f"Failed to load pretrained weights: {str(e)}")

# Save parameters
save_params(params, CONFIG['model_names']['params'])

# Set up training parameters
train_params = params['train_params']
train_loader = create_data_loaders(X_train, X_train_spectral, y_train, 
                                 batch_size=train_params['batch_size'], is_train=True)
val_loader = create_data_loaders(X_val, X_val_spectral, y_val, 
                               batch_size=train_params['batch_size'], is_train=False)

# Set up optimizer and scheduler
optimizer = optim.AdamW(ensemble_model.parameters(), lr=train_params['lr'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True)



# Train model
best_model_state, best_accuracy = train_model(
    ensemble_model, train_loader, (X_val, X_val_spectral, y_val),
    optimizer, scheduler, criterion, device, 
    epochs=train_params['num_epochs'],
)

# Save best model
if best_model_state is not None:
    ensemble_model.load_state_dict(best_model_state)
    save_model(ensemble_model, CONFIG['model_names']['ensemble'])
    logging.info(f"Best ensemble model saved. Final validation accuracy: {best_accuracy:.4f}")

    # Evaluate on test set
    test_loss, test_accuracy, test_predictions = evaluate_model(
        ensemble_model, (X_test, X_test_spectral, y_test), criterion, device)
    logging.info(f"Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test.cpu().numpy(), test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['model_dir'], CONFIG['model_names']['confusion_matrix']))
    plt.close()

    # Generate classification report
    report = classification_report(y_test.cpu().numpy(), test_predictions)
    logging.info(f"Classification Report:\n{report}")

# Train diverse ensemble
diverse_ensemble = DiverseEnsembleModel(params['model_params']).to(device)
diverse_optimizer = optim.AdamW(diverse_ensemble.parameters(), lr=1e-3, weight_decay=1e-4)
diverse_scheduler = CosineAnnealingWarmRestarts(
    diverse_optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Set up TensorBoard
writer = TorchSummaryWriter(log_dir=os.path.join(CONFIG['model_dir'], CONFIG['model_names']['tensorboard']))

logging.info("Training diverse ensemble model...")
best_diverse_accuracy = 0
diverse_best_state = None

for epoch in range(train_params['num_epochs']):
    diverse_ensemble.train()
    epoch_loss = 0
    for batch_idx, (data, spectral_features, target) in enumerate(train_loader):
        data, spectral_features, target = data.to(device), spectral_features.to(device), target.to(device)
        diverse_optimizer.zero_grad()
        output = diverse_ensemble(data, spectral_features)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(diverse_ensemble.parameters(), max_norm=1.0)
        
        diverse_optimizer.step()
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Calculate average training loss for the epoch
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    diverse_ensemble.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, spectral_features, target in val_loader:
            data, spectral_features, target = data.to(device), spectral_features.to(device), target.to(device)
            output = diverse_ensemble(data, spectral_features)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)

    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    writer.add_scalar('Learning Rate', diverse_optimizer.param_groups[0]['lr'], epoch)

    diverse_scheduler.step()

    # Check for improvement and save the best model
    if accuracy > best_diverse_accuracy:
        best_diverse_accuracy = accuracy
        diverse_best_state = diverse_ensemble.state_dict()
        save_model(diverse_ensemble, CONFIG['model_names']['diverse'])
        logging.info(f"New best diverse ensemble model saved. Accuracy: {accuracy:.4f}")

# Close TensorBoard writer
writer.close()

# Evaluate the best diverse ensemble model
if diverse_best_state is not None:
    diverse_ensemble.load_state_dict(diverse_best_state)
    test_loss, test_accuracy, test_predictions = evaluate_model(
        diverse_ensemble, (X_test, X_test_spectral, y_test), criterion, device)
    logging.info(f"Diverse Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")

    # Generate and save confusion matrix for diverse ensemble
    cm = confusion_matrix(y_test.cpu().numpy(), test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Diverse Ensemble')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['model_dir'], CONFIG['model_names']['diverse_confusion_matrix']))
    plt.close()

# Knowledge distillation
single_model = ImprovedSleepdetector(**params['model_params']).to(device)
logging.info("Performing knowledge distillation...")
distilled_model = distill_knowledge(ensemble_model, single_model, train_loader, 
                                  (X_val, X_val_spectral, y_val), device)
save_model(distilled_model, CONFIG['model_names']['distilled'])

# Final evaluation of all models
_, ensemble_accuracy, _ = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), criterion, device)
_, diverse_accuracy, _ = evaluate_model(diverse_ensemble, (X_test, X_test_spectral, y_test), criterion, device)
_, distilled_accuracy, _ = evaluate_model(distilled_model, (X_test, X_test_spectral, y_test), criterion, device)

logging.info(f"""Training completed.
Ensemble Model - Final Test Accuracy: {ensemble_accuracy:.4f}
Diverse Ensemble Model - Final Test Accuracy: {diverse_accuracy:.4f}
Distilled Model - Final Test Accuracy: {distilled_accuracy:.4f}""")