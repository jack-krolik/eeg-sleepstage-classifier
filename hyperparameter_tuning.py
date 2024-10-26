
from tools.functions import *
from tools.classes import *
from tools.utils import *
from tools.config import CONFIG, device
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

set_seed()


# Ensure the model save directory exists
ensure_dir(CONFIG['new_model_path'])

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
minority_classes = [cls for cls, count in class_counts.items() if count < len(y_train) / len(class_counts) * 0.5]

# Apply augmentation
X_train, X_train_spectral, y_train = augment_minority_classes(X_train, X_train_spectral, y_train, minority_classes)



run_tuning = False
get_best_lr = False

if run_tuning:
    logging.info("Starting hyperparameter tuning...")
    # After tuning
    best_params = run_hyperparameter_tuning(X_train, X_train_spectral, y_train, device)
    params = {
        'model_params': {k: v for k, v in best_params.items() if k in ['n_filters', 'lstm_hidden', 'lstm_layers', 'dropout']},
        'train_params': {k: v for k, v in best_params.items() if k in ['lr', 'batch_size']}
    }
else:
    params = CONFIG['initial_params']

# Initialize model and get parameters
ensemble_model, params = initialize_model(device)

if CONFIG['use_pretrained_weights']:
    pretrained_path = os.path.join(CONFIG['old_model_path'], CONFIG['model_names']['ensemble'])
    ensemble_model.load_state_dict(torch.load(pretrained_path))
    logging.info(f"Loaded pretrained weights from {pretrained_path}")

# Save initial parameters
save_params(params, os.path.join(CONFIG['new_model_path'], 'initial_params2.json'))

# Set up training parameters
train_params = params['train_params']
# balanced_sampler = BalancedBatchSampler(y_train.numpy(), batch_size=train_params['batch_size'])
# train_loader = DataLoader(TensorDataset(X_train, X_train_spectral, y_train), batch_sampler=balanced_sampler)
train_loader = create_data_loaders(X_train, X_train_spectral, y_train, batch_size=train_params['batch_size'], is_train=True)
val_loader = create_data_loaders(X_val, X_val_spectral, y_val, batch_size=train_params['batch_size'], is_train=False)

# Set up loss function
class_weights = get_class_weights(y_train).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights + 1e-6, label_smoothing=0.1)

 # Find best learning rate if requested
if get_best_lr:
    temp_optimizer = optim.AdamW(ensemble_model.parameters(), lr=1e-7, weight_decay=1e-5)
    best_lr = find_lr(ensemble_model, train_loader, val_loader, temp_optimizer, criterion, device)
    train_params['lr'] = best_lr
    logging.info(f"Best learning rate found: {best_lr}")



# Set up optimizer and scheduler with the selected learning rate
optimizer = optim.AdamW(ensemble_model.parameters(), lr=train_params['lr'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train model
best_model_state, best_accuracy = train_model(
    ensemble_model, train_loader, (X_val, X_val_spectral, y_val),
    optimizer, scheduler, criterion, device, epochs=train_params['num_epochs']
)

# Save best model
if best_model_state is not None:
    
    save_model(ensemble_model, os.path.join(CONFIG['new_model_path'], CONFIG['model_names']['ensemble2']))

    logging.info(f"Best ensemble model saved. Final validation accuracy: {best_accuracy:.4f}")

    # Evaluate on test set
    ensemble_model.load_state_dict(best_model_state)
    test_loss, test_accuracy, test_predictions = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), criterion, device)
    logging.info(f"Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test.cpu().numpy(), test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['new_model_path'], 'confusion_matrix2.png'))



    # Generate classification report
    report = classification_report(y_test.cpu().numpy(), test_predictions)
    logging.info(f"Classification Report:\n{report}")

