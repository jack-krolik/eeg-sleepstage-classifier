from functions import *
from classes import *
from utils import *
from config import CONFIG, get_device
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = get_device(force_cpu=True)

set_seed()

# Ensure the model save directory exists
ensure_dir(CONFIG['model_path'])

# Load Data
try:
    x, y = load_data()
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

# Initialize model and get parameters
ensemble_model, params = initialize_model(device)

# Save initial parameters
save_params(params, 'initial_params.json')

# Set up training parameters
train_params = params['train_params']
balanced_sampler = BalancedBatchSampler(y_train.numpy(), batch_size=train_params['batch_size'])
train_loader = DataLoader(TensorDataset(X_train, X_train_spectral, y_train), batch_sampler=balanced_sampler)

# Set up loss function
class_weights = get_class_weights(y_train).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights + 1e-6, label_smoothing=0.1)

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
    save_model(ensemble_model, CONFIG['best_model_name'])
    logging.info(f"Best ensemble model saved. Final validation accuracy: {best_accuracy:.4f}")

    # Evaluate on test set
    ensemble_model.load_state_dict(best_model_state)
    test_accuracy, test_predictions = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), device)
    logging.info(f"Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test.cpu().numpy(), test_predictions.cpu().numpy())
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(CONFIG['model_path'], 'confusion_matrix.png'))

    # Generate classification report
    report = classification_report(y_test.cpu().numpy(), test_predictions.cpu().numpy())
    logging.info(f"Classification Report:\n{report}")

# Train diverse ensemble
diverse_ensemble = DiverseEnsembleModel(CONFIG['initial_params']['model_params']).to(device)
diverse_optimizer = optim.AdamW(diverse_ensemble.parameters(), lr=best_lr, weight_decay=1e-2)
diverse_scheduler = get_scheduler(diverse_optimizer, num_warmup_steps=len(train_loader)*5, num_training_steps=len(train_loader)*train_params['num_epochs'])

logging.info("Training diverse ensemble model...")
diverse_best_state, diverse_accuracy = train_model(
    diverse_ensemble, train_loader, (X_val, X_val_spectral, y_val),
    diverse_optimizer, diverse_scheduler, criterion, device, epochs=train_params['num_epochs']
)

save_model(diverse_ensemble, CONFIG['diverse_model_name'])
logging.info(f"Best diverse ensemble model saved. Final accuracy: {diverse_accuracy:.4f}")

# Distill knowledge
single_model = ImprovedSleepdetector(**CONFIG['initial_params']['model_params']).to(device)

logging.info("Performing knowledge distillation...")
distilled_model = distill_knowledge(ensemble_model, single_model, train_loader, (X_val, X_val_spectral, y_val), device)

save_model(distilled_model, CONFIG['distilled_model_name'])

# Final evaluation
ensemble_accuracy, _ = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), device)
diverse_accuracy, _ = evaluate_model(diverse_ensemble, (X_test, X_test_spectral, y_test), device)
distilled_accuracy, _ = evaluate_model(distilled_model, (X_test, X_test_spectral, y_test), device)

logging.info(f"Training completed.")
logging.info(f"Ensemble Model - Final Test Accuracy: {ensemble_accuracy:.4f}")
logging.info(f"Diverse Ensemble Model - Final Test Accuracy: {diverse_accuracy:.4f}")
logging.info(f"Distilled Model - Final Test Accuracy: {distilled_accuracy:.4f}")