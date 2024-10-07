import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sleepdetector_new import ImprovedSleepdetector
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from scipy.signal import welch
from imblearn.over_sampling import SMOTE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import os
import logging
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from scipy.interpolate import CubicSpline
from torch_lr_finder import LRFinder
import torch.nn.functional as F
from classes import EarlyStopping, EnsembleModel, DiverseEnsembleModel, OptunaPruneCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data loading and preprocessing
def load_data(filepath, add_dim=False):
    try:
        # Load the data from the .mat file
        mat_file = sio.loadmat(filepath)
        
        # Stack the signals into x
        x = np.stack((mat_file['sig1'], mat_file['sig2'], mat_file['sig3'], mat_file['sig4']), axis=1)
        x = torch.from_numpy(x).float()  # Convert to PyTorch tensor
        
        # Load the labels
        y = torch.from_numpy(mat_file['labels'].flatten()).long()
        
        # Remove epochs where y is -1 (if any)
        valid_indices = y != -1
        x = x[valid_indices]
        y = y[valid_indices]
        
        # Ensure x is in shape [number of epochs, 4, 3000]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if add_dim:
            x = x.unsqueeze(1)  # Add an extra dimension if required
        
        print(f"Loaded data shape: {x.shape}, Labels shape: {y.shape}")
        
        return x, y

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def extract_spectral_features(x):
    features = []
    for epoch in x:
        epoch_features = []
        for channel in epoch:
            # Check if channel is a PyTorch tensor, if so convert to numpy array
            if isinstance(channel, torch.Tensor):
                channel = channel.numpy()
            f, psd = welch(channel, fs=100, nperseg=1000)
            delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
            theta = np.sum(psd[(f > 4) & (f <= 8)])
            alpha = np.sum(psd[(f > 8) & (f <= 13)])
            beta = np.sum(psd[(f > 13) & (f <= 30)])
            epoch_features.extend([delta, theta, alpha, beta])
        features.append(epoch_features)
    return np.array(features)

def prepare_data(x, y, test_size=0.2, split=True):
    """
    Prepare data for training or testing.
    
    :param x: Input data tensor
    :param y: Labels tensor
    :param test_size: Proportion of the dataset to include in the test split
    :param split: If True, split the data into train and test sets. If False, process all data without splitting.
    :return: Processed data tensors
    """
    if split:
        X_train, X_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=test_size, stratify=y, random_state=42)
        
        X_train_spectral = extract_spectral_features(torch.from_numpy(X_train))
        X_test_spectral = extract_spectral_features(torch.from_numpy(X_test))
        
        X_train_combined = np.concatenate([X_train.reshape(X_train.shape[0], -1), X_train_spectral], axis=1)
        X_test_combined = np.concatenate([X_test.reshape(X_test.shape[0], -1), X_test_spectral], axis=1)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
        
        original_shape = list(X_train.shape)
        original_shape[0] = X_train_resampled.shape[0]
        spectral_shape = (X_train_resampled.shape[0], X_train_spectral.shape[1])
        
        X_train_final = X_train_resampled[:, :-X_train_spectral.shape[1]].reshape(original_shape)
        X_train_spectral_final = X_train_resampled[:, -X_train_spectral.shape[1]:].reshape(spectral_shape)
        
        return (torch.from_numpy(X_train_final).float(),
                torch.from_numpy(X_train_spectral_final).float(),
                torch.from_numpy(y_train_resampled).long(),
                torch.from_numpy(X_test).float(),
                torch.from_numpy(X_test_spectral).float(),
                torch.from_numpy(y_test).long())
    else:
        X_spectral = extract_spectral_features(x)
        
        return (x.float(),
                torch.from_numpy(X_spectral).float(),
                y.long())
    



def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps),
            CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
        ],
        milestones=[num_warmup_steps]
    )
    return lr_scheduler

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

def find_lr(model, train_loader, optimizer, criterion, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

    # You can also get the suggested learning rate programmatically
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {suggested_lr}")
    return suggested_lr

# Model initialization function
def initialize_model(config, best_model_params, device):
    ensemble_model = EnsembleModel(best_model_params, n_models=3).to(device)
    
    if config['use_pretrained']:
        if os.path.exists(config['pretrained_weights_path']):
            ensemble_model.load_state_dict(torch.load(config['pretrained_weights_path']))
            logging.info(f"Loaded pre-trained weights from {config['pretrained_weights_path']}")
        else:
            logging.warning(f"Pre-trained weights file not found at {config['pretrained_weights_path']}. Initializing with random weights.")
    else:
        initial_weights_path = os.path.join(config['new_model_path'], config['initial_weights_name'])
        torch.save(ensemble_model.state_dict(), initial_weights_path)
        logging.info(f"Initialized with random weights. Initial weights saved to {initial_weights_path}")
    
    return ensemble_model

# Training and evaluation functions
def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, device, epochs=100, accumulation_steps=4):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    best_accuracy = 0
    best_model_state = None
    scaler = GradScaler()
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        for i, (batch_x, batch_x_spectral, batch_y) in enumerate(train_loader):
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)
            
            with autocast('cuda'):
                outputs = model(batch_x, batch_x_spectral)
                loss = criterion(outputs, batch_y)
                loss = loss / accumulation_steps

            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Evaluate the model
        accuracy, _, _ = evaluate_model(model, val_data, device)
        
        # Step the scheduler with the validation accuracy
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(accuracy)
        else:
            scheduler.step()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
        
        early_stopping(1 - accuracy, model)  # Use 1 - accuracy as a proxy for loss
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return best_model_state, best_accuracy

def train_ensemble_model(config, best_model_params, train_loader, val_data, device, num_epochs=100):
    num_warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
    num_training_steps = len(train_loader) * num_epochs

    if config['use_pretrained']:
        model = initialize_model(config, best_model_params, device)
        logging.info("Initialized ensemble with pre-trained weights")
    else:
        model = DiverseEnsembleModel(best_model_params, n_models=3).to(device)
        logging.info("Initialized ensemble with random weights")

    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=1e-2)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    criterion = nn.CrossEntropyLoss()

    best_model_state, best_accuracy = train_model(
        model, train_loader, val_data,
        optimizer, scheduler, criterion, device,
        epochs=num_epochs, accumulation_steps=4
    )

    return best_model_state, best_accuracy


def distill_knowledge(teacher_model, student_model, train_loader, val_data, device, num_epochs=50):
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = get_scheduler(optimizer, num_warmup_steps=len(train_loader) * 5, num_training_steps=len(train_loader) * num_epochs)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    teacher_model.eval()
    for epoch in tqdm(range(num_epochs), desc="Distillation Progress"):
        student_model.train()
        for batch_x, batch_x_spectral, batch_y in train_loader:
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(batch_x, batch_x_spectral)
            
            student_outputs = student_model(batch_x, batch_x_spectral)
            
            loss = criterion(F.log_softmax(student_outputs / 2, dim=1),
                             F.softmax(teacher_outputs / 2, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Evaluate student model
        accuracy, _, _ = evaluate_model(student_model, val_data, device)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}")
    
    return student_model

def evaluate_model(model, data, device, return_loss=False):
    model.eval()
    try:
        X, X_spectral, y = data
        logging.info(f"Data shapes - X: {X.shape}, X_spectral: {X_spectral.shape}, y: {y.shape}")
        
        with torch.no_grad():
            outputs = model(X.to(device), X_spectral.to(device))
            logging.info(f"Model output shape: {outputs.shape}")
            
            _, predicted = torch.max(outputs, 1)
            logging.info(f"Predicted shape: {predicted.shape}")
            
            accuracy = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
            kappa = cohen_kappa_score(y.cpu().numpy(), predicted.cpu().numpy())

        if return_loss:
            return loss.item()
        else:
            
            return accuracy, kappa, predicted.cpu().numpy()
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise

# Hyperparameter optimization
def objective(trial, X_train, X_train_spectral, y_train, X_test, X_test_spectral, y_test, device):
    model_params = {
        'n_filters': trial.suggest_categorical('n_filters', ((32, 64, 128), (64, 128, 256))),  # Changed to tuple of tuples
        'lstm_hidden': trial.suggest_int('lstm_hidden', 64, 512),
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    train_params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    }
    
    model = ImprovedSleepdetector(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    train_loader = DataLoader(TensorDataset(X_train, X_train_spectral, y_train), batch_size=train_params['batch_size'], shuffle=True)
    
    _, accuracy = train_model(model, train_loader, (X_test, X_test_spectral, y_test), optimizer, scheduler, nn.CrossEntropyLoss(), device, epochs=10)
    
    logging.info(f"Trial {trial.number} - Accuracy: {accuracy:.4f}")
    return accuracy


# Cross-validation
def cross_validate(X, X_spectral, y, model_params, train_params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        X_spectral_train_fold, X_spectral_val_fold = X_spectral[train_idx], X_spectral[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = DiverseEnsembleModel(model_params, n_models=3).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=train_params['lr'], weight_decay=1e-2)
        train_loader = DataLoader(TensorDataset(X_train_fold, X_spectral_train_fold, y_train_fold), batch_size=train_params['batch_size'], shuffle=True)
        
        num_epochs = 50  # You can adjust this
        num_warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
        
        best_model_state, accuracy = train_model(model, train_loader, (X_val_fold, X_spectral_val_fold, y_val_fold), optimizer, scheduler, nn.CrossEntropyLoss(), device, epochs=num_epochs)
        scores.append(accuracy)
        
        logging.info(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
    
    logging.info(f"Average Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores

# Confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Class names in the correct order (0 to 4)
    class_names = ['N3', 'N2', 'N1', 'REM', 'Awake']

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, square=True, xticklabels=class_names, yticklabels=class_names)
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig