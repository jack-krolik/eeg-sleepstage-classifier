import torch
import numpy as np
import scipy.io as sio
import logging
from scipy.signal import welch
from scipy.interpolate import CubicSpline
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from classes import EnsembleModel, DiverseEnsembleModel
from config import CONFIG, get_device
import random
import math


device = get_device()

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    try:
        mat_file = sio.loadmat(CONFIG['data_path'])
        
        x = np.stack((mat_file['sig1'], mat_file['sig2'], mat_file['sig3'], mat_file['sig4']), axis=1)
        x = torch.from_numpy(x).float()
        
        y = torch.from_numpy(mat_file['labels'].flatten()).long()
        
        valid_indices = y != -1
        x = x[valid_indices]
        y = y[valid_indices]
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        print(f"Loaded data shape: {x.shape}, Labels shape: {y.shape}")
        
        return x, y

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(X, X_spectral):
    # Remove outliers (e.g., clip values beyond 3 standard deviations)
    X = torch.clamp(X, -3, 3)
    
    # Ensure X_spectral is non-negative (if it represents power spectral density)
    X_spectral = torch.clamp(X_spectral, min=0)
    
    return X, X_spectral


def generate_random_params():
    return {
        'model_params': {
            'n_filters': [random.choice([16, 32, 64]) for _ in range(3)],
            'lstm_hidden': random.choice([128, 256, 512]),
            'lstm_layers': random.randint(1, 3),
            'dropout': random.uniform(0.1, 0.5)
        },
        'train_params': {
            'lr': 10**random.uniform(-4, -2),
            'batch_size': random.choice([16, 32, 64]),
            'num_epochs': 1000,
            'patience': 10
        }
    }

def initialize_model(device):
    if CONFIG['use_pretrained_params']:
        params = CONFIG['initial_params']
        logging.info("Using predefined parameters from config.")
    else:
        params = generate_random_params()
        logging.info("Using randomly generated parameters.")
    
    model_params = params['model_params']
    ensemble_model = EnsembleModel(model_params, n_models=3).to(device)
    ensemble_model.apply(ensemble_model._init_weights)
    
    logging.info(f"Model initialized with parameters: {model_params}")
    return ensemble_model, params


def extract_spectral_features(x):
    features = []
    for channel in range(x.shape[0]):  # Iterate over channels
        # Convert to NumPy array for scipy.signal.welch
        channel_data = x[channel].cpu().numpy()
        f, psd = welch(channel_data, fs=100, nperseg=min(1000, len(channel_data)))
        delta = np.sum(psd[(f >= 0.5) & (f <= 4)])
        theta = np.sum(psd[(f > 4) & (f <= 8)])
        alpha = np.sum(psd[(f > 8) & (f <= 13)])
        beta = np.sum(psd[(f > 13) & (f <= 30)])
        features.extend([delta, theta, alpha, beta])
    return np.array(features)





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
            augmented = torch.from_numpy(time_warp(x[i].unsqueeze(0).numpy(), sigma=0.3, knot=5)).squeeze(0)
            augmented_x.append(augmented)
            augmented_x_spectral.append(x_spectral[i])  # Duplicate spectral features for augmented data
            augmented_y.append(y[i])
    return torch.stack(augmented_x), torch.stack(augmented_x_spectral), torch.tensor(augmented_y)


def simple_oversample(X, X_spectral, y):
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    oversampled_X = []
    oversampled_X_spectral = []
    oversampled_y = []
    
    for class_label in class_counts:
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)
        n_oversample = max_count - n_samples
        
        oversampled_X.append(X[class_indices])
        oversampled_X_spectral.append(X_spectral[class_indices])
        oversampled_y.extend([class_label] * n_samples)
        
        if n_oversample > 0:
            oversampled_indices = np.random.choice(class_indices, size=n_oversample, replace=True)
            oversampled_X.append(X[oversampled_indices])
            oversampled_X_spectral.append(X_spectral[oversampled_indices])
            oversampled_y.extend([class_label] * n_oversample)
    
    return np.concatenate(oversampled_X), np.concatenate(oversampled_X_spectral), np.array(oversampled_y)
    
def prepare_data(x, y, test_size=0.2, val_size=0.1):
    # Convert PyTorch tensors to NumPy arrays for scikit-learn and SMOTE
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    # Split the data into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(x_np, y_np, test_size=test_size, stratify=y_np, random_state=42)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), stratify=y_train_val, random_state=42)
    
    # Convert back to PyTorch tensors for spectral feature extraction
    X_train_torch = torch.from_numpy(X_train).float()
    X_val_torch = torch.from_numpy(X_val).float()
    X_test_torch = torch.from_numpy(X_test).float()

    # Extract spectral features
    X_train_spectral = np.array([extract_spectral_features(x) for x in X_train_torch])
    X_val_spectral = np.array([extract_spectral_features(x) for x in X_val_torch])
    X_test_spectral = np.array([extract_spectral_features(x) for x in X_test_torch])
    
    print("Original train set class distribution:")
    print(Counter(y_train))
    
    # Reshape the data for SMOTE
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_train_spectral_reshaped = X_train_spectral.reshape(X_train_spectral.shape[0], -1)
    X_combined = np.hstack((X_train_reshaped, X_train_spectral_reshaped))
    
    # Check if we have enough samples in each class for SMOTE
    class_counts = Counter(y_train)
    min_samples = min(class_counts.values())
    
    if min_samples >= 6:  # SMOTE requires at least 6 samples in the minority class
        # Apply SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_combined, y_train)
        
        print("After SMOTE train set class distribution:")
        print(Counter(y_resampled))
        
        # Reshape the resampled data back to the original shape
        X_train_resampled = X_resampled[:, :X_train_reshaped.shape[1]].reshape(-1, X_train.shape[1], X_train.shape[2])
        X_train_spectral_resampled = X_resampled[:, X_train_reshaped.shape[1]:].reshape(-1, X_train_spectral.shape[1])
    else:
        print("Not enough samples in minority class for SMOTE. Using simple oversampling.")
        X_train_resampled, X_train_spectral_resampled, y_resampled = simple_oversample(X_train, X_train_spectral, y_train)
        
        print("After simple oversampling train set class distribution:")
        print(Counter(y_resampled))
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train_resampled).float()
    X_train_spectral = torch.from_numpy(X_train_spectral_resampled).float()
    y_train = torch.from_numpy(y_resampled).long()
    X_val = torch.from_numpy(X_val).float()
    X_val_spectral = torch.from_numpy(X_val_spectral).float()
    y_val = torch.from_numpy(y_val).long()
    X_test = torch.from_numpy(X_test).float()
    X_test_spectral = torch.from_numpy(X_test_spectral).float()
    y_test = torch.from_numpy(y_test).long()
    
    return X_train, X_train_spectral, y_train, X_val, X_val_spectral, y_val, X_test, X_test_spectral, y_test


def get_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def find_lr(model, train_loader, optimizer, criterion, device, num_iter=100, start_lr=1e-8, end_lr=1):
    logging.info("Starting learning rate finder...")
    model.train()
    num_samples = len(train_loader.dataset)
    update_step = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    optimizer.param_groups[0]["lr"] = lr
    running_loss = 0
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []
    
    progress_bar = tqdm(range(num_iter), desc="Finding best LR")
    for i in progress_bar:
        try:
            inputs, spectral_features, targets = next(iter(train_loader))
        except StopIteration:
            train_loader = iter(train_loader)
            inputs, spectral_features, targets = next(train_loader)
        
        inputs, spectral_features, targets = inputs.to(device), spectral_features.to(device), targets.to(device)
        batch_size = inputs.size(0)
        batch_num += 1
        
        optimizer.zero_grad()
        outputs = model(inputs, spectral_features)
        loss = criterion(outputs, targets)
        
        # Compute the smoothed loss
        running_loss = 0.98 * running_loss + 0.02 * loss.item()
        smoothed_loss = running_loss / (1 - 0.98**batch_num)
        
        # Record the best loss
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            logging.info(f"Loss is exploding, stopping early at lr={lr:.2e}")
            break
        
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
        
        progress_bar.set_postfix({'loss': f'{smoothed_loss:.4f}', 'lr': f'{lr:.2e}'})
    
    plt.figure(figsize=(10, 6))
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.xlabel("Log Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate vs. Loss")
    plt.savefig('lr_finder_plot.png')
    plt.close()
    
    # Find the learning rate with the steepest negative gradient
    smoothed_losses = np.array(losses[10:-5])
    smoothed_lrs = np.array(log_lrs[10:-5])
    gradients = (smoothed_losses[1:] - smoothed_losses[:-1]) / (smoothed_lrs[1:] - smoothed_lrs[:-1])
    best_lr = 10 ** smoothed_lrs[np.argmin(gradients)]
    
    # Adjust the learning rate to be slightly lower than the one with steepest gradient
    best_lr *= 0.1
    
    logging.info(f"Learning rate finder completed. Suggested Learning Rate: {best_lr:.2e}")
    logging.info("Learning rate vs. loss plot saved as 'lr_finder_plot.png'")
    return best_lr


def get_class_weights(y):
    class_counts = torch.bincount(y)
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    return class_weights


def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, device, epochs=100, patience=10, accumulation_steps=4):
    scaler = GradScaler()
    best_accuracy = 0
    best_loss = float('inf')
    best_model_state = None
    no_improve = 0
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        for i, (batch_x, batch_x_spectral, batch_y) in enumerate(train_loader):
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)
            
            with autocast(device_type=device.type):
                outputs = model(batch_x, batch_x_spectral)
                loss = criterion(outputs, batch_y)
                loss = loss / accumulation_steps  # Normalize the loss because it is accumulated
            
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        # Validation loop
        model.eval()
        val_loss, val_accuracy, _ = evaluate_model(model, val_data, criterion, device)
        
        # Step the scheduler with the validation loss
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"LR: {current_lr:.6f}")
        
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return best_model_state, best_accuracy

def evaluate_model(model, data, criterion, device):
    model.eval()
    X, X_spectral, y = data
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    
    with torch.no_grad():
        outputs = model(X.to(device), X_spectral.to(device))
        loss = criterion(outputs, y.to(device))
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        total += y.size(0)
        correct += (predicted == y.to(device)).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / total
    return avg_loss, accuracy, np.array(all_predictions)

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