# %%
# Imports
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
from torch.optim.lr_scheduler import SequentialLR
from torch.amp import autocast, GradScaler
from scipy.interpolate import CubicSpline
from torch_lr_finder import LRFinder
import torch.nn.functional as F
from sleepdetector_new import ImprovedSleepdetector
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
from sklearn.metrics import accuracy_score, cohen_kappa_score
import torch.nn.functional as F
import math
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

set_seed()


# %%
class EnsembleModel(nn.Module):
    def __init__(self, model_params, n_models=3):
        super().__init__()
        self.models = nn.ModuleList([ImprovedSleepdetector(**model_params) for _ in range(n_models)])
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=0.1)  # Reduced 'a' parameter
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=0.1)  # Reduced gain
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x, spectral_features):
        outputs = [model(x.clone(), spectral_features.clone()) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

class DiverseEnsembleModel(nn.Module):
    def __init__(self, model_params, n_models=3):
        super().__init__()
        self.models = nn.ModuleList([
            ImprovedSleepdetector(**{**model_params, 'dropout': model_params['dropout'] * (i+1)/n_models})
            for i in range(n_models)
        ])
    
    def forward(self, x, spectral_features):
        outputs = [model(x.clone(), spectral_features.clone()) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# %%
def load_best_params(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params['best_model_params']



def load_data_and_params(config):
    data_dict = torch.load(config['preprocessed_data_path'])
    best_params_path = os.path.join(config['previous_model_path'], config['best_params_name'])
    best_params = load_best_params(best_params_path)
    return data_dict, best_params


def print_model_structure(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

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


def initialize_model(config, best_model_params, device):
    ensemble_model = EnsembleModel(best_model_params, n_models=3).to(device)
    
    if config['use_pretrained']:
        if os.path.exists(config['pretrained_weights_path']):
            # Load the state dict
            state_dict = torch.load(config['pretrained_weights_path'], map_location=device)
            
            # Filter out unnecessary keys
            model_dict = ensemble_model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # Update the model dict
            model_dict.update(pretrained_dict)
            
            # Load the filtered state dict
            ensemble_model.load_state_dict(model_dict, strict=False)
            
            # Log the loaded and missing keys
            loaded_keys = set(pretrained_dict.keys())
            all_keys = set(model_dict.keys())
            missing_keys = all_keys - loaded_keys
            
            logging.info(f"Loaded pre-trained weights from {config['pretrained_weights_path']}")
            logging.info(f"Number of loaded parameters: {len(loaded_keys)}")
            logging.info(f"Number of missing parameters: {len(missing_keys)}")
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys}")
        else:
            logging.warning(f"Pre-trained weights file not found at {config['pretrained_weights_path']}. Initializing with random weights.")
    else:
        ensemble_model.apply(ensemble_model._init_weights)
        logging.info("Model initialized with random weights.")
    
    return ensemble_model


def load_params_and_initialize_model(config, device):
    params_path = os.path.join(config['new_model_path'], config['best_params_name'])
    
    try:
        with open(params_path, 'r') as f:
            loaded_params = json.load(f)
        
        best_model_params = loaded_params['best_model_params']
        best_train_params = loaded_params['best_train_params']
        
        print("Parameters loaded successfully.")
        print(f"Best model parameters: {best_model_params}")
        print(f"Best training parameters: {best_train_params}")
        
        ensemble_model = initialize_model(config, best_model_params, device)
        
        return ensemble_model, best_model_params, best_train_params
    
    except FileNotFoundError:
        print(f"Error: The file {params_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {params_path} is not a valid JSON file.")
        raise
    except KeyError as e:
        print(f"Error: The key {e} was not found in the loaded parameters.")
        raise



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





def train_model(model, train_loader, val_data, optimizer, scheduler, criterion, device, epochs=100, patience=10):
    scaler = GradScaler()
    best_accuracy = 0
    best_model_state = None
    no_improve = 0
    
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        for batch_x, batch_x_spectral, batch_y in train_loader:
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)
            logging.debug(f"Batch X shape: {batch_x.shape}, Batch X spectral shape: {batch_x_spectral.shape}")
            logging.debug(f"Batch X range: [{batch_x.min()}, {batch_x.max()}], Batch X spectral range: [{batch_x_spectral.min()}, {batch_x_spectral.max()}]")
            if torch.isnan(batch_x).any() or torch.isnan(batch_x_spectral).any():
                logging.error("NaN detected in input data")
                continue
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type):
                outputs = model(batch_x, batch_x_spectral)
                loss = criterion(outputs, batch_y)
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN or Inf loss detected: {loss.item()}")
                    logging.error(f"Outputs: {outputs}")
                    logging.error(f"Targets: {batch_y}")
                    return None, 0  # Stop training if NaN or Inf loss occurs
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"NaN or Inf loss detected: {loss.item()}")
                return None, 0  # Stop training if NaN or Inf loss occurs
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm_before = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
            logging.info(f"Gradient norm before clipping: {grad_norm_before}")
            if torch.isfinite(grad_norm_before):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased from 0.1 to 1.0
                logging.info(f"Gradient norm after clipping: {grad_norm}")
            else:
                logging.error(f"Infinite gradient norm detected before clipping: {grad_norm_before}")
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        logging.error(f"Infinite gradient in parameter: {name}")
                return None, 0  # Stop training if infinite gradient is detected
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logging.error(f"NaN or Inf gradient norm detected: {grad_norm}")
                return None, 0  # Stop training if NaN or Inf gradient norm occurs
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()

            running_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_accuracy = evaluate_model(model, val_data, device)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Grad norm: {grad_norm:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            logging.error(f"NaN detected in {name}")
    
    return best_model_state, best_accuracy

def evaluate_model(model, data, device):
    model.eval()
    X, X_spectral, y = data
    with torch.no_grad():
        outputs = model(X.to(device), X_spectral.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y.cpu().numpy(), predicted.cpu().numpy())
    return accuracy

def distill_knowledge(teacher_model, student_model, train_loader, val_data, device, num_epochs=50, log_interval=5):
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = get_scheduler(optimizer, num_warmup_steps=len(train_loader) * 5, num_training_steps=len(train_loader) * num_epochs)
    criterion = nn.KLDivLoss(reduction='batchmean')
    temperature = 2.0  # Make sure this value is reasonable

    teacher_model.eval()
    overall_progress = tqdm(total=num_epochs, desc="Overall Distillation Progress", position=0)
    
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        
        epoch_progress = tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        for batch_x, batch_x_spectral, batch_y in epoch_progress:
            batch_x, batch_x_spectral, batch_y = batch_x.to(device), batch_x_spectral.to(device), batch_y.to(device)

            # Check for NaNs or Infs in input data
            if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                print("NaNs or Infs detected in input data!")
            
            if torch.isnan(batch_x_spectral).any() or torch.isinf(batch_x_spectral).any():
                print("NaNs or Infs detected in spectral input data!")

            
            with torch.no_grad():
                teacher_outputs = teacher_model(batch_x, batch_x_spectral)
            
            student_outputs = student_model(batch_x, batch_x_spectral)
            

            epsilon = 1e-8  # Small constant to prevent log(0)
            loss = criterion(
                F.log_softmax(student_outputs / temperature + epsilon, dim=1),
                F.softmax(teacher_outputs / temperature + epsilon, dim=1)
            )

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            epoch_progress.set_postfix({'loss': f'{running_loss/(epoch_progress.n+1):.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        
        # Evaluate and log every log_interval epochs
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            accuracy = evaluate_model(student_model, val_data, device)
            logging.info(f"Distillation Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        overall_progress.update(1)
    
    overall_progress.close()
    return student_model


# %% [markdown]
# # Load Data

# %%
preprocessed_file_name = './preprocessing/preprocessed_data.mat'

# Load the data
x, y = load_data(preprocessed_file_name)
print(f"Loaded data shape: {x.shape}, Labels shape: {y.shape}")

# Prepare the data (includes SMOTE)
X_train, X_train_spectral, y_train, X_val, X_val_spectral, y_val, X_test, X_test_spectral, y_test = prepare_data(x, y)

print("After SMOTE:")
print(f"X_train shape: {X_train.shape}")
print(f"X_train_spectral shape: {X_train_spectral.shape}")
print(f"y_train shape: {y_train.shape}")

# Identify minority classes for augmentation
class_counts = Counter(y_train.numpy())
minority_classes = [cls for cls, count in class_counts.items() if count < len(y_train) / len(class_counts) * 0.5]

# Apply augmentation
X_train, X_train_spectral, y_train = augment_minority_classes(X_train, X_train_spectral, y_train, minority_classes)

print("After augmentation:")
print(f"X_train shape: {X_train.shape}")
print(f"X_train_spectral shape: {X_train_spectral.shape}")
print(f"y_train shape: {y_train.shape}")
print("Final class distribution:")
print(Counter(y_train.numpy()))

# %%


# %% [markdown]
# # Hyparameter Optimization

# %%
config = {
    'previous_model_path': './models/original/',
    'new_model_path': './models/new/',
    'best_model_name': 'best_ensemble_model.pth',
    'best_params_name': 'best_params_ensemble.json',
    'test_data_name': 'test_data.json',
    'confusion_matrix_norm': 'confusion_matrix_normalized.png',
    'confusion_matrix_non_norm': 'confusion_matrix_non_normalized.png',
    # 'initial_weights_name': 'best_ensemble_model.pth',
    'initial_weights_name': 'best_ensemble_model.pth',
    'use_pretrained': False,  # Set to True to use previous weights
}

# Ensure the model save directory exists

os.makedirs(config['new_model_path'], exist_ok=True)
config['pretrained_weights_path'] = os.path.join(config['previous_model_path'], config['best_model_name'])

# %%
ensemble_model, best_model_params, best_train_params = load_params_and_initialize_model(config, device)



# %% [markdown]
# # TRAINING ENSEMBLE

# %%


# %%
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.label_to_indices = {label: np.where(labels == label)[0] for label in set(labels)}
        self.used_label_indices_count = {label: 0 for label in set(labels)}
        self.count = 0
        self.n_classes = len(set(labels))
        self.n_samples = len(labels)
        
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_samples:
            classes = list(self.label_to_indices.keys())
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                    self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.batch_size // self.n_classes
                ])
                self.used_label_indices_count[class_] += self.batch_size // self.n_classes
                if self.used_label_indices_count[class_] + self.batch_size // self.n_classes > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_samples // self.batch_size



# %%
logging.info("Starting training process...")
overall_steps = 4  # LR finding, Ensemble training, Diverse Ensemble training, Knowledge Distillation
overall_progress = tqdm(total=overall_steps, desc="Overall Training Progress", position=0)

# %%
balanced_sampler = BalancedBatchSampler(y_train.numpy(), batch_size=best_train_params['batch_size'])
train_loader = DataLoader(TensorDataset(X_train, X_train_spectral, y_train), batch_sampler=balanced_sampler)

# %%
# Set up loss function
class_weights = get_class_weights(y_train).to(device)
logging.info(f"Class weights: {class_weights}")
if torch.isnan(class_weights).any() or torch.isinf(class_weights).any():
    logging.error("NaN or Inf detected in class weights")
criterion = nn.CrossEntropyLoss(weight=class_weights + 1e-6, label_smoothing=0.1)

# Find best learning rate
initial_optimizer = optim.Adam(ensemble_model.parameters(), lr=1e-8, weight_decay=1e-5)
best_lr = find_lr(ensemble_model, train_loader, initial_optimizer, criterion, device, num_iter=100, start_lr=1e-10, end_lr=1e-2)
best_lr *= 0.01  # Further reduce the found learning rate

logging.info(f"Adjusted best learning rate: {best_lr:.2e}")




# %%
assert not torch.isnan(X_train).any(), "NaN values found in X_train"
assert not torch.isinf(X_train).any(), "Inf values found in X_train"
assert not torch.isnan(X_train_spectral).any(), "NaN values found in X_train_spectral"
assert not torch.isinf(X_train_spectral).any(), "Inf values found in X_train_spectral"

# %%
num_epochs = 1000  # Adjust as needed
num_warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
num_training_steps = len(train_loader) * num_epochs

# Set up optimizer and scheduler with best learning rate
optimizer = optim.AdamW(ensemble_model.parameters(), lr=best_lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


X_train = (X_train - X_train.mean()) / (X_train.std() + 1e-8)
X_train_spectral = (X_train_spectral - X_train_spectral.mean()) / (X_train_spectral.std() + 1e-8)
X_val = (X_val - X_val.mean()) / (X_val.std() + 1e-8)
X_val_spectral = (X_val_spectral - X_val_spectral.mean()) / (X_val_spectral.std() + 1e-8)
X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)
X_test_spectral = (X_test_spectral - X_test_spectral.mean()) / (X_test_spectral.std() + 1e-8)
# Train model
logging.info("Training ensemble model...")
try:
    best_model_state, best_accuracy = train_model(
        ensemble_model, train_loader, (X_val, X_val_spectral, y_val),
        optimizer, scheduler, criterion, device, epochs=num_epochs
    )
    overall_progress.update(1)

    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(config['new_model_path'], config['best_model_name']))
        logging.info(f"Best ensemble model saved. Final validation accuracy: {best_accuracy:.4f}")
        
        # Evaluate on test set
        ensemble_model.load_state_dict(best_model_state)
        test_accuracy = evaluate_model(ensemble_model, (X_test, X_test_spectral, y_test), device)
        logging.info(f"Ensemble Model - Final Test Accuracy: {test_accuracy:.4f}")
    else:
        logging.error("Training failed due to NaN loss.")
except Exception as e:
    logging.error(f"An error occurred during training: {str(e)}")
    raise


