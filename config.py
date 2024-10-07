import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

CONFIG = {
    'model_path': os.path.join(BASE_DIR, 'models', 'new'),
    'data_path': os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data.mat'),
    'best_model_name': 'best_ensemble_model.pth',
    'best_params_name': 'best_params_ensemble.json',
    'diverse_model_name': 'best_diverse_ensemble_model.pth',
    'distilled_model_name': 'distilled_model.pth',
    'test_data_name': 'test_data.json',
    'confusion_matrix_norm': 'confusion_matrix_normalized.png',
    'confusion_matrix_non_norm': 'confusion_matrix_non_normalized.png',
    'use_pretrained_params': True,  # Set this to False for random parameters
    'seed': 42,
    'initial_params': {
        'model_params': {
            'n_filters': [32, 64, 128],
            'lstm_hidden': 264,
            'lstm_layers': 2,
            'dropout': 0.22931168779815797
        },
        'train_params': {
            'lr': 0.0007068011562596943,
            'batch_size': 32,
            'num_epochs': 1000,
            'patience': 10
        }
    }
}




def get_device(force_cpu=False):
    if force_cpu:
        device = torch.device("cpu")
        print("Forced CPU usage")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for acceleration")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for acceleration (with CPU fallback for unsupported operations)")
    else:
        device = torch.device("cpu")
        print("Using CPU for computation")
    return device