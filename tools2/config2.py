import os
import logging
import torch

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cuda.log')
    ]
)

class CUDAManager:
    _instance = None
    _device = None
    _initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Set CUDA environment variables first
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Start with DSA disabled
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages

    def initialize_cuda(self):
        if self._initialized:
            return self._device

        try:
            if torch.cuda.is_available():
                # Configure PyTorch settings
                torch.backends.cuda.enable_flash_sdp(False)  # Disable Flash SDP for stability
                torch.backends.cudnn.benchmark = False  # Disable benchmarking
                torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
                
                # Initialize CUDA
                torch.cuda.init()
                torch.cuda.empty_cache()

                # Get GPU with most free memory
                device_id = 0
                max_free_memory = 0
                
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            total_memory = torch.cuda.get_device_properties(i).total_memory
                            allocated_memory = torch.cuda.memory_allocated(i)
                            free_memory = total_memory - allocated_memory
                            
                            if free_memory > max_free_memory:
                                max_free_memory = free_memory
                                device_id = i
                    except Exception as e:
                        logging.warning(f"Error checking GPU {i}: {str(e)}")
                        continue

                # Initialize selected device
                self._device = torch.device(f'cuda:{device_id}')
                torch.cuda.set_device(self._device)
                
                # Basic test with DSA disabled
                try:
                    test_tensor = torch.ones(1, device=self._device)
                    _ = test_tensor + 1
                    torch.cuda.synchronize()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    # If basic test passes, enable DSA
                    os.environ['TORCH_USE_CUDA_DSA'] = '1'
                    
                    logging.info(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
                    logging.info(f"Memory available: {free_memory/1e9:.2f}GB")
                    
                except RuntimeError as e:
                    logging.error(f"CUDA initialization failed: {str(e)}")
                    self._device = torch.device('cpu')
            else:
                logging.info("CUDA not available, using CPU")
                self._device = torch.device('cpu')
                
        except Exception as e:
            logging.error(f"Error in CUDA initialization: {str(e)}")
            self._device = torch.device('cpu')
        
        self._initialized = True
        return self._device

    @property
    def device(self):
        if not self._initialized:
            self.initialize_cuda()
        return self._device

# Base configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize CUDA manager first
cuda_manager = CUDAManager.get_instance()
device = cuda_manager.device

import os

# Path to the preprocessed data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '../preprocessing/preprocessed_data')

# Collect all .mat files in the data directory
DATA_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.mat')]

CONFIG = {
    'data_paths': [
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_201_N1.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_201_N2.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_202_N1.mat'),
        os.path.join(BASE_DIR, 'preprocessing', 'preprocessed_data_202_N2.mat'),
    ],
    'model_dir': os.path.join(BASE_DIR, 'models', 'new9'),
    'old_model_path': os.path.join(BASE_DIR, 'models', 'old'),
    
    'settings': {
        'use_pretrained_params': True,
        'use_pretrained_weights': False,
        'seed': 42,
        'force_cpu': False,
        'gpu_settings': {
            'device_id': 0,  # Default to first GPU
            'memory_fraction': 0.8,  # Use 80% of GPU memory
            'allow_growth': True,
            'multi_gpu': False  # Set to True if you want to use multiple GPUs - WARNING: THIS HAS NOT BEEN IMPLEMENTED
        }
    },
    
    'model_names': {
        'ensemble': 'best_ensemble_model.pth',
        'diverse': 'best_diverse_ensemble_model.pth',
        'distilled': 'distilled_model.pth',
        'params': 'tuned_params.json',
        'confusion_matrix': 'confusion_matrix.png',
        'diverse_confusion_matrix': 'diverse_confusion_matrix.png',
        'tensorboard': 'tensorboard_logs'
    },
    
    'model_params': {  # Restructured from initial_params
        'initial': {
            'n_filters': [32, 64, 128],
            'lstm_hidden': 264,
            'lstm_layers': 2,
            'dropout': 0.22931168779815797
        },
        'tuning_ranges': {
            'n_filters': [[16, 32, 64], [32, 64, 128], [64, 128, 256]],
            'lstm_hidden': [128, 256, 512],
            'lstm_layers': [1, 2, 3],
            'dropout': [0.1, 0.5]
        }
    },
    
    'train_params': {
        'initial': {
            'lr': 0.0007068011562596943,
            'batch_size': 32,
            'num_epochs': 10,
            'patience': 10,
            'min_epochs': 20,
            'min_delta': 0.001,
            'early_stopping': {
                'patience': 10,
                'min_epochs': 20,
                'min_delta': 0.001,
                'monitor': ['loss', 'accuracy']
            }
        },
        'tuning_ranges': {
            'n_trials': 5,
            'num_epochs': 10,
            'batch_size': [16, 32, 64],
            'lr': [1e-4, 1e-3],
            'early_stopping': {
                'patience': 5,  # Shorter patience for tuning
                'min_epochs': 10,  # Fewer minimum epochs for tuning
                'min_delta': 0.001,
                'monitor': ['loss', 'accuracy']
            },
            'accumulation_steps': 4,
            'weight_decay': 1e-5,
            'label_smoothing': 0.1
        },
        'scheduler': {  # Add scheduler configuration
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6,
            'verbose': False
            }
        },
    'training_mode': {
        'hyperparameter_tuning': False,  # Set to True if you want to tune
        'find_lr': False,  # Set to True if you want to find best learning rate
    }
    }

cuda_manager = CUDAManager.get_instance()
device = cuda_manager.device

# if device.type == 'cuda':
#     # Set these after device is initialized
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
#     logging.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
#     logging.info(f"CUDA Memory Available: {torch.cuda.get_device_properties(device).total_memory/1e9:.2f}GB")
# else:
#     logging.info("Using CPU device")