import os
import logging
import torch
import multiprocessing

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cuda.log')
    ]
)

# class CUDAManager:
#     _instance = None
#     _device = None
#     _initialized = False

#     @classmethod
#     def get_instance(cls):
#         if cls._instance is None:
#             cls._instance = cls()
#         return cls._instance

#     def __init__(self):
#         total_cores = multiprocessing.cpu_count()
#         num_threads = max(1, total_cores // 2)  # Use half of the total cores, at least 1 thread
#         os.environ['NUMEXPR_MAX_THREADS'] = str(num_threads)
#         print(f"Setting NUMEXPR_MAX_THREADS to {num_threads} out of {total_cores} total cores.")
        
#         # Set CUDA environment variables first
#         os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#         os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#         os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
#         # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
#         os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#         os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Start with DSA disabled
#         os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages

        

#     def initialize_cuda(self):
#         if self._initialized:
#             return self._device

#         try:
#             if torch.cuda.is_available():
#                 # Configure PyTorch settings
#                 torch.backends.cuda.enable_flash_sdp(False)  # Disable Flash SDP for stability
#                 # torch.backends.cudnn.benchmark = False  # Disable benchmarking
#                 torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
#                 # At the start of your script
#                 torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
#                 torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere
#                 torch.backends.cudnn.allow_tf32 = True
                
#                 # Initialize CUDA
#                 torch.cuda.init()
#                 torch.cuda.empty_cache()

#                 # Get GPU with most free memory
#                 device_id = 0
#                 max_free_memory = 0
                
#                 for i in range(torch.cuda.device_count()):
#                     try:
#                         with torch.cuda.device(i):
#                             total_memory = torch.cuda.get_device_properties(i).total_memory
#                             allocated_memory = torch.cuda.memory_allocated(i)
#                             free_memory = total_memory - allocated_memory
                            
#                             if free_memory > max_free_memory:
#                                 max_free_memory = free_memory
#                                 device_id = i
#                     except Exception as e:
#                         logging.warning(f"Error checking GPU {i}: {str(e)}")
#                         continue

#                 # Initialize selected device
#                 self._device = torch.device(f'cuda:{device_id}')
#                 torch.cuda.set_device(self._device)
                
#                 # Basic test with DSA disabled
#                 try:
#                     test_tensor = torch.ones(1, device=self._device)
#                     _ = test_tensor + 1
#                     torch.cuda.synchronize()
#                     del test_tensor
#                     torch.cuda.empty_cache()
                    
#                     # If basic test passes, enable DSA
#                     os.environ['TORCH_USE_CUDA_DSA'] = '1'
                    
#                     logging.info(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
#                     logging.info(f"Memory available: {free_memory/1e9:.2f}GB")
                    
#                 except RuntimeError as e:
#                     logging.error(f"CUDA initialization failed: {str(e)}")
#                     self._device = torch.device('cpu')
#             else:
#                 logging.info("CUDA not available, using CPU")
#                 self._device = torch.device('cpu')
                
#         except Exception as e:
#             logging.error(f"Error in CUDA initialization: {str(e)}")
#             self._device = torch.device('cpu')
        
#         self._initialized = True
#         return self._device

#     @property
#     def device(self):
#         if not self._initialized:
#             self.initialize_cuda()
#         return self._device

class CUDAManager:
    _instance = None
    _device = None
    _initialized = False
    
    @classmethod
    def get_instance(cls, gpu_id=None, memory_fraction=0.8, allow_fallback=True):
        """
        Get CUDAManager instance with specific GPU preferences.
        
        Args:
            gpu_id (int, optional): Preferred GPU ID. If None, selects GPU with most free memory
            memory_fraction (float): Maximum fraction of GPU memory to use (0.0 to 1.0)
            allow_fallback (bool): If True, falls back to other GPUs if preferred GPU is unavailable
        """
        if cls._instance is None:
            cls._instance = cls(gpu_id, memory_fraction, allow_fallback)
        return cls._instance

    def __init__(self, gpu_id=None, memory_fraction=0.8, allow_fallback=True):
        # Core initialization
        self.preferred_gpu = gpu_id
        self.memory_fraction = memory_fraction
        self.allow_fallback = allow_fallback
        self._available_gpus = []
        self._gpu_info = {}
        
        # Configure system
        total_cores = multiprocessing.cpu_count()
        num_threads = max(1, total_cores // 2)
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_threads)
        print(f"Setting NUMEXPR_MAX_THREADS to {num_threads} out of {total_cores} total cores.")
        
        # Basic CUDA configuration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_USE_CUDA_DSA'] = '0'
        
        # Initialize GPU information
        self._scan_gpus()

    def _scan_gpus(self):
        """Scan and collect information about available GPUs"""
        if not torch.cuda.is_available():
            logging.info("No CUDA devices available")
            return

        self._available_gpus = list(range(torch.cuda.device_count()))
        for gpu_id in self._available_gpus:
            with torch.cuda.device(gpu_id):
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory = props.total_memory
                allocated_memory = torch.cuda.memory_allocated(gpu_id)
                free_memory = total_memory - allocated_memory
                
                self._gpu_info[gpu_id] = {
                    'name': props.name,
                    'total_memory': total_memory,
                    'free_memory': free_memory,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count
                }

    def select_gpu(self):
        """Select the most suitable GPU based on configuration and availability"""
        if not self._available_gpus:
            return None

        # If specific GPU requested and available
        if self.preferred_gpu is not None:
            if self.preferred_gpu in self._available_gpus:
                if self._check_gpu_viability(self.preferred_gpu):
                    return self.preferred_gpu
                elif not self.allow_fallback:
                    raise RuntimeError(f"Preferred GPU {self.preferred_gpu} not viable and fallback disabled")
            elif not self.allow_fallback:
                raise RuntimeError(f"Preferred GPU {self.preferred_gpu} not available and fallback disabled")

        # Find GPU with most free memory
        viable_gpus = []
        for gpu_id in self._available_gpus:
            if self._check_gpu_viability(gpu_id):
                viable_gpus.append((gpu_id, self._gpu_info[gpu_id]['free_memory']))
        
        if not viable_gpus:
            return None
            
        return max(viable_gpus, key=lambda x: x[1])[0]

    def _check_gpu_viability(self, gpu_id):
        """Check if GPU meets memory requirements"""
        info = self._gpu_info[gpu_id]
        required_memory = info['total_memory'] * self.memory_fraction
        return info['free_memory'] >= required_memory

    def initialize_cuda(self):
        """Initialize CUDA with selected GPU"""
        if self._initialized:
            return self._device

        try:
            if torch.cuda.is_available():
                # Select GPU
                selected_gpu = self.select_gpu()
                if selected_gpu is None:
                    logging.warning("No viable GPU found, falling back to CPU")
                    self._device = torch.device('cpu')
                    return self._device

                # Important: Set CUDA_VISIBLE_DEVICES before initializing CUDA
                os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
                
                # Re-initialize CUDA with new device visibility
                torch.cuda.empty_cache()
                torch.cuda.init()

                # Configure PyTorch settings
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                # Now device 0 will map to our selected GPU
                self._device = torch.device('cuda:0')
                torch.cuda.set_device(self._device)
                
                # Set memory limit
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(self.memory_fraction, 0)

                # Test GPU
                try:
                    test_tensor = torch.ones(1, device=self._device)
                    _ = test_tensor + 1
                    torch.cuda.synchronize()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    os.environ['TORCH_USE_CUDA_DSA'] = '1'
                    
                    # Log detailed GPU information with physical GPU ID
                    info = self._gpu_info[selected_gpu]
                    logging.info(f"""
GPU Configuration:
    Physical GPU ID: {selected_gpu} ({info['name']})
    Logical Device ID: cuda:0
    Compute Capability: {info['compute_capability']}
    Total Memory: {info['total_memory']/1e9:.2f}GB
    Free Memory: {info['free_memory']/1e9:.2f}GB
    Memory Fraction: {self.memory_fraction:.2f}
    Multi-Processors: {info['multi_processor_count']}
                    """)
                    
                except RuntimeError as e:
                    logging.error(f"GPU initialization failed: {str(e)}")
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

    def get_memory_info(self):
        """Get current memory information for the selected GPU"""
        if self._device.type == 'cuda':
            current_device = self._device.index
            try:
                with torch.cuda.device(current_device):
                    return {
                        'allocated': torch.cuda.memory_allocated()/1e9,
                        'cached': torch.cuda.memory_reserved()/1e9,
                        'free': (torch.cuda.get_device_properties(current_device).total_memory -
                               torch.cuda.memory_allocated())/1e9,
                        'total': torch.cuda.get_device_properties(current_device).total_memory/1e9
                    }
            except Exception as e:
                logging.error(f"Error getting memory info: {str(e)}")
                return None
        return None
    

# Base configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize CUDA manager first
def get_cuda_device(gpu_id=None, memory_fraction=0.8, allow_fallback=True):
    """
    Get CUDA device with specific preferences.
    
    Args:
        gpu_id (int, optional): Preferred GPU ID
        memory_fraction (float): Maximum fraction of GPU memory to use
        allow_fallback (bool): Allow falling back to other GPUs
    """
    manager = CUDAManager.get_instance(
        gpu_id=gpu_id,
        memory_fraction=memory_fraction,
        allow_fallback=allow_fallback
    )
    return manager.device

# Default initialization
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
    'model_dir': os.path.join(BASE_DIR, 'models', 'new11'),
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
            'min_epochs': 10,
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

# cuda_manager = CUDAManager.get_instance()
# device = cuda_manager.device
