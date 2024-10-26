import os
import json
import torch
import logging
from tools.config import CONFIG
import gc

def save_params(params, filename):
    if not os.path.exists(CONFIG['model_dir']):
        os.makedirs(CONFIG['model_dir'])
    filepath = os.path.join(CONFIG['model_dir'], filename)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def load_pretrained_weights(model, model_name):
    if CONFIG['settings']['use_pretrained_weights']:
        pretrained_path = os.path.join(CONFIG['old_model_path'], CONFIG['model_names'][model_name])
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path))
            logging.info(f"Loaded pretrained weights from {pretrained_path}")
        else:
            logging.warning(f"Pretrained weights file not found: {pretrained_path}")
    return model

def save_model(model, filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    torch.save(model.state_dict(), filepath)

def load_model(model, filename):
    filepath = os.path.join(CONFIG['model_dir'], filename)
    if os.path.exists(filepath):
        if CONFIG['settings']['force_cpu']:
            model.load_state_dict(torch.load(filepath, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(filepath))
    return model

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def monitor_memory():
    """Monitor memory usage during training"""
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                cached = torch.cuda.memory_reserved(i) / 1024**2
                logging.info(f"GPU {i}:")
                logging.info(f"  Allocated: {allocated:.1f} MB")
                logging.info(f"  Cached: {cached:.1f} MB")
        except Exception as e:
            logging.error(f"Error monitoring GPU memory: {str(e)}")


def check_cuda_environment():
    """Check CUDA environment and configuration"""
    
    # Set CUDA environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # More conservative memory split
    
    # Check CUDA configuration
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        try:
            n_devices = torch.cuda.device_count()
            logging.info(f"Number of CUDA devices: {n_devices}")
            for i in range(n_devices):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"GPU {i}: {props.name}")
                logging.info(f"  Memory: {props.total_memory / 1024**2:.0f} MB")
                logging.info(f"  Compute Capability: {props.major}.{props.minor}")
        except Exception as e:
            logging.error(f"Error getting CUDA device info: {str(e)}")

def check_cuda():
    """Verify and initialize CUDA environment with DSA handling"""
    if CONFIG['settings']['force_cpu']:
        logging.info("Forced CPU usage due to configuration")
        return torch.device('cpu')
    
    try:
        if torch.cuda.is_available():
            # Set CUDA environment variables for better compatibility
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions
            
            try:
                # Test CUDA initialization with minimal operation
                test_device = torch.device('cuda:0')
                torch.cuda.init()
                
                # Basic test tensor with error handling
                try:
                    with torch.cuda.device(test_device):
                        test_tensor = torch.ones(1, device=test_device)
                        _ = test_tensor + 1
                        torch.cuda.synchronize()  # Ensure operation completes
                        
                        # Get device info
                        props = torch.cuda.get_device_properties(test_device)
                        logging.info(f"Successfully initialized CUDA device: {props.name}")
                        logging.info(f"Compute capability: {props.major}.{props.minor}")
                        
                        # Configure for optimal performance
                        if props.major >= 7:  # For newer GPUs
                            torch.backends.cudnn.enabled = True
                            torch.backends.cudnn.benchmark = True
                            torch.backends.cudnn.deterministic = True
                        
                        return test_device
                        
                except RuntimeError as e:
                    if "device-side assertion" in str(e):
                        logging.error("CUDA device-side assertion error. Falling back to CPU.")
                        os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Disable DSA
                        return torch.device('cpu')
                    raise e
                    
            except RuntimeError as e:
                logging.error(f"CUDA initialization failed: {str(e)}")
                return torch.device('cpu')
                
    except Exception as e:
        logging.error(f"Error checking CUDA availability: {str(e)}")
    
    return torch.device('cpu')

def manage_gpu_memory():
    """Manage GPU memory allocation with improved error handling"""
    if torch.cuda.is_available():
        try:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Check memory status for all devices
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory
                    reserved = torch.cuda.memory_reserved(i)
                    allocated = torch.cuda.memory_allocated(i)
                    free = total - allocated
                    
                    logging.info(f"GPU {i} ({props.name}) Memory:")
                    logging.info(f"  Total: {total / 1024**2:.0f} MB")
                    logging.info(f"  Reserved: {reserved / 1024**2:.0f} MB")
                    logging.info(f"  Allocated: {allocated / 1024**2:.0f} MB")
                    logging.info(f"  Free: {free / 1024**2:.0f} MB")
                    
                except RuntimeError as e:
                    logging.error(f"Error checking memory for GPU {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error managing GPU memory: {str(e)}")


def initialize_cuda_environment():
    """Initialize CUDA environment before any framework loads"""
    
    # Set environment variables first
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'  # Disable DSA for initialization
    
    # Check if CUDA is truly available
    if not torch.cuda.is_available():
        logging.warning("CUDA not available in PyTorch")
        return False
        
    try:
        # Get and verify device count
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logging.error("No CUDA devices found")
            return False
            
        # Try to initialize each device
        available_devices = []
        for i in range(device_count):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                
                # Try to use the device
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    test_tensor = torch.ones(1, device=f'cuda:{i}')
                    _ = test_tensor * 1.0
                    torch.cuda.synchronize()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    # Device is working
                    available_devices.append(i)
                    logging.info(f"GPU {i}: {props.name} (Compute {props.major}.{props.minor})")
                    logging.info(f"  Memory: {props.total_memory / 1024**2:.0f} MB")
                    
            except Exception as e:
                logging.error(f"Error initializing GPU {i}: {str(e)}")
                continue
                
        # Check if we have any working devices
        if not available_devices:
            logging.error("No working CUDA devices found")
            return False
            
        # Set primary device to first working one
        primary_device = available_devices[0]
        torch.cuda.set_device(primary_device)
        logging.info(f"Using GPU {primary_device} as primary device")
        
        # Now enable DSA if needed
        os.environ['TORCH_USE_CUDA_DSA'] = '1'
        
        return True
        
    except Exception as e:
        logging.error(f"CUDA initialization failed: {str(e)}")
        return False
    

def verify_cuda_capability(device):
    """Verify CUDA capability of a device"""
    if device.type != 'cuda':
        return False
    
    try:
        with torch.cuda.device(device):
            # Clear any existing allocations
            torch.cuda.empty_cache()
            
            # Test basic operations
            test_tensor = torch.ones(1, device=device)
            _ = test_tensor * 1.0
            torch.cuda.synchronize()
            
            # Test memory allocation
            try:
                # Try to allocate and free a larger tensor
                large_tensor = torch.zeros(1000, 1000, device=device)
                del large_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logging.warning(f"Memory test failed: {str(e)}")
                return False
                
            return True
            
    except Exception as e:
        logging.error(f"CUDA verification failed: {str(e)}")
        return False
    

def setup_cuda_environment():
    """Set up CUDA environment with proper error handling"""
    # Clear any existing CUDA settings
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    # Set up basic CUDA environment
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Initially disable DSA for setup
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            return None, "CUDA not available"
            
        # Get available devices
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return None, "No CUDA devices found"
            
        # Test each device and find the best one
        best_device = None
        max_free_memory = 0
        
        for i in range(device_count):
            try:
                with torch.cuda.device(i):
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)
                    
                    # Skip if compute capability is too low
                    if props.major < 3:
                        continue
                    
                    # Test device
                    torch.cuda.empty_cache()
                    total_memory = props.total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    free_memory = total_memory - allocated_memory
                    
                    # Try a test computation
                    try:
                        test_tensor = torch.ones(1, device=f'cuda:{i}')
                        _ = test_tensor + 1
                        torch.cuda.synchronize()
                        del test_tensor
                        
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_device = i
                            
                    except RuntimeError:
                        continue
                        
            except Exception as e:
                continue
        
        if best_device is not None:
            # Set up the chosen device
            device = torch.device(f'cuda:{best_device}')
            
            # Configure device settings
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                
                # Set up cuDNN
                if torch.backends.cudnn.is_available():
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = True
                
                # Now enable DSA
                os.environ['TORCH_USE_CUDA_DSA'] = '1'
                
                # Final test
                try:
                    test_tensor = torch.ones(1000, device=device)
                    _ = torch.nn.functional.relu(test_tensor)
                    torch.cuda.synchronize()
                    del test_tensor
                    
                    return device, None
                    
                except RuntimeError as e:
                    return None, f"Final device test failed: {str(e)}"
        
        return None, "No suitable CUDA device found"
        
    except Exception as e:
        return None, f"CUDA initialization failed: {str(e)}"

def initialize_training_device():
    """Initialize device for training with fallback options"""
    # Try CUDA initialization
    device, error = setup_cuda_environment()
    
    if device is not None:
        logging.info(f"Successfully initialized CUDA device: {torch.cuda.get_device_name(device)}")
        return device
        
    # If CUDA failed, log error and fall back to CPU
    logging.warning(f"CUDA initialization failed: {error}")
    logging.info("Falling back to CPU")
    return torch.device('cpu')


def log_gpu_memory(device):
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory
        logging.info(f"GPU Memory on {torch.cuda.get_device_name(device)}:")
        logging.info(f"  Total: {total_memory / 1024**2:.0f} MB")
        logging.info(f"  Allocated: {allocated_memory / 1024**2:.0f} MB")
        logging.info(f"  Free: {free_memory / 1024**2:.0f} MB")