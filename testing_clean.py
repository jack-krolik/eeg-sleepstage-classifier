import os
import json
import torch
from tools_clean.config_clean import device, DATA_FILES, CONFIG, cuda_manager
from tools_clean.classes_clean import EnsembleModel, SleepDataManager, SleepStageEvaluator
from tools_clean.functions_clean import convert_to_serializable, format_class_distribution, set_seed
from tools_clean.utils_clean import *
import logging
from datetime import datetime
import pandas as pd

set_seed(CONFIG['settings']['seed'])


def evaluate_new_data(model_dir, new_data_files, model_params=None, batch_size=32, save_results=True):
    """
    Evaluate a saved model on new data
    
    Args:
        model_dir: Directory containing the saved model and configuration
        new_data_files: List of paths to new .mat files to evaluate
        model_params: Optional model parameters (if None, loads from config)
        batch_size: Batch size for evaluation
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = os.path.join(model_dir, f'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load model configuration if not provided
    if model_params is None:
        config_path = os.path.join(model_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            model_params = config['model_params']
    
    # Load model
    model_path = os.path.join(model_dir, 'best_model.pt')
    model = EnsembleModel(model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load new data
    logging.info(f"\nLoading new data from {len(new_data_files)} files...")
    data_manager = SleepDataManager(
        data_files=new_data_files,
        val_ratio=0.0,  # No validation split needed for evaluation
        seed=42
    )
    data_manager.load_and_preprocess()
    
    # Set up evaluator
    evaluator = SleepStageEvaluator(model_dir=eval_dir)
   

    # Evaluate on new data
    evaluation_results = evaluator.evaluate_model(
        model=model,
        X=data_manager.data['x'],
        X_spectral=data_manager.data['x_spectral'],
        y=data_manager.data['y'],
        model_name="Model Evaluation on New Data",
        batch_size=batch_size
    )
    
    if save_results:
        # Save data information
        data_info = {
            'n_nights': int(len(torch.unique(data_manager.data['night_idx']))),
            'n_samples': int(len(data_manager.data['y'])),
            'class_distribution': convert_to_serializable(
                Counter(data_manager.data['y'].numpy()).most_common()
            ),
            'data_files': new_data_files,
            'timestamp': timestamp
        }
        
        info_path = os.path.join(eval_dir, 'evaluation_info.json')
        with open(info_path, 'w') as f:
            json.dump(data_info, f, indent=4)
        
        # Save evaluation metrics
        if isinstance(evaluation_results.get('metrics'), pd.DataFrame):
            metrics_csv_path = os.path.join(eval_dir, 'evaluation_metrics.csv')
            evaluation_results['metrics'].to_csv(metrics_csv_path)
            evaluation_results['metrics'] = evaluation_results['metrics'].to_dict(orient='records')
        
        # Save other evaluation results
        eval_results = {
            'predictions': evaluation_results['predictions'].tolist(),
            'true_labels': evaluation_results['true_labels'].tolist(),
            'confusion_matrix_absolute': evaluation_results['confusion_matrix_absolute'].tolist(),
            'confusion_matrix_percentage': evaluation_results['confusion_matrix_percentage'].tolist()
        }
        
        eval_path = os.path.join(eval_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=4)
    
    # Log evaluation summary
    logging.info("\nModel Performance on New Data:")
    logging.info(f"Total samples evaluated: {len(data_manager.data['y'])}")
    logging.info(f"Number of nights: {len(torch.unique(data_manager.data['night_idx']))}")
    logging.info("\nClass Distribution:")
    logging.info(format_class_distribution(Counter(data_manager.data['y'].numpy())))
    
    return eval_dir, evaluation_results

# Example usage
new_data_files = DATA_FILES[60:76]
# Evaluate using a previously trained model
eval_dir, results = evaluate_new_data(
    model_dir='./models/new7/',
    new_data_files=new_data_files,
    batch_size=32,
    save_results=False
)




