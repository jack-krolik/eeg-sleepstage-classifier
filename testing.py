import os
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.config import CONFIG, device, cuda_manager
from tools.classes import EnsembleModel, DiverseEnsembleModel, ImprovedSleepdetector, SleepStageEvaluator
from tools.functions import load_data, prepare_data_multi_night, preprocess_data, set_seed
from tools.utils import load_model, ensure_dir
from IPython.display import display, HTML
from sklearn.metrics import f1_score
import json


# Configure logging to display in notebook
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

set_seed(CONFIG['settings']['seed'])

SAVE_OUTPUTS = True
if SAVE_OUTPUTS:
    logging.info(f"Saving outputs to {CONFIG['model_dir']}")
else:
    logging.info("Not saving outputs")

def compare_models(results, save_dir):
    """Generate and display model comparison"""
    if not results:
        print("No results to compare!")
        return
        
    try:
        # Extract overall metrics for each model
        comparison_data = {
            model_name: {
                'Accuracy': (result['true_labels'] == result['predictions']).mean() * 100,
                'Macro F1': f1_score(result['true_labels'], result['predictions'], average='macro') * 100,
                'Weighted F1': f1_score(result['true_labels'], result['predictions'], average='weighted') * 100
            }
            for model_name, result in results.items()
        }
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).round(1)
        
        # Display comparison
        print("\nModel Comparison (%):")
        styled_comparison = comparison_df.style\
            .format("{:.1f}%")\
            .background_gradient(cmap='RdYlGn')\
            .set_caption("Model Performance Comparison")
        display(styled_comparison)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        comparison_df.T.plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score (%)')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and display plot
        if SAVE_OUTPUTS:
            plt.savefig(os.path.join(save_dir, 'test_results', 'model_comparison.png'), 
                        dpi=300, bbox_inches='tight')
        display(plt.gcf())
        plt.close()
        
        # Save comparison to CSV
        if SAVE_OUTPUTS:
            comparison_df.to_csv(os.path.join(save_dir, 'test_results', 'model_comparison.csv'))
        
    except Exception as e:
        print(f"Error in compare_models: {str(e)}")
        raise  # Add this to see the full error traceback

def load_and_evaluate_models():
    """Load and evaluate all models"""
    try:
        # Load test data
        print("Loading test data...")
        x, y, night_indices = load_data(CONFIG['data_paths'])
        _, _, _, _, _, _, X_test, X_test_spectral, y_test = prepare_data_multi_night(x, y, night_indices)
        X_test, X_test_spectral = preprocess_data(X_test, X_test_spectral)
        
        # Initialize evaluator
        evaluator = SleepStageEvaluator()
        evaluator.save_outputs = SAVE_OUTPUTS
        
        # Load model parameters
        try:
            params_path = os.path.join(CONFIG['model_dir'], CONFIG['model_names']['params'])
            with open(params_path, 'r') as f:
                model_params = json.load(f)['model_params']
            print("Successfully loaded model parameters")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print("Using default parameters from CONFIG")
            model_params = CONFIG['model_params']['initial']
        
        # Model name mapping
        model_file_mapping = {
            'Ensemble Model': 'ensemble',
            'Diverse Ensemble': 'diverse',
            'Distilled Model': 'distilled'
        }
        
        # Initialize and evaluate models
        results = {}
        for model_name, file_name in model_file_mapping.items():
            try:
                print(f"\nEvaluating {model_name}...")
                
                # Initialize appropriate model
                if model_name == 'Ensemble Model':
                    model = EnsembleModel(model_params)
                elif model_name == 'Diverse Ensemble':
                    model = DiverseEnsembleModel(model_params)
                else:  # Distilled Model
                    model = ImprovedSleepdetector(**model_params)
                
                # Load model weights
                model_path = os.path.join(CONFIG['model_dir'], CONFIG['model_names'][file_name])
                print(f"Loading model from: {model_path}")
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                model = load_model(model, model_path)
                model = model.to(device)
                print(f"Successfully loaded {model_name}")
                
                # Evaluate model
                results[model_name] = evaluator.evaluate_model(
                    model, X_test, X_test_spectral, y_test, model_name
                )
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        if not results:
            print("No models were successfully evaluated!")
            return None
            
        # Compare models only if we have results
        compare_models(results, evaluator.model_dir)
        
        return results
        
    except Exception as e:
        print(f"Error in load_and_evaluate_models: {str(e)}")
        return None



# Run evaluation
print("Starting model evaluation...")
results = load_and_evaluate_models()


