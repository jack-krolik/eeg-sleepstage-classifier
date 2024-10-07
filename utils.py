import os
import json
import torch
from config import CONFIG

def save_params(params, filename):
    filepath = os.path.join(CONFIG['model_path'], filename)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

def load_params(filename):
    filepath = os.path.join(CONFIG['model_path'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_model(model, filename):
    filepath = os.path.join(CONFIG['model_path'], filename)
    torch.save(model.state_dict(), filepath)

def load_model(model, filename):
    filepath = os.path.join(CONFIG['model_path'], filename)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
    return model

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)