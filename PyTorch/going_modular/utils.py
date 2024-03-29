""" 
Contains various utility functions for PyTorch model training and saving. 
"""
import torch 
from pathlib import Path 

def save_model(model, target_dir, model_name): 
    """Saves a PyTorchj model to a target directory. 
    
    Args: 
        model: A target PyTorch model to. 
        target_dir: A directory for saving the model to. 
        model_name; A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.  
    """
    target_dir_path = Path(target_dir) 
    target_dir_path.mkdir(parents=True, exist_ok=True) 
    
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or 'pth'"
    model_save_path = target_dir / Path(model_name) 
    
    print(f"[INfO] Saving model to: {model_save_path}") 
    torch.save(obj=model.state_dict(), f=model_save_path) 
