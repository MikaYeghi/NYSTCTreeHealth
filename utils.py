import os
import torch
from model_zoo import MLP

def load_model(models_dir: str, model_name: str, input_dim, device='cpu'):
    # Initialize an empty model
    model = MLP(input_dim=input_dim).to(device)

    # Load the weights
    if model_name:
        if model_name.split('.')[-1] == "pth":
            ckpt_path = os.path.join(models_dir, model_name)
        else:
            ckpt_path = os.path.join(models_dir, model_name + ".pth")
    else:
        ckpt_path = os.path.join(models_dir, "last.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    # Load the checkpoint
    ckpt = torch.load(ckpt_path)

    # Assign the weights to the model
    model.load_state_dict(ckpt['model_state_dict'])

    return model