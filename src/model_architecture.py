import torch
import torch.nn as nn
from torchvision import models

def get_baseball_model():
    # Load the standard ResNet-18
    model = models.resnet18(weights=None) 
    
    # Simple linear head to match the original 4-coordinate output
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    return model
