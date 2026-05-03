import torch
import torch.nn as nn
from torchvision import models

def get_baseball_model():
    # Load pre-trained ResNet-18 to leverage existing "shape" knowledge
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Technical Pivot: Replace the classification head with a Regression Head
    # We output 4 nodes: x, y, width, height
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.2), # Prevents overfitting to background noise
        nn.Linear(128, 4),
        nn.Sigmoid() # Force output to 0.0-1.0 range for easier IoU math
    )
    return model
