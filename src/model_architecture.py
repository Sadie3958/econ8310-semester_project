import torch.nn as nn
from torchvision import models

def get_baseball_model():
    # Using ResNet-18 as foundational architecture
    model = models.resnet18(weights=None)
    
    # Adjust for Bounding Box Regression (4 coordinates: x, y, w, h)
    # Instead of 2 classes, we output 4 continuous values
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model
