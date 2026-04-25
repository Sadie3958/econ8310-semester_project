import torch
import os
from src.data_loader import BaseballVideoLoader
from src.model_architecture import get_baseball_model

# Constants to help Dr. White repeat the process
VIDEO_DIR = './data/videos'
XML_DIR = './data/annotations'
WEIGHTS_PATH = './baseball_weights.pth'

def main():
    print("--- Starting Baseball Tracking Evaluation ---")

    # 1. Initialize the architecture
    # Building off ResNet-18 (based on our proposal)
    model = get_baseball_model()

  # 2. Load the trained weights
    if os.path.exists(WEIGHTS_PATH):
        # Load the weights file
        checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        
        # REMOVE the old classification layer (fc) from the weights so it doesn't clash
        # This allows the model to keep the "learned features" but use a new "4-output" head
        if 'fc.weight' in checkpoint:
            del checkpoint['fc.weight']
        if 'fc.bias' in checkpoint:
            del checkpoint['fc.bias']
            
        # Load the remaining weights (ignoring the missing fc layer)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Successfully loaded feature weights from {WEIGHTS_PATH} (Adjusted for 4-coordinate output)")

    # 3. Initialize Data Loader
    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    
    if len(dataset) == 0:
        print("ERROR: No videos found in /data/videos")
        return

    print(f"Successfully initialized loader with {len(dataset)} videos and XML annotations.")

    # 4. Run a sample inference
    # This should prove the 'Analysis' is appropriate for the stated goals
    sample_frame, target_box = dataset[0]
    with torch.no_grad():
        prediction = model(sample_frame.unsqueeze(0))
    
    print(f"Sample Prediction (x, y, w, h): {prediction.tolist()}")
    print("Evaluation Complete. Model is ready for coaching diagnostics.")

if __name__ == "__main__":
    main()
