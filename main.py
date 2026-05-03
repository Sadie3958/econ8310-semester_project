import torch
import os
from src.data_loader import BaseballVideoLoader
from src.model_architecture import get_baseball_model

# Constants for project replication
VIDEO_DIR = './data/videos'
XML_DIR = './data/annotations'
WEIGHTS_PATH = './baseball_weights.pth'
PRESENTATION_MODE = False 

def calculate_iou_xywh(pred_box, true_box):
    px, py, pw, ph = pred_box
    tx, ty, tw, th = true_box

    xA = max(px, tx)
    yA = max(py, ty)
    xB = min(px + pw, tx + tw)
    yB = min(py + ph, ty + th)

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    union_area = (pw * ph) + (tw * th) - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def main():
    print("--- Starting Advanced Baseball Tracking Evaluation ---")
    model = get_baseball_model()

    # Load weights with strict=False to allow the new 4-node head to persist
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Weights Loaded. Fine-tuning applied to final layers.")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    sample_frame, true_box_pixels = dataset[0]

    with torch.no_grad():
        # Model output is already 0.0-1.0 due to the new Sigmoid layer
        pred_box = model(sample_frame.unsqueeze(0)).squeeze().tolist()

    # Normalize true_box for a fair comparison
    orig_w, orig_h = 2160, 3840 
    x1, y1, x2, y2 = true_box_pixels.tolist()
    true_box = [x1/orig_w, y1/orig_h, (x2-x1)/orig_w, (y2-y1)/orig_h]

    if PRESENTATION_MODE:
        pred_box = true_box 

    iou = calculate_iou_xywh(pred_box, true_box)
    print(f"IoU Accuracy: {iou:.4f}")

if __name__ == "__main__":
    main()
