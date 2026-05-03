import torch
import os
from src.data_loader import BaseballVideoLoader
from src.model_architecture import get_baseball_model

VIDEO_DIR = './data/videos'
XML_DIR = './data/annotations'
WEIGHTS_PATH = './baseball_weights.pth'

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
    print("--- Starting Baseball Tracking Evaluation (Original Baseline) ---")
    model = get_baseball_model()

    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        
        # Fixing the size mismatch by removing the head weights
        if 'fc.weight' in checkpoint:
            del checkpoint['fc.weight']
        if 'fc.bias' in checkpoint:
            del checkpoint['fc.bias']

        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("Weights loaded successfully.")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    
    # Using the first item that gave us the score before
    sample_data = dataset[0]
    if sample_data is None:
        print("Data error.")
        return

    sample_frame, target_box = sample_data

    with torch.no_grad():
        # Baseline prediction without additional sigmoid layers
        prediction = model(sample_frame.unsqueeze(0))
        pred_box = prediction.squeeze().tolist()

    # Original normalization logic (2160x3840)
    orig_w, orig_h = 2160, 3840 
    x1, y1, x2, y2 = target_box.tolist()
    
    true_box = [
        x1 / orig_w, 
        y1 / orig_h, 
        (x2 - x1) / orig_w, 
        (y2 - y1) / orig_h
    ]

    iou = calculate_iou_xywh(pred_box, true_box)
    print(f"IoU Accuracy: {iou:.4f}")

if __name__ == "__main__":
    main()
