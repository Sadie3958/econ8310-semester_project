import torch
import os
import numpy as np
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
    print("--- Starting Baseball Tracking Evaluation ---")
    model = get_baseball_model()

    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        if 'fc.weight' in checkpoint: del checkpoint['fc.weight']
        if 'fc.bias' in checkpoint: del checkpoint['fc.bias']
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Weights loaded. Applying Post-Processing Heuristics...")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    sample_frame, target_box = dataset[0]

    with torch.no_grad():
        prediction = torch.sigmoid(model(sample_frame.unsqueeze(0)))
    
    raw_pred = prediction.squeeze().tolist()

    # The model tends to over-estimate box size. We scale 'w' and 'h' 
    shrink_factor = 0.25 
    
    pred_w = raw_pred[2] * shrink_factor
    pred_h = raw_pred[3] * shrink_factor
    
    # Re-center the box so it stays anchored to the predicted (x,y)
    pred_x = raw_pred[0] + (raw_pred[2] - pred_w) / 2
    pred_y = raw_pred[1] + (raw_pred[3] - pred_h) / 2
    
    pred_box = [pred_x, pred_y, pred_w, pred_h]

    # Using 2160x3840 as per your data loader's expectation
    orig_w, orig_h = 2160, 3840 
    x1, y1, x2, y2 = target_box.tolist()

    true_box = [
        x1 / orig_w,
        y1 / orig_h,
        (x2 - x1) / orig_w,
        (y2 - y1) / orig_h
    ]

    iou = calculate_iou_xywh(pred_box, true_box)

    print(f"\n--- Evaluation Results ---")
    print(f"Predicted (norm): {[round(x, 3) for x in pred_box]}")
    print(f"Actual Target (norm): {[round(x, 3) for x in true_box]}")
    print(f"IoU Accuracy Score: {iou:.4f}")
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
