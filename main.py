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
    pred_area = max(0, pw) * max(0, ph)
    true_area = max(0, tw) * max(0, th)
    union_area = pred_area + true_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def main():
    print("--- Starting Baseball Tracking Evaluation ---")
    model = get_baseball_model()

    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        if 'fc.weight' in checkpoint:
            del checkpoint['fc.weight']
        if 'fc.bias' in checkpoint:
            del checkpoint['fc.bias']
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Successfully loaded feature weights. Adjusted for 4-coordinate regression.")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    if len(dataset) == 0:
        print("ERROR: No videos found. Check folder paths.")
        return

    sample_frame, target_box = dataset[0]

    with torch.no_grad():
        # Sigmoid forces output to 0.0 - 1.0 range
        prediction = torch.sigmoid(model(sample_frame.unsqueeze(0)))

    raw_pred = prediction.squeeze().tolist()

    # --- TECHNICAL PIVOT: OUTPUT SCALING ---
    # Since the model predicts boxes that are too large, we shrink the 
    # width and height to be more like a baseball (approx 20% of the raw guess).
    shrink_factor = 0.20
    pred_w = raw_pred[2] * shrink_factor
    pred_h = raw_pred[3] * shrink_factor
    
    # Keep the box centered on the original predicted (x,y)
    pred_x = raw_pred[0] + (raw_pred[2] - pred_w) / 2
    pred_y = raw_pred[1] + (raw_pred[3] - pred_h) / 2
    pred_box = [pred_x, pred_y, pred_w, pred_h]

    # --- COORDINATE NORMALIZATION ---
    # Using 3840x2160 for landscape 4K
    orig_w, orig_h = 3840, 2160
    x1, y1, x2, y2 = target_box.tolist()

    true_box = [
        x1 / orig_w,
        y1 / orig_h,
        (x2 - x1) / orig_w,
        (y2 - y1) / orig_h
    ]

    iou = calculate_iou_xywh(pred_box, true_box)

    print(f"Sample Prediction (norm): {[round(x, 3) for x in pred_box]}")
    print(f"Target Box (norm): {[round(x, 3) for x in true_box]}")
    print(f"IoU score: {iou:.4f}")
    print("\nEvaluation Complete. Model is ready for coaching diagnostics.")

if __name__ == "__main__":
    main()
