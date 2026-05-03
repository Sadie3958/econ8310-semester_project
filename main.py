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

    # Intersection coordinates
    xA = max(px, tx)
    yA = max(py, ty)
    xB = min(px + pw, tx + tw)
    yB = min(py + ph, ty + th)

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    pred_area = max(0, pw) * max(0, ph)
    true_area = max(0, tw) * max(0, th)
    union_area = pred_area + true_area - inter_area

    if union_area <= 0:
        return 0

    return inter_area / union_area

def main():
    print("--- Starting Baseball Tracking Evaluation ---")

    # 1. Initialize model
    model = get_baseball_model()

    # 2. Handle Weight Loading (Technical Pivot: size mismatch fix)
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))

        # Remove old classification head weights to allow new 4-node head
        if 'fc.weight' in checkpoint:
            del checkpoint['fc.weight']
        if 'fc.bias' in checkpoint:
            del checkpoint['fc.bias']

        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Successfully loaded feature weights. Adjusted for 4-coordinate regression.")

    # 3. Load Dataset
    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)

    if len(dataset) == 0:
        print("ERROR: No videos found. Check folder paths.")
        return

    # 4. Run Prediction on sample
    sample_frame, target_box = dataset[0]

    with torch.no_grad():
        # Use sigmoid to keep prediction between 0.0 and 1.0
        prediction = torch.sigmoid(model(sample_frame.unsqueeze(0)))

    pred_box = prediction.squeeze().tolist()

    # 5. Coordinate Normalization (Diagnosis of Data)
    # Corrected for 4K Landscape Orientation
    orig_w = 3840
    orig_h = 2160

    x1, y1, x2, y2 = target_box.tolist()

    # Convert CVAT xtl, ytl, xbr, ybr -> Normalized x, y, w, h
    true_box = [
        x1 / orig_w,
        y1 / orig_h,
        (x2 - x1) / orig_w,
        (y2 - y1) / orig_h
    ]

    # 6. Final Results
    iou = calculate_iou_xywh(pred_box, true_box)

    print(f"\n--- Evaluation Results ---")
    print(f"Predicted (norm): {[round(x, 3) for x in pred_box]}")
    print(f"Actual Target (norm): {[round(x, 3) for x in true_box]}")
    print(f"IoU score: {iou:.4f}")
    print("\nEvaluation Complete. Ready for coaching diagnostics.")

# FIXED: Corrected underscores for execution
if __name__ == "__main__":
    main()
