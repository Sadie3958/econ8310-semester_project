import torch
import os
from src.data_loader import BaseballVideoLoader
from src.model_architecture import get_baseball_model

VIDEO_DIR = './data/videos'
XML_DIR = './data/annotations'
WEIGHTS_PATH = './baseball_weights.pth'

def calculate_iou_xywh(pred_box, true_box):
    # Ensure we are comparing (x, y, w, h)
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
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        # Use strict=False to maintain your custom 4-node output head
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("Model Weights Loaded Successfully.")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    
    # Iterate until we find a valid sample with a label
    sample_data = None
    for i in range(len(dataset)):
        sample_data = dataset[i]
        if sample_data is not None:
            break

    if sample_data is None:
        print("ERROR: No valid labels found in XML files.")
        return

    sample_frame, true_box_pixels = sample_data

    with torch.no_grad():
        # Predict normalized coordinates (0.0 - 1.0)
        prediction = model(sample_frame.unsqueeze(0))
        pred_box = torch.sigmoid(prediction).squeeze().tolist()

    # Apply 4K resolution normalization (Diagnosis of Data)
    orig_w, orig_h = 2160, 3840 
    x1, y1, x2, y2 = true_box_pixels.tolist()
    
    true_box = [
        x1 / orig_w, 
        y1 / orig_h, 
        (x2 - x1) / orig_w, 
        (y2 - y1) / orig_h
    ]

    iou = calculate_iou_xywh(pred_box, true_box)
    
    print(f"\n--- Final Analysis ---")
    print(f"Predicted Box: {[round(x, 3) for x in pred_box]}")
    print(f"True Box: {[round(x, 3) for x in true_box]}")
    print(f"IoU Accuracy: {iou:.4f}")

if __name__ == "__main__":
    main()
