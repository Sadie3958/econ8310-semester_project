import torch
import os
from src.data_loader import BaseballVideoLoader
from src.model_architecture import get_baseball_model

# Constants
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
    print("--- Starting Advanced Baseball Tracking Evaluation ---")
    model = get_baseball_model()

    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print(f"Weights Loaded. Analysis ready.")

    dataset = BaseballVideoLoader(VIDEO_DIR, XML_DIR)
    
    # Check if dataset is empty before accessing
    if len(dataset) == 0:
        print("ERROR: Check your folder paths. No data found.")
        return

    # Unpack the frame and box
    data = dataset[0]
    if data is None:
        print("ERROR: Failed to load sample at index 0. Check XML names match Video names.")
        return
        
    sample_frame, true_box_pixels = data

    with torch.no_grad():
        # Get prediction (0.0 to 1.0)
        pred_box = model(sample_frame.unsqueeze(0)).squeeze().tolist()

    # Normalize true_box (2160x3840) to match model output
    orig_w, orig_h = 2160, 3840 
    x1, y1, x2, y2 = true_box_pixels.tolist()
    true_box = [x1/orig_w, y1/orig_h, (x2-x1)/orig_w, (y2-y1)/orig_h]

    iou = calculate_iou_xywh(pred_box, true_box)
    print(f"\n--- Results ---")
    print(f"IoU Accuracy: {iou:.4f}")
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
