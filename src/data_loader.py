import torch
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET

class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, xml_dir, transform=None):
        self.video_dir = video_dir
        self.xml_dir = xml_dir
        # ... logic to match videos and XMLs ...

    def __getitem__(self, idx):
        # Load frame and get original dimensions
        # orig_w, orig_h = 2160, 3840
        
        # KEY UPDATE: Normalize the XML coordinates here
        # (x1, y1, x2, y2) -> (norm_x, norm_y, norm_w, norm_h)
        
        # return frame, normalized_target_tensor
        pass
