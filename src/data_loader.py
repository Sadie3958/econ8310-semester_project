import torch
import cv2
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, xml_dir, transform=None):
        self.video_dir = video_dir
        self.xml_dir = xml_dir
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mov')]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def parse_xml(self, xml_path):
        """Parses CVAT XML to get bounding box coordinates."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Finding the first box in the XML
        box = root.find('.//box')
        if box is not None:
            return [
                float(box.get('xtl')), 
                float(box.get('ytl')), 
                float(box.get('xbr')), 
                float(box.get('ybr'))
            ]
        return [0, 0, 0, 0]

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        xml_path = os.path.join(self.xml_dir, self.video_files[idx].replace('.mov', '.xml'))
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read() # Grabbing the first frame
        cap.release()

        # Implementation of Frame Differencing
        
        box = self.parse_xml(xml_path)
        
        # Standardize for the model
        frame = cv2.resize(frame, (224, 224))
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        return frame_tensor, torch.tensor(box)
