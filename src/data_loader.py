import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np

class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, xml_dir, transform=None):
        self.video_dir = video_dir
        self.xml_dir = xml_dir
        self.transform = transform
        # Get list of all video files and strip extensions to match with XMLs
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov'))])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        xml_path = os.path.join(self.xml_dir, self.video_files[idx].rsplit('.', 1)[0] + '.xml')

        # 1. Load the first frame of the video
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if not success or not os.path.exists(xml_path):
            return None # This triggers the error you saw; let's ensure paths are correct

        # 2. Parse the XML for the target box
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extracting coordinates (assumes standard VOC format)
        # Using the 2160x3840 resolution identified in our Project Diagnosis
        bndbox = root.find('.//bndbox')
        x1 = float(bndbox.find('xmin').text)
        y1 = float(bndbox.find('ymin').text)
        x2 = float(bndbox.find('xmax').text)
        y2 = float(bndbox.find('ymax').text)

        # 3. Process Frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)) # ResNet standard input size
        frame = frame.transpose((2, 0, 1)) # HWC to CHW
        frame_tensor = torch.from_numpy(frame).float() / 255.0

        # 4. Return as a Tensor
        target_box = torch.tensor([x1, y1, x2, y2])
        
        return frame_tensor, target_box
