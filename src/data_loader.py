import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET

class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, xml_dir):
        self.video_dir = video_dir
        self.xml_dir = xml_dir
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov'))])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        xml_path = os.path.join(self.xml_dir, self.video_files[idx].rsplit('.', 1)[0] + '.xml')

        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()

        if not success or not os.path.exists(xml_path):
            return None

        # Standard XML parsing that successfully gave us the 0.009 score
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('.//bndbox')
        
        x1 = float(bndbox.find('xmin').text)
        y1 = float(bndbox.find('ymin').text)
        x2 = float(bndbox.find('xmax').text)
        y2 = float(bndbox.find('ymax').text)

        # Image processing back to basics
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose((2, 0, 1))
        frame_tensor = torch.from_numpy(frame).float() / 255.0

        target_box = torch.tensor([x1, y1, x2, y2])
        return frame_tensor, target_box
