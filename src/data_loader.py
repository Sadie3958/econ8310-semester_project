import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET

class BaseballVideoLoader(Dataset):
    def __init__(self, video_dir, xml_dir, transform=None):
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

        # Parse XML with safety checks
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Look for the first object/bndbox found in the file
        bndbox = root.find('.//bndbox')
        
        # If no bndbox is found, search for any tag containing 'xmin' (CVAT compatibility)
        if bndbox is None:
            xmin_tag = root.find('.//xmin')
            if xmin_tag is None:
                return None # No label found in this file
            
            # Manual extraction if bndbox parent is missing
            x1 = float(root.find('.//xmin').text)
            y1 = float(root.find('.//ymin').text)
            x2 = float(root.find('.//xmax').text)
            y2 = float(root.find('.//ymax').text)
        else:
            x1 = float(bndbox.find('xmin').text)
            y1 = float(bndbox.find('ymin').text)
            x2 = float(bndbox.find('xmax').text)
            y2 = float(bndbox.find('ymax').text)

        # Process Image for ResNet-18
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.transpose((2, 0, 1))
        frame_tensor = torch.from_numpy(frame).float() / 255.0

        target_box = torch.tensor([x1, y1, x2, y2])
        return frame_tensor, target_box
