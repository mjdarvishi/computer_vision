import os
import zipfile
from torch.utils.data import Dataset
import json
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, Resize, ToTensor
import torch
import shutil
import subprocess
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    model = YOLO('yolov8m.pt')

    # yolov8_ships2
    results = model.train(data="ships.yaml", epochs = 8, imgsz = 768, seed = 42, batch = 8, workers = 4)

    #yolov8_ships2
    # subprocess.run([
    #     'yolo', 'train',
    #     'data=ships.yaml',  # Data file
    #     'model=yolov8n.pt',  # Model configuration
    #     'epochs=1',  # Number of epochs
    #     'imgsz=640',  # Image size
    #     'batch=16',  # Batch size
    #     'name=yolov8_ships'  # Name of the run
    #     ])