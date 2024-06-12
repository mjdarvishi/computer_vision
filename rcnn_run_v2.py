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

def load_data_set():
    dataset_dir = 'ships-in-aerial-images'
    dataset_zip = 'ships-in-aerial-images.zip'
    if not os.path.exists(dataset_zip):
        os.system('kaggle datasets download siddharthkumarsah/ships-in-aerial-images')
        print(f"Dataset {dataset_zip} downloaded.")
    else:
        print(f"Dataset {dataset_zip} already exists.")

    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            print(f"Dataset unzipped into directory {dataset_dir}.")
    else:
        print(f"Dataset directory {dataset_dir} already exists.")


def create_coco_dir():
    directories = [
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_train',
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_test',
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid'
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")
        else:
            print(f"Directory {directory} already exists.")


def convert_2_COCO(input_dir, output_dir):
    annotations_path = os.path.join(output_dir, 'annotations.json')
    if os.path.exists(annotations_path):
        print("Conversion has already been done. Exiting...")
        return
    categories = [{"id" : 1, "name" : "ship"}]
    ann_id = 1
    img_id = 0

    coco_dataset = {
        "info" : {},
        "licenses" : [],
        "categories" : categories,
        "images" : [],
        "annotations" : []
    }

    images_path = os.path.join(input_dir, "images")
    labels_path = os.path.join(input_dir, "labels")

    for image_file in os.listdir(images_path):
        # Getting Dimensions
        image_path = os.path.join(images_path, image_file)
        image = Image.open(image_path)
        width, height = image.size

        # Add the image to the COCO dataset
        image_dict = {
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": image_path
        }
        coco_dataset["images"].append(image_dict)

        # Load the bounding box annotations for the image
        with open(os.path.join(labels_path, f'{image_file[:-4]}.txt'), 'r') as f:
            annotations = f.read().strip().split("\n")

        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            if len(ann.split()) != 5: continue
            _, x, y, w, h = map(float, ann.split())
            x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
            x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
            ann_dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)
            ann_id += 1
        img_id += 1

    print("Conversion Complete!")
    print("Final Image Count : {}".format(len(coco_dataset["images"])))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f)

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
    def __len__(self):
        return len(self.annotations['images'])
    

    def __getitem__(self, idx):
        image_info = self.annotations['images'][idx]
        image_id = image_info['id']
        file_name = image_info['file_name']
        image = Image.open(file_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)  # Ensure image is a tensor even if no transform is provided
        
        # Prepare the target
        target = {
            'image_id': torch.tensor(image_id),
            'boxes': [],
            'labels': []
        }
        
        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                target['boxes'].append(ann['bbox'])
                target['labels'].append(ann['category_id'])

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        
        return image, target
    
def get_faster_rcnn_model(num_classes):
    # Load a pre-trained model for classification and return only the features
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    
    # Faster R-CNN model
    model = FasterRCNN(backbone, num_classes=num_classes)
    
    return model
if __name__ == "__main__":
    load_data_set()
    create_coco_dir()
    train_input_dir = "ships-in-aerial-images/ships-aerial-images/train"
    train_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_train"
    test_input_dir = "ships-in-aerial-images/ships-aerial-images/test"
    test_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_test"
    valid_input_dir = "ships-in-aerial-images/ships-aerial-images/valid"
    valid_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid"

    convert_2_COCO(train_input_dir, train_output_dir)
    convert_2_COCO(test_input_dir, test_output_dir)
    convert_2_COCO(valid_input_dir, valid_output_dir)
    transform = None
    # train_dataset = COCODataset(train_input_dir, train_output_dir+'/'+'annotations.json',transform=transform)
    # test_dataset = COCODataset(test_input_dir, test_output_dir+'/'+'annotations.json',transform=transform)
    # valid_dataset = COCODataset(valid_input_dir, valid_output_dir+'/'+'annotations.json',transform=transform)

    # Define data transformations
    transform = Compose([Resize((800, 800)), ToTensor()])

    # Initialize the datasets and data loaders
    train_dataset = COCODataset(root_dir='coco_content/ships-in-aerial-images/ships-aerial-images/coco_train', 
                                annotation_file='coco_content/ships-in-aerial-images/ships-aerial-images/coco_train/annotations.json', 
                                transform=transform)
    valid_dataset = COCODataset(root_dir='coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid', 
                                annotation_file='coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid/annotations.json', 
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
# Helper function for collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the model
model = get_faster_rcnn_model(num_classes=2)  # 2 classes (background and ship)
model.to(device)

# Define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Sum the losses
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()
        
        # Backward pass and optimization
        losses.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}')
    
    # Validation loop
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for images, targets in valid_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            valid_loss += losses.item()
    
    valid_loss /= len(valid_loader)
    print(f'Validation Loss: {valid_loss}')