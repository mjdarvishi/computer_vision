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
    
def convert_coco_to_yolo(coco_json_path, output_dir):
    # Check if the YOLO dataset directory already exists
    if os.path.exists(output_dir):
        print(f"YOLO dataset directory {output_dir} already exists. Skipping conversion.")
        return

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    for img_id, img_info in images.items():
        image_filename = os.path.basename(img_info['file_name'])
        yolo_label_path = os.path.join(output_dir, 'labels', image_filename.replace('.jpg', '.txt'))
        os.makedirs(os.path.dirname(yolo_label_path), exist_ok=True)

        with open(yolo_label_path, 'w') as f:
            for ann in annotations:
                if ann['image_id'] == img_id:
                    x_min, y_min, width, height = ann['bbox']
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    img_width, img_height = img_info['width'], img_info['height']
                    x_center /= img_width
                    y_center /= img_height
                    width /= img_width
                    height /= img_height
                    f.write(f"{ann['category_id'] - 1} {x_center} {y_center} {width} {height}\n")
    
    for img_info in images.values():
        image_src_path = img_info['file_name']
        image_dst_path = os.path.join(output_dir, 'images', os.path.basename(image_src_path))
        os.makedirs(os.path.dirname(image_dst_path), exist_ok=True)
        shutil.copy(image_src_path, image_dst_path)

    print(f"Conversion to YOLO format complete. Dataset saved to {output_dir}")

    
def test_yolo_model(model_path, data_yaml, test_images_dir, output_dir):
    # Load the trained model
    model = YOLO(model_path)

    # Load the data configuration
    model.data = data_yaml

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Perform inference on the test dataset
    test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpg')]
    
    for img_path in test_images:
        img = cv2.imread(img_path)
        results = model(img)

        # Plot the image with bounding boxes
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for box in results.xyxy[0]:  # xyxy format
            x_min, y_min, x_max, y_max, conf, cls = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{model.names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5), clip_box=ax.clipbox, clip_on=True)

        plt.axis('off')
        plt.tight_layout()

        # Save the plot
        output_img_path = os.path.join(output_dir, os.path.basename(img_path))
        plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print(f"Inference completed. Results saved to {output_dir}")


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
    # transform = Compose([Resize((416, 416)), ToTensor()])
    transform = None
    train_dataset = COCODataset(train_input_dir, train_output_dir+'/'+'annotations.json',transform=transform)
    test_dataset = COCODataset(test_input_dir, test_output_dir+'/'+'annotations.json',transform=transform)
    valid_dataset = COCODataset(valid_input_dir, valid_output_dir+'/'+'annotations.json',transform=transform)

    # Convert datasets
    convert_coco_to_yolo('coco_content/ships-in-aerial-images/ships-aerial-images/coco_train/annotations.json', 'yolo_dataset/train')
    convert_coco_to_yolo('coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid/annotations.json', 'yolo_dataset/val')
    convert_coco_to_yolo('coco_content/ships-in-aerial-images/ships-aerial-images/coco_test/annotations.json', 'yolo_dataset/test')

    # Training YOLOv5
    subprocess.run([
        'yolo', 'train',
        'data=ships.yaml',  # Data file
        'model=yolov8n.pt',  # Model configuration
        'epochs=50',  # Number of epochs
        'imgsz=640',  # Image size
        'batch=16',  # Batch size
        'name=yolov8_ships'  # Name of the run
        ])
