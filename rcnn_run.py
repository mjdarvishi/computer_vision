import os
import zipfile
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import json
from PIL import Image


def load_data_set():
    dataset_dir = 'ships-in-aerial-images'
    dataset_zip = 'ships-in-aerial-images.zip'
    # Check if the dataset zip file exists
    if not os.path.exists(dataset_zip):
        # Download the Kaggle dataset
        os.system('kaggle datasets download siddharthkumarsah/ships-in-aerial-images')
        print(f"Dataset {dataset_zip} downloaded.")
    else:
        print(f"Dataset {dataset_zip} already exists.")

    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        # Unzip the dataset
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            print(f"Dataset unzipped into directory {dataset_dir}.")
    else:
        print(f"Dataset directory {dataset_dir} already exists.")


def create_coco_dir():
    # Define the directories to be created
    directories = [
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_train',
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_test',
        'coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid'
    ]

    # Check and create each directory if it does not exist
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
    categories = [{"id": 1, "name": "ship"}]
    ann_id = 1
    img_id = 0

    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
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
            if len(ann.split()) != 5:
                continue
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

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f)


class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load COCO annotations
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
        
        return image, image_id
    

#dictionaries to store metadata and dataset instances.
class MetadataCatalog:
    _metadata = {}

    @staticmethod
    def get(name):
        return MetadataCatalog._metadata.get(name, {})

    @staticmethod
    def register(name, metadata):
        MetadataCatalog._metadata[name] = metadata

class DatasetCatalog:
    _datasets = {}

    @staticmethod
    def get(name):
        return DatasetCatalog._datasets.get(name, [])

    @staticmethod
    def register(name, dataset):
        DatasetCatalog._datasets[name] = dataset


# laod data set
load_data_set()
create_coco_dir()
train_input_dir = "ships-in-aerial-images/ships-aerial-images/train"
train_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_train"

test_input_dir = "ships-in-aerial-images/ships-aerial-images/test"
test_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_test"

valid_input_dir = "ships-in-aerial-images/ships-aerial-images/valid"
valid_output_dir = "coco_content/ships-in-aerial-images/ships-aerial-images/coco_valid"

# convert to coco 
convert_2_COCO(train_input_dir, train_output_dir)
convert_2_COCO(test_input_dir, test_output_dir)
convert_2_COCO(valid_input_dir, valid_output_dir)

# load coco data
train_dataset = COCODataset(train_input_dir, train_output_dir+'/'+'annotations.json')
test_dataset = COCODataset(test_input_dir, test_output_dir+'/'+'annotations.json')
valid_dataset = COCODataset(valid_input_dir, valid_output_dir+'/'+'annotations.json')


# Register metadata
# Define metadata for the training dataset, specifying the classes present in the dataset
metadata = {"thing_classes": ["ship"]} 

# Register the metadata with a unique identifier for the training dataset
MetadataCatalog.register("trainDataset_v3", metadata)

# Register the training dataset itself with the same unique identifier
DatasetCatalog.register("trainDataset_v3", train_dataset)

# Retrieve registered metadata for the training dataset
trainMetadata = MetadataCatalog.get("trainDataset_v3")

# Retrieve registered training dataset using the same identifier
trainDicts = DatasetCatalog.get("trainDataset_v3")
for i in range(5):
    image, image_id= trainDicts[i]
    print("Image ID:", image_id,image)



import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.transforms import Compose, Resize, ToTensor
import torch

# Define backbone for the Faster R-CNN model
backbone = resnet_fpn_backbone('resnet50', pretrained=True)

# Define anchor generator for the Faster R-CNN model
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# Define the Faster R-CNN model
model = FasterRCNN(backbone, num_classes=1, rpn_anchor_generator=anchor_generator)

# Define the optimizer
optimizer = SGD(model.parameters(), lr=0.001)

# Define data loaders (assuming you have train_loader and test_loader already defined)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define the number of warmup iterations
warmup_iters = 600

# Define the maximum number of iterations
max_iters = 800

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.05)

# Define the evaluation period
test_eval_period = 500


import random
import cv2
import matplotlib.pyplot as plt

# Convert trainDicts to a list of dictionaries where each dictionary contains image and image_id
train_dicts = [{"file_name": image_id, "image": image} for image, image_id in trainDicts]

# Randomly sample 3 elements from the train_dicts list
sampled_dicts = random.sample(train_dicts, 3)

# Visualize sampled images
for d in sampled_dicts:
    img = cv2.imread(d["image"])
    img_rgb = cv2.cvtColor(d["image"], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
