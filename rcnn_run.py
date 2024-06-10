import os
import zipfile
from torch.utils.data import Dataset
import json
from PIL import Image
import random
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
import numpy as np

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
                if any(x <= 0 for x in ann['bbox'][2:]):  # Check for negative width or height
                    print("Invalid bounding box:", ann['bbox'], "in image:", file_name,'size of image',image_info['width'],image_info['height'])
                    # continue
                target['boxes'].append(ann['bbox'])
                target['labels'].append(ann['category_id'])

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        
        return image, target
    


def train_model(model, train_loader, optimizer, scheduler, num_epochs, eval_period):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")

        if (epoch + 1) % eval_period == 0:
            model.eval()
            with torch.no_grad():
                # evaluate(model, test_loader, device)
                #TODO: add evaluation
                pass
    return model

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    print(targets)
    return list(images), list(targets)

def show_images_with_boxes(dataset, num_images=10):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    random_indices = random.sample(range(len(dataset)), num_images)
    
    for i, idx in enumerate(random_indices):
        image, target = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)  # Clip values to range [0, 1]

        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        axes[i].imshow(image)
        axes[i].axis('off')

        for box, label in zip(boxes, labels):
            x, y, w, h = box
            category = dataset.annotations['categories'][label - 1]['name']  # Subtract 1 since COCO class indices start from 1
            axes[i].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none'))
            axes[i].text(x, y, s=category, color='yellow', verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
def show_first_10_images_with_boxes(dataset):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i in range(10):
        image, target = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)  # Clip values to range [0, 1]

        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        axes[i].imshow(image)
        axes[i].axis('off')

        for box, label in zip(boxes, labels):
            x, y, w, h = box
            category = dataset.annotations['categories'][label - 1]['name']  # Subtract 1 since COCO class indices start from 1
            axes[i].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none'))
            axes[i].text(x, y, s=category, color='yellow', verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
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

    train_dataset = COCODataset(train_input_dir, train_output_dir+'/'+'annotations.json')
    test_dataset = COCODataset(test_input_dir, test_output_dir+'/'+'annotations.json')
    valid_dataset = COCODataset(valid_input_dir, valid_output_dir+'/'+'annotations.json')
    # show_images_with_boxes(train_dataset)

    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone, num_classes=1, rpn_anchor_generator=anchor_generator)

    optimizer = SGD(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, shuffle=False, collate_fn=custom_collate_fn,num_workers=2,batch_size=5)
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=custom_collate_fn,num_workers=2,batch_size=5)

    model_save_path = "faster_rcnn_model.pth"
    num_epochs = 20
    eval_period = 5

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.05)

    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.eval()
        print("Model loaded successfully.")
    else:
        model = train_model(model, train_loader, optimizer, scheduler, num_epochs, eval_period)

        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, model_save_path)
        print("Training complete and model saved successfully.")

    num_epochs = 20
    eval_period = 5

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader)}")
