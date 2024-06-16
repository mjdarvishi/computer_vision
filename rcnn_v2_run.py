import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset class (you can reuse your existing COCODataset)
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
            image = ToTensor()(image)  # Ensure image is a tensor

        # Prepare the target
        target = {
            'image_id': torch.tensor(image_id),
            'boxes': [],
            'labels': []
        }

        for ann in self.annotations['annotations']:
            if ann['image_id'] == image_id:
                # COCO format provides [x_min, y_min, width, height]
                # Convert to [x_min, y_min, x_max, y_max]
                bbox = ann['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                target['boxes'].append(bbox)
                target['labels'].append(ann['category_id'])

        target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)

        return image, target

# Data loading and transformations
data_dir = 'coco_content/ships-in-aerial-images/ships-aerial-images' # Update with your data directory
train_data_path = os.path.join(data_dir, 'coco_train/annotations.json')
valid_data_path = os.path.join(data_dir, 'coco_valid/annotations.json')

# Define transformations
data_transform = Compose([
    Resize((256, 256)),  # Resize images for faster training
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Create datasets
train_dataset = COCODataset(data_dir, train_data_path, transform=data_transform)
valid_dataset = COCODataset(data_dir, valid_data_path, transform=data_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)

# Model setup
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one for your single class
num_classes = 2  # 1 class (ship) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 1  # Adjust as needed

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {losses.item()}')

    # Validation (add validation logic if needed)

# Save the trained model
torch.save(model.state_dict(), 'faster_rcnn_ship_model.pth')

# Visualization (example)
def visualize_predictions(image, target, prediction):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  # Assuming image is in CHW format

    # Draw ground truth boxes
    for box in target['boxes']:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Draw predicted boxes (if available)
    if prediction is not None:
        for box in prediction['boxes']:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

    plt.show()

# Example usage for visualization
model.eval()
with torch.no_grad():
    for images, targets in valid_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        # Visualize the first image in the batch
        visualize_predictions(images[0].cpu(), targets[0], predictions[0])
        break
