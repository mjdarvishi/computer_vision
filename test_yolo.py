from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns


model = YOLO('runs/detect/yolov8_ships2/weights/best.pt')

# # Evaluating the model on the test dataset
# metrics = model.val(conf = 0.25, split = 'test')

# # Create the barplot
# ax = sns.barplot(x=['mAP50-95', 'mAP50', 'mAP75'], y=[metrics.box.map, metrics.box.map50, metrics.box.map75])

# # Set the title and axis labels
# ax.set_title('YOLO Evaluation Metrics')
# ax.set_xlabel('Metric')
# ax.set_ylabel('Value')

# # Set the figure size
# fig = plt.gcf()
# fig.set_size_inches(8, 6)

# # Add the values on top of the bars
# for p in ax.patches:
#     ax.annotate('{:.3f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    
# # Show the plot
# plt.show()


# img = mpimg.imread('runs/detect/train/confusion_matrix.png')

# # Plotting the confusion matrix image
# fig, ax = plt.subplots(figsize = (15, 15))

# ax.imshow(img)
# ax.axis('off')

def ship_detect(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Pass the image through the detection model and get the result
    detect_result = model(img)

    # Plot the detections
    detect_img = detect_result[0].plot()

    # Convert the image to RGB format
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)

    return detect_img

# Define the directory where the custom images are stored
custom_image_dir = 'ships-in-aerial-images/ships-aerial-images/test/images'

# Get the list of image files in the directory
image_files = os.listdir(custom_image_dir)

# Select 16 random images from the list
selected_images = random.sample(image_files, 16)

# Create a figure with subplots for each image
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

# Iterate over the selected images and plot each one
for i, img_file in enumerate(selected_images):
    # Compute the row and column index of the current subplot
    row_idx = i // 4
    col_idx = i % 4

    # Load the current image and run object detection
    img_path = os.path.join(custom_image_dir, img_file)
    detect_img = ship_detect(img_path)

    # Plot the current image on the appropriate subplot
    axes[row_idx, col_idx].imshow(detect_img)
    axes[row_idx, col_idx].axis('off')

# Adjust the spacing between the subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Show the plot
plt.show()