from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns



#yolov8_ships
model = YOLO('runs/detect/yolov8_ships2/weights/best.pt')


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




# model = YOLO("/runs/detect/yolov8_ships1/weights/best.pt")

# img_path = "/ships-in-aerial-images/ships-aerial-images/valid/images/0006c52e8_jpg.rf.3dffdbf399b44601151f36d50bc8bba2.jpg"

# img = cv2.imread(img_path)
# results = model.predict(img_path, stream=True, imgsz=768, conf=0.5)

# for result in results:
#   boxes = result.boxes.cpu().numpy()
#   for box in boxes:
#     r = box.xyxy[0].astype(int)
#     cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)

# cv2_imshow(img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()