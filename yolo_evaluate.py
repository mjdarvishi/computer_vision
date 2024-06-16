

# model one

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


# img = mpimg.imread('runs/detect/yolov8_ships/confusion_matrix.png')

# # Plotting the confusion matrix image
# fig, ax = plt.subplots(figsize = (15, 15))

# ax.imshow(img)
# ax.axis('off')


# model two


# from IPython.display import Image
# from IPython.display import display

# x = Image(filename='runs/detect/train2/F1_curve.png')
# y = Image(filename='runs/detect/train2/PR_curve.png')
# z = Image(filename='runs/detect/train2/confusion_matrix.png')
# display(x, y,z)