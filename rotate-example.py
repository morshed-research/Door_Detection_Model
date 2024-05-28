import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Function to visualize image and bounding boxes
def visualize(image, bboxes, category_ids=None):
    image = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = (255, 0, 0)  # Red color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        if category_id is not None:
            cv2.putText(image, str(category_id), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example image and bounding boxes
image_path = 'test-floorplan.png'
image = cv2.imread(image_path)
bbox = [[350, 350, 400, 400]]  # [xmin, ymin, xmax, ymax]
category_id = [1]  # Example category ids for the bounding boxes

# Display original image with bounding boxes
visualize(image, bbox, category_id)

# Define the augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=360, p=1.0),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Apply the transformation
augmented = transform(image=image, bboxes=bbox, category_ids=category_id)
augmented_image = augmented['image'].permute(1, 2, 0).numpy()  # Convert from tensor to numpy array
augmented_bboxes = augmented['bboxes']
augmented_category_ids = augmented['category_ids']

# Display augmented image with bounding boxes
visualize(augmented_image, augmented_bboxes, augmented_category_ids)