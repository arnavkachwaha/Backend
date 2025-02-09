import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from pathlib import Path

# Get absolute paths relative to the script
model_path = Path(__file__).resolve().parent / "v8-obb" / "best.pt"
image_folder = Path(__file__).resolve().parent.parent / "images" / "train"
output_folder = Path(__file__).resolve().parent / "predictions" / "train"
output_folder.mkdir(exist_ok=True)

# Load YOLOv8-OBB model
model = YOLO(model_path)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process images
for img_name in os.listdir(image_folder):
    img_path = str(image_folder / img_name)
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Run inference
    results = model(img_tensor)

    if len(results) == 0:
        print(f"No detections for {img_name}")
        continue

    obb_data = results[0].obb.data  # Get the OBB tensor
    class_names = results[0].names  # Get class labels

    if obb_data is not None and isinstance(obb_data, torch.Tensor):
        for obb in obb_data.cpu().numpy():  # Convert to NumPy for easy handling
            x_center, y_center, width, height, angle, conf, cls = obb
            cls_name = class_names[int(cls)]  # Convert class ID to name
            
            # Convert angle from radians to degrees
            angle_deg = np.degrees(angle)

            # Define rotated rectangle
            rect = ((x_center, y_center), (width, height), angle_deg)
            box = cv2.boxPoints(rect)  # Get 4 corner points
            box = np.int0(box)  # Convert to integer

            # Draw rotated bounding box
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            # Draw class label and confidence
            text = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, text, (int(x_center -width/2), int(y_center) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Class: {cls_name}, Confidence: {conf:.2f}")
            print(f"X: {x_center}, Y: {y_center}, W: {width}, H: {height}, Angle: {angle_deg}\n")

    # Save annotated image
    output_path = str(output_folder / img_name)
    cv2.imwrite(output_path, img)

print(f"Predictions saved in: {output_folder}")
