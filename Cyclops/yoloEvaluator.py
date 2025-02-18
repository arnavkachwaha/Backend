import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple
from . import segformerModelDef
import torchvision.transforms as transforms
from .roboflowService import RoboflowService
from ultralytics import YOLO
from pathlib import Path

class YOLOEvaluator:
    # Load the model and set it to evaluation mode
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(Path(__file__).resolve().parent / "models" / "yolo_v8_obb.pt")
        self.IRIS_MM = 11.7 # Iris diameter in mm
        self.last_pupil_diameter = 0
        self.last_iris_circle = None
        self.last_pupil_circle = None

        print("GPU Available: ", torch.cuda.is_available())


    # OpenCV
    def preprocess_image(self, image: np.ndarray, input_shape=(1, 3, 640, 640)) -> torch.Tensor:
        # Convert the image from BGR (OpenCV format) to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        # resized_image = cv2.resize(rgb_image, (input_shape[3], input_shape[2]))

        # Convert to tensor 
        transform = transforms.ToTensor()
        tensor_image = transform(rgb_image).unsqueeze(0)  # Add batch dimension
       
        return tensor_image.to(self.device)

    # Postprocess the mask to get the final segmentation mask
    def postprocess_mask(self, mask: torch.Tensor, output_shape=(640, 640)) -> np.ndarray:
        # upsampled_mask = torch.nn.functional.interpolate(mask, size=output_shape, mode='bilinear', align_corners=False)
        # np_mask = upsampled_mask.squeeze(0).argmax(0).detach().cpu().numpy()

        obb_data = mask.obb.data  # Get the OBB tensor
        class_names = mask.names  # Get class labels  # {0: 'Eye', 1: 'Iris', 2: 'Pupil'}
        bounding_boxex = []

        if obb_data is not None and isinstance(obb_data, torch.Tensor):
            for obb in obb_data.cpu().numpy():  # Convert to NumPy for easy handling
                x_center, y_center, width, height, angle, conf, cls = obb
                cls_name = class_names[int(cls)]  # Convert class ID to name

                # Convert angle from radians to degrees
                angle_deg = np.degrees(angle)

                # Define rotated rectangle
                rect = ((x_center, y_center), (width, height), angle_deg)
            
                bounding_boxex.append((cls_name, rect))
                
        return bounding_boxex
    
    # Fit a circle to the binary mask
    def fit_circles_to_mask(self, bounding_boxex: np.ndarray) -> Tuple[Tuple[int, int], int]:
        class_circles = []

        for bounding_box in bounding_boxex:
            cls_name, rect = bounding_box
            x_center, y_center = rect[0]
            width, height = rect[1]
            radius = int(max(width, height) / 2)
            center = (int(x_center), int(y_center))
            class_circles.append((cls_name, (center, radius)))
        
        return class_circles  # Return default values if no contours found

    # Overlay the mask and circle on the original image
    def get_pupil_diameter(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        # Check brightness of the image
        brightness = self.calculate_brightness(image)

        # If brightness is less than 50, return with radius 0
        if brightness < 50:
            print(f"Frame discarded due to low brightness: {brightness}")
            return image, 0 
        
        # Preprocess the image
        input_data = self.preprocess_image(image)
        
        # Perform a prediction
        with torch.no_grad():
            output = self.model(input_data)

        if len(output) == 0:
            print(f"No detections for {img_name}")
            return image, 0

        # Post-process the mask
        mask = self.postprocess_mask(output[0])

        
        # Fit the circle to the mask
        # center, pixel_radius = self.fit_circles_to_mask(mask)
        class_circles = self.fit_circles_to_mask(mask)
        iris_circle = None
        pupil_circle = None
        for class_circle in class_circles:
            cls_name, (center, pixel_radius) = class_circle
            if cls_name == 'Iris':
                iris_circle = (center, pixel_radius)
            elif cls_name == 'Pupil':
                pupil_circle = (center, pixel_radius)
        
        if iris_circle is None:
            print("No iris detected")
            return image, 0
        
        if pupil_circle is None:
            print("No pupil detected")
            return image, 0

        if self.last_iris_circle is None:
            self.last_iris_circle = iris_circle
        
        #TODO: looking into checking for sudden changes in pupil size due to error or blinking
        # check if pupil location has changed significantly
        # we also check using diameter below 
        # a more systematic analysis of the data is needed to determine the best threshold and method 
        
        if abs(iris_circle[0][0] - self.last_iris_circle[0][0]) > 50 or abs(iris_circle[0][1] - self.last_iris_circle[0][1]) > 50:
            print("Iris location changed significantly")
            return image, 0
        
        self.last_iris_circle = iris_circle
 
        if self.last_pupil_circle is None:
            self.last_pupil_circle = pupil_circle
        if abs(pupil_circle[0][0] - self.last_pupil_circle[0][0]) > 50 or abs(pupil_circle[0][1] - self.last_pupil_circle[0][1]) > 50:
            print("Pupil location changed significantly")
            return image, 0
        
        self.last_pupil_circle = pupil_circle

        pupil_pixel_radius = pupil_circle[1]
        pupil_pixel_diameter = pupil_pixel_radius * 2
        iris_diameter = iris_circle[1] * 2
        
        # TODO: Convert Radius from pixels to mm
        mm_diameter = self.get_mm_diameter(iris_diameter, pupil_pixel_diameter)

        # check for sudden large changes in diameter indicatig error
        if self.last_pupil_diameter == 0:
            self.last_pupil_diameter = mm_diameter

        # still toying with this threshold
        if abs(mm_diameter - self.last_pupil_diameter) > 0.5:
            print(f"Frame discarded due to large change in diameter: {mm_diameter}")
            return image, 0

        # check if change in diameter is less than 0.5 mm
        if abs(mm_diameter - self.last_pupil_diameter) < 0.05:
            print(f"Frame discarded due to small change in diameter: {mm_diameter}")
            mm_diameter = self.last_pupil_diameter
        else:
            # update the last pupil diameter
            self.last_pupil_diameter = mm_diameter
        
        return mm_diameter, pupil_circle, iris_circle

    # Add a function to calculate the brightness of the image
    def calculate_brightness(self, image: np.ndarray) -> float:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(grayscale_image)  # Return the average pixel value

    # Add a function to convert the radius from pixels to mm
    def get_mm_diameter(self, irirs_diameter: int, pupil_diameter: int) -> float:
        # calculate the mm per pixel
        scale_factor = self.IRIS_MM / irirs_diameter

        # convert the radius from pixels to mm
        mm_diameter = pupil_diameter * scale_factor
        return round(mm_diameter, 2)


        