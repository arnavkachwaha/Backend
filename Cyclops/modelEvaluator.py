import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple
from . import segformerModelDef
import torchvision.transforms as transforms
from .roboflowService import RoboflowService

class ModelEvaluator:
    # Load the model and set it to evaluation mode
    def __init__(self):
        print("GPU Available: ", torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6019, 0.4767, 0.4340], std=[0.229, 0.224, 0.225]) # will need to update these values based on the dataset used for training
        ])
        self.model = segformerModelDef.CyclopsSegformerModule().to(self.device)
        state_dict = torch.load("segformer_model.pth", map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.roboflowService = RoboflowService()
        self.IRIS_MM = 11.7 # Iris diameter in mm

    # OpenCV
    def preprocess_image(self, image: np.ndarray, input_shape=(1, 3, 480, 640)) -> torch.Tensor:
        # Convert the image from BGR (OpenCV format) to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image
        resized_image = cv2.resize(rgb_image, (input_shape[3], input_shape[2]))

        # Convert to tensor and normalize
        tensor_image = self.normal_transform(resized_image).unsqueeze(0)  # Add batch dimension

        return tensor_image.to(self.device)

    # Postprocess the mask to get the final segmentation mask
    def postprocess_mask(self, mask: torch.Tensor, output_shape=(480, 640)) -> np.ndarray:
        upsampled_mask = torch.nn.functional.interpolate(mask, size=output_shape, mode='bilinear', align_corners=False)
        np_mask = upsampled_mask.squeeze(0).argmax(0).detach().cpu().numpy()

        return np_mask
    
    # Fit a circle to the binary mask
    def fit_circle_to_mask(self, binary_mask: np.ndarray) -> Tuple[Tuple[int, int], int]:
        binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            return center, radius
        
        return (0, 0), 0  # Return default values if no contours found

    # Overlay the mask and circle on the original image
    def overlay_mask_on_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
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

        # Post-process the mask
        mask = self.postprocess_mask(output.logits)
        
        # Fit the circle to the mask
        center, pixel_radius = self.fit_circle_to_mask(mask)
        pixel_diameter = pixel_radius * 2

        # TODO: Convert Radius from pixels to mm
        mm_diameter = self.get_mm_diameter(image, pixel_diameter)
        print(f"Diameter: {pixel_diameter} pixels, {mm_diameter} mm")

        # Draw the circle onto the blended image
        if center != (0, 0):
            cv2.circle(image, center, pixel_radius, (0, 255, 0), 2)  # Green circle
            cv2.putText(image, str(mm_diameter*2), 
                        org = (center[0] + pixel_radius + 1, center[1] + pixel_radius + 1), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1, 
                        color = (255, 255, 0), 
                        thickness = 2)
        
        return image, mm_diameter

    # Function to return the final predicted image with mask and circle
    # TODO: refactor or remove this function and just call overlay_mask_on_image 
    def get_predicted_output(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.overlay_mask_on_image(image)
    
    # Add a function to calculate the brightness of the image
    def calculate_brightness(self, image: np.ndarray) -> float:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(grayscale_image)  # Return the average pixel value

    # Add a function to convert the radius from pixels to mm
    def get_mm_diameter(self, image: np.ndarray, pixel_diameter: int) -> float:
        detected_width = self.roboflowService.process_image(image)
        print("detected_width: ", detected_width)

        if detected_width == 0.0:
            return 0.0

        # TODO: determine minimum width of detection
        # sometimes the pupil is detected and mislabeled as iris and will be much smaller
        if detected_width < 50:
            return 0.0

        # calculate the mm per pixel
        scale_factor = self.IRIS_MM / detected_width

        # convert the radius from pixels to mm
        mm_diameter = pixel_diameter * scale_factor
        return round(mm_diameter, 2)


        