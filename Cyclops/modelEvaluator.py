import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Tuple
from . import segformerModelDef
import torchvision.transforms as transforms

class ModelEvaluator:
    # Load the model and set it to evaluation mode
    def __init__(self):
        self.normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6019, 0.4767, 0.4340], std=[0.229, 0.224, 0.225]) # will need to update these values based on the dataset used for training
        ])
        self.model = segformerModelDef.CyclopsSegformerModule()
        state_dict = torch.load("/Users/arnavkachwaha/Desktop/Insight-Server/segformer_model.pth")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    # Preprocess the image to get it ready for the model
    def preprocess_image(self, image: np.ndarray, input_shape=(1, 3, 480, 640)) -> torch.Tensor:
        # Convert numpy array (image) to PIL, resize, and apply transformations
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((input_shape[3], input_shape[2]))
        tensor_image = self.normal_transform(resized_image).unsqueeze(0)  # Add batch dimension

        return tensor_image

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
    def overlay_mask_on_image(self, image: np.ndarray) -> np.ndarray:
        # Preprocess the image
        input_data = self.preprocess_image(image)
        
        # Perform a prediction
        with torch.no_grad():
            output = self.model(input_data)

        # Post-process the mask
        mask = self.postprocess_mask(output.logits)
        
        # Fit the circle to the mask
        center, radius = self.fit_circle_to_mask(mask)

        # Overlay the circle and mask on the original image
        mask_colored = np.zeros_like(image)
        mask_colored[mask == 1] = [0, 255, 0]  # Green for mask

        # Blend the original image with the mask
        blended_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

        # Draw the circle onto the blended image
        if center != (0, 0):
            cv2.circle(blended_image, center, radius, (255, 0, 0), 2)  # Blue circle
        
        return blended_image

    # Function to return the final predicted image with mask and circle
    def get_predicted_output(self, image: np.ndarray) -> np.ndarray:
        return self.overlay_mask_on_image(image)
