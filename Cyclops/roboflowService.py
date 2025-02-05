from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np

class RoboflowService:
    def __init__(self):
        self.initialize_model()

    # Initialize the Roboflow model
    def initialize_model(self):
        rf = Roboflow(api_key="rsXNxxW9CvcYI9TSLFpu")
        project = rf.workspace().project("video1-ba4g1")
        self.model = project.version(2).model


    def process_image(self, image_path, confidence_threshold=20, overlap_threshold=10):
        """
        Process an image using the Roboflow model.
        Args:
            image_path (str): Path to the image file.
            confidence_threshold (float): Confidence threshold for predictions.
            overlap_threshold (float): Overlap threshold for predictions.

        Returns:
            width (float): Width of the detected object in pixels
        """

        # Predict using the Roboflow model
        result = self.model.predict(image_path, confidence=confidence_threshold, overlap=overlap_threshold).json()

        # TODO: verify if all early return checks are necessary
        predictions = result.get("predictions", [])
        if not predictions:
            print("No iris detected.")
            return 0.0

        # Filter predictions to keep only the iris class
        iris_predictions = [item for item in predictions if item["class"] == "iris"]
        if len(iris_predictions) == 0:
            print("No iris detected.")
            return 0.0

        return iris_predictions[0]["width"]

    def draw_detections(self, image, detections):
        """
        Draw detections on the image.
        Args:
            image (np.ndarray): The image to draw on.
            detections (sv.Detections): The detections to draw.
        """
        # Annotate the image
        label_annotator = sv.LabelAnnotator()
        box_annotator = sv.BoxAnnotator()

        image = cv2.imread(image_path)

        annotated_image = box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=class_labels)  # Pass original labels

        sv.plot_image(image=annotated_image, size=(16, 16))
