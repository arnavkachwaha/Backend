from roboflow import Roboflow
import supervision as sv
import cv2

class RoboflowService:
    def __init__(self):
        self.initialize_model()

    # Initialize the Roboflow model
    def initialize_model(self):
        rf = Roboflow(api_key="rsXNxxW9CvcYI9TSLFpu")
        project = rf.workspace().project("video1-ba4g1")
        self.model = self.project.version(2).model


    def process_image(self, image_path, confidence_threshold=20, overlap_threshold=10):
        """
        Process an image using the Roboflow model.
        Args:
            image_path (str): Path to the image file.
            confidence_threshold (float): Confidence threshold for predictions.
            overlap_threshold (float): Overlap threshold for predictions.

        Returns:
            detections (sv.Detections): Processed detections from the image.
        """

        # Predict using the Roboflow model
        result = self.model.predict(image_path, confidence=confidence_threshold, overlap=overlap_threshold).json()

        if len(result["predictions"]) == 0:
            print("No iris detected.")
            return

        # keep only the iris class
        result["predictions"] = [item for item in result["predictions"] if item["class"] == "iris"]

        if len(result["predictions"]) == 0:
            print("No iris detected.")
            return

        print(result["predictions"])

        # Parse the Roboflow result
        boxes = np.array([[item["x"] - item["width"] / 2 ,  # xmin
                        item["y"] - item["height"] / 2,  # ymin
                        item["x"] + item["width"] / 2 ,  # xmax
                        item["y"] + item["height"]]  # ymax
                        for item in result["predictions"]])

        if len(boxes) == 0:
            print("No iris detected.")
            return

        confidences = np.array([item["confidence"] for item in result["predictions"]])

        # Map string labels to integers
        class_labels = [item["class"] for item in result["predictions"]]
        unique_labels = list(set(class_labels))  # Get unique labels
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}  # Map labels to IDs
        class_ids = np.array([label_to_id[label] for label in class_labels])


        # Create a Detections object
        detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

        if len(detections) == 0:
            print("No iris detected.")
            return
        
        print(detections)

        return detections

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
