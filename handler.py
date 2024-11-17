from typing import Dict
from PIL import Image
import io
from ultralytics import YOLO
import base64

class EndpointHandler:
    def __init__(self, path=""):
        # Initialize the YOLO model
        self.model = YOLO("yolov8n.pt")
        self.initialized = True

    def __call__(self, data: Dict) -> Dict:
        """
        Args:
            data: Dictionary with key 'inputs' containing the image bytes
        Returns:
            Dictionary with crowd count prediction
        """
        # Check if the model is initialized
        if not self.initialized:
            return {"error": "Model not initialized"}

        # Check if 'inputs' key exists
        if "inputs" not in data:
            return {"error": "No 'inputs' key in request data"}

        try:
            # Extract and validate input
            image_bytes = data["inputs"]
            if not isinstance(image_bytes, (bytes, str)):
                return {"error": "Invalid 'inputs' format. Expected bytes or base64 string."}

            # Decode base64 if necessary
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run inference
            results = self.model(image)

            # Validate results
            if not results or not hasattr(results[0], "boxes"):
                return {"error": "No valid detections found in the image"}

            # Count people (class 0 in COCO dataset is 'person')
            person_count = sum(
                1 for box in results[0].boxes if int(box.cls[0]) == 0
            )

            return {
                "count": person_count,
                "message": f"Detected {person_count} people in the image"
            }

        except Exception as e:
            import traceback
            return {
                "error": f"Error processing image: {str(e)}",
                "traceback": traceback.format_exc()  # Optional, remove in production
            }
