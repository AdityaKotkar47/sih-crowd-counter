from typing import Dict
from PIL import Image
import io
from ultralytics import YOLO
import base64
import logging

logger = logging.getLogger(__name__)

class EndpointHandler:
    def __init__(self, path=""):
        """
        Initialize the YOLO model for crowd detection.
        
        Args:
            path: Optional path to custom model weights
        """
        try:
            self.model = YOLO("yolov8n.pt")
            self.initialized = True
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            self.initialized = False
            raise

    def __call__(self, data: Dict) -> Dict:
        """
        Process an image and return crowd count prediction.
        
        Args:
            data: Dictionary with key 'inputs' containing the image bytes
        Returns:
            Dictionary with crowd count prediction or error message
        """
        # Validate initialization
        if not self.initialized:
            logger.error("Model not initialized")
            return {"error": "Model not initialized"}

        # Validate input data
        if "inputs" not in data:
            logger.error("No 'inputs' key in request data")
            return {"error": "No 'inputs' key in request data"}

        try:
            # Extract and validate input
            image_bytes = data["inputs"]
            if not isinstance(image_bytes, (bytes, str)):
                logger.error("Invalid input format")
                return {"error": "Invalid 'inputs' format. Expected bytes or base64 string."}

            # Decode base64 if necessary
            if isinstance(image_bytes, str):
                try:
                    image_bytes = base64.b64decode(image_bytes)
                except Exception as e:
                    logger.error(f"Base64 decoding failed: {str(e)}")
                    return {"error": "Invalid base64 string"}

            # Convert to PIL Image
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to open image: {str(e)}")
                return {"error": "Failed to open image. Please ensure it's a valid image file."}

            # Run inference
            results = self.model(image)

            # Validate results
            if not results or not hasattr(results[0], "boxes"):
                logger.warning("No valid detections found")
                return {
                    "count": 0,
                    "message": "No people detected in the image"
                }

            # Count people (class 0 in COCO dataset is 'person')
            person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
            
            logger.info(f"Successfully detected {person_count} people")
            return {
                "count": person_count,
                "message": f"Detected {person_count} people in the image"
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "error": f"Error processing image: {str(e)}"
            }