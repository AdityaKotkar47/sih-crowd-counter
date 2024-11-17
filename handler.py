from typing import Dict
from PIL import Image
import io
from ultralytics import YOLO
import base64
import logging
import torch
import os

logger = logging.getLogger(__name__)

class EndpointHandler:
    def __init__(self, path=""):
        """Initialize the YOLO model for crowd detection."""
        try:
            # Force CPU usage if GPU is not needed
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # Initialize model with CPU
            self.model = YOLO("yolov8n.pt")
            self.model.to('cpu')
            
            # Clear CUDA cache if it was used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.initialized = True
            logger.info("YOLO model initialized successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            self.initialized = False
            raise

    def __call__(self, data: Dict) -> Dict:
        """Process an image and return crowd count prediction."""
        try:
            # Basic validation
            if not self.initialized:
                return {"error": "Model not initialized"}
            
            if "inputs" not in data:
                return {"error": "No 'inputs' key in request data"}

            image_bytes = data["inputs"]
            
            # Convert bytes to PIL Image
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Resize image if it's too large
                max_size = 800
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.LANCZOS)
                    
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                return {"error": "Failed to process image"}

            # Run inference with timeout
            try:
                with torch.no_grad():  # Disable gradient calculation
                    results = self.model(image, verbose=False)
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                return {"error": "Model inference failed"}

            # Process results
            if not results or len(results) == 0:
                return {
                    "count": 0,
                    "message": "No detections found"
                }

            # Count people (class 0 is 'person')
            try:
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
                
                logger.info(f"Successfully detected {person_count} people")
                return {
                    "count": person_count,
                    "message": f"Detected {person_count} people in the image"
                }
            except Exception as e:
                logger.error(f"Result processing error: {str(e)}")
                return {"error": "Failed to process detection results"}

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error occurred"}
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()