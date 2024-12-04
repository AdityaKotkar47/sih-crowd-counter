from typing import Dict
from PIL import Image
import io
from ultralytics import YOLO
import base64
import logging
import torch
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class EndpointHandler:
    _instance = None
    _model = None
    _last_used = None
    _model_path = "yolov8n.pt"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EndpointHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, path=""):
        """Initialize the YOLO model for crowd detection with caching."""
        if not hasattr(self, 'initialized'):
            try:
                # Force CPU usage
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                torch.set_num_threads(4)  # Limit CPU threads
                
                # Check if model file exists locally
                if not Path(self._model_path).exists():
                    logger.info("Downloading model for the first time...")
                
                # Initialize model with CPU and cache it
                if self._model is None:
                    self._model = YOLO(self._model_path)
                    self._model.to('cpu')
                
                # Clear any GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self._last_used = time.time()
                self.initialized = True
                logger.info("YOLO model initialized successfully on CPU")
            except Exception as e:
                logger.error(f"Failed to initialize YOLO model: {str(e)}")
                self.initialized = False
                raise

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image with size limits and optimization."""
        try:
            # Set maximum dimensions
            max_size = 640  # YOLO's preferred size
            
            # Calculate new size maintaining aspect ratio
            ratio = max_size / max(image.size)
            if ratio < 1:  # Only resize if image is larger than max_size
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def __call__(self, data: Dict) -> Dict:
        """Process an image and return crowd count prediction with improved error handling."""
        start_time = time.time()
        try:
            # Basic validation
            if not self.initialized or self._model is None:
                return {"error": "Model not initialized"}
            
            if "inputs" not in data:
                return {"error": "No 'inputs' key in request data"}

            image_bytes = data["inputs"]
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                return {"error": "Image size too large"}
            
            # Convert bytes to PIL Image with error handling
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image = self.preprocess_image(image)
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
                return {"error": "Failed to process image"}

            # Run inference with memory optimization
            try:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                with torch.no_grad():
                    results = self._model(image, verbose=False)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                logger.error(f"Inference error: {str(e)}")
                return {"error": "Model inference failed"}

            # Process results
            if not results or len(results) == 0:
                return {"count": 0}

            # Count people (class 0 is 'person')
            try:
                person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
                processing_time = time.time() - start_time
                
                logger.info(f"Successfully detected {person_count} people in {processing_time:.2f}s")
                return {
                    "count": person_count,
                    "processing_time": f"{processing_time:.2f}s"
                }
            except Exception as e:
                logger.error(f"Result processing error: {str(e)}")
                return {"error": "Failed to process detection results"}

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error occurred"}
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._last_used = time.time()