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
import threading

logger = logging.getLogger(__name__)

class EndpointHandler:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_path = "yolov8n.pt"
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EndpointHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, path=""):
        """Initialize the YOLO model with proper locking and caching."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    try:
                        # Force CPU usage and limit threads
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                        torch.set_num_threads(4)

                        # Check if model exists locally
                        model_path = Path(self._model_path)
                        if not model_path.exists():
                            logger.info("Downloading model for the first time...")
                            
                        # Initialize model with CPU
                        if self._model is None:
                            self._model = YOLO(self._model_path)
                            self._model.to('cpu')
                            
                        # Clear GPU memory if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        self._initialized = True
                        logger.info("YOLO model initialized successfully on CPU")
                    except Exception as e:
                        logger.error(f"Failed to initialize YOLO model: {str(e)}")
                        raise

    def preprocess_image(self, image_bytes):
        """Preprocess image with size and format validation."""
        try:
            # Check image size
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image size exceeds 10MB limit")

            # Open and validate image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize if needed
            max_size = 640
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
                
            return image
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def __call__(self, data):
        """Process image with proper error handling and resource management."""
        start_time = time.time()
        try:
            # Validate initialization
            if not self._initialized or self._model is None:
                raise RuntimeError("Model not initialized")

            # Validate input
            if "inputs" not in data:
                raise ValueError("No image data provided")

            # Process image
            image = self.preprocess_image(data["inputs"])
            
            # Run inference with resource management
            with torch.no_grad():
                results = self._model(image, verbose=False)
                
            # Process results
            if not results or len(results) == 0:
                return {"count": 0}

            # Count people (class 0 is person)
            person_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
            processing_time = time.time() - start_time
            
            return {
                "count": person_count,
                "processing_time": f"{processing_time:.2f}s"
            }

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return {"error": str(e)}
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()