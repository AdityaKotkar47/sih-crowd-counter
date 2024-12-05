import logging
import torch
from PIL import Image
import io
from ultralytics import YOLO
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EndpointHandler:
    def __init__(self):
        """Initialize the YOLO model for local use."""
        try:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Set model path
            model_path = models_dir / "yolov8n.pt"
            
            # Download model if it doesn't exist
            if not model_path.exists():
                logger.info("Downloading model for the first time...")
                torch.hub.download_url_to_file(
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                    str(model_path)
                )
            
            # Initialize model
            self.model = YOLO(str(model_path))
            
            # Use GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            logger.info(f"YOLO model initialized successfully on {self.device}")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            self.initialized = False
            raise

    def preprocess_image(self, image_bytes):
        """Preprocess image for inference."""
        try:
            # Open and validate image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize if needed (optional, YOLO handles this automatically)
            max_size = 1280  # Increased for local use
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
                
            return image
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def __call__(self, data):
        """Process image and return crowd count."""
        start_time = time.time()
        
        try:
            # Validate input
            if not self.initialized:
                raise RuntimeError("Model not initialized")
            if "inputs" not in data:
                raise ValueError("No image data provided")

            # Process image
            image = self.preprocess_image(data["inputs"])
            
            # Run inference
            with torch.no_grad():
                results = self.model(image, verbose=False)
                
            # Process results
            if not results or len(results) == 0:
                return {
                    "count": 0,
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }

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
            # Clean up CUDA memory if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()