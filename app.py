from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from handler import EndpointHandler
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crowd Detection API",
    description="API for detecting and counting people in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the handler
try:
    handler = EndpointHandler()
    logger.info("Handler initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize handler: {str(e)}")
    raise

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {
        "status": "online",
        "message": "Pravaah Crowd Detection API",
        "endpoints": {
            "/predict/": "POST endpoint for crowd detection (requires image file)"
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict crowd count from an image.
    
    Args:
        file: Uploaded image file (must be an image format)
    Returns:
        JSON response with crowd count or error message
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read the image file
        image_bytes = await file.read()
        
        # Prepare input for handler
        data = {"inputs": image_bytes}
        
        # Get prediction
        response = handler(data)
        
        # Check for errors in handler response
        if "error" in response:
            raise HTTPException(
                status_code=500,
                detail=response["error"]
            )
            
        return JSONResponse(content=response)
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

# Add an error handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )