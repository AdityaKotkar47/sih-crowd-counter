from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from handler import EndpointHandler
from PIL import Image
import io
import logging
import asyncio
from functools import partial
import time

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the handler as a global variable
handler = None

@app.on_event("startup")
async def startup_event():
    global handler
    try:
        # Initialize handler in the background
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
        "model_status": "loaded" if handler and handler.initialized else "initializing",
        "endpoints": {
            "/predict/": "POST endpoint for crowd detection (requires image file)"
        }
    }

async def process_image(file_content: bytes):
    """Process image in a separate thread to avoid blocking."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, handler, {"inputs": file_content})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict crowd count from an image."""
    global handler
    start_time = time.time()
    
    # Check if handler is initialized
    if handler is None or not hasattr(handler, 'initialized'):
        raise HTTPException(
            status_code=503,
            detail="Server is still initializing. Please try again in a few moments."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read the image file with size limit
        file_size_limit = 10 * 1024 * 1024  # 10MB
        image_bytes = await file.read()
        
        if len(image_bytes) > file_size_limit:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 10MB"
            )
        
        # Set a timeout for the entire processing
        timeout = 30  # seconds
        try:
            response = await asyncio.wait_for(
                process_image(image_bytes),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error("Processing timeout")
            raise HTTPException(
                status_code=504,
                detail="Processing timeout"
            )
        
        # Check for errors in handler response
        if "error" in response:
            raise HTTPException(
                status_code=500,
                detail=response["error"]
            )
        
        # Add processing time to response
        response["total_processing_time"] = f"{time.time() - start_time:.2f}s"
        return JSONResponse(content=response)
    
    except asyncio.TimeoutError:
        logger.error("Request timed out")
        raise HTTPException(
            status_code=504,
            detail="Request timed out"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

# Add an error handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
        }
    )