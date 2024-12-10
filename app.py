from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from handler import EndpointHandler
from contextlib import asynccontextmanager
import logging
import uvicorn
import json
import os
from heatmap_gen import generate_heatmap, HEATMAP_OUTPUT_PATH
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize handler as None
handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global handler
    try:
        # Initialize on startup
        handler = EndpointHandler()
        logger.info("Handler initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize handler: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        if handler and hasattr(handler, 'model'):
            del handler.model
            handler = None

# Create FastAPI app without docs
app = FastAPI(
    docs_url=None,    # Disable swagger documentation
    redoc_url=None,   # Disable redoc documentation
    openapi_url=None, # Disable openapi schema
    lifespan=lifespan # Add lifespan context manager
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to check API status"""
    return {
        "status": "online",
        "model_status": "loaded" if handler and handler.initialized else "initializing"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict crowd count from an image"""
    if file is None or file.content_type is None:
        raise HTTPException(status_code=400, detail="Invalid file")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    if not handler or not handler.initialized:
        raise HTTPException(
            status_code=503,
            detail="Model is initializing. Please try again in a few moments."
        )
    
    try:
        contents = await file.read()
        # Preprocess image and get prediction
        count = handler.preprocess_image(contents)
        
        result = {
            "count": count,
            "status": "success"
        }
            
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/heatmap")
async def get_heatmap():
    """Serve the latest heatmap.svg file"""
    if not os.path.exists(HEATMAP_OUTPUT_PATH):
        raise HTTPException(status_code=404, detail="Heatmap SVG file not found")
    return Response(content=open(HEATMAP_OUTPUT_PATH, 'rb').read(), media_type="image/svg+xml")

@app.post("/update-data")
async def update_data(data: dict):
    """Update crowd counts and regenerate the heatmap SVG"""
    try:
        # Assuming data contains regions and their respective crowd counts
        regions = data.get("regions", [])
        if not regions:
            raise HTTPException(status_code=400, detail="No region data provided")

        # Save the updated region data to the config file
        with open("config/regions.json", "w") as f:
            json.dump({"regions": regions}, f)

        # Call the heatmap generation function
        generate_heatmap()

        return JSONResponse(content={"message": "Heatmap updated successfully"})
    
    except Exception as e:
        logger.error(f"Error updating heatmap: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

