from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from handler import EndpointHandler
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
        "model_status": "loaded" if handler and handler._initialized else "initializing",
        "endpoints": {
            "/predict/": "POST endpoint for crowd detection (requires image file)"
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict crowd count from an image"""
    global handler
    
    # Check if handler is initialized
    if not handler or not handler._initialized:
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
        # Read file
        contents = await file.read()
        
        # Process image
        result = handler({"inputs": contents})
        
        # Check for errors in result
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )
            
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )