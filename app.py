from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from handler import EndpointHandler
from PIL import Image
import io

# Initialize FastAPI app and model handler
app = FastAPI()
handler = EndpointHandler()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the crowd count from an uploaded image file.
    Args:
        file: UploadFile - the image file to process
    Returns:
        JSON response with the count of people in the image
    """
    try:
        # Read the uploaded file's bytes
        image_bytes = await file.read()
        
        # Prepare data for the handler
        data = {"inputs": image_bytes}
        result = handler(data)

        # If the handler returns an error, raise an exception
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, status_code=500
        )