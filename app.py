from fastapi import FastAPI, File, UploadFile
from handler import EndpointHandler
from PIL import Image
import io

app = FastAPI()

# Initialize the handler
handler = EndpointHandler()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict crowd count.
    Args:
        file: Uploaded image file.
    Returns:
        JSON response with crowd count.
    """
    try:
        # Read the image file
        image_bytes = await file.read()
        
        # Prepare input for handler
        data = {"inputs": image_bytes}
        
        # Get prediction
        response = handler(data)
        return response
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
