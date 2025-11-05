import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# The original imports for os and sys were removed, and the import below
# was changed to a relative import for standard package structure.
from app.ml_service import load_student_model, predict_deepfake
from fastapi.responses import HTMLResponse
import os

# --- FastAPI App Initialization ---
app = FastAPI(title="Deepfake Detection API")

# --- CORS Configuration (Fixes "Could not connect to the API" error) ---
# Allow all origins for local development and testing
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "file:///",  # Essential to allow index.html running directly from the file system
    "*" # Catch-all for simplicity during local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Load the ML model when the application starts."""
    print("Application Startup: Starting model loading...")
    try:
        # Calls the function from the mock ml_service.py
        load_student_model()
        print("Application Startup: Model is ready for inference.")
    except Exception as e:
        print(f"FATAL ERROR during model startup: {e}")
        # Re-raise the exception to prevent the server from starting with a broken model
        raise

# --- API Endpoint ---
@app.post("/predict_deepfake")
async def predict_deepfake_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for predicting if an uploaded image is a deepfake.
    """
    try:
        # Read the file content as bytes
        image_bytes = await file.read()
        
        # Perform prediction using the ML service
        result = predict_deepfake(image_bytes)
        
        return result
        
    except ValueError as e:
        # Handle preprocessing errors (like bad image file format)
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
    except Exception as e:
        # Handle general inference errors
        print(f"Inference error: {e}")
        # In a real app, log the error details securely.
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
    
    from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# --- Serve index.html ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "../index.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"


if __name__ == "__main__":
    # To run this from the command line, you would typically use:
    # uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
    # This block is for direct execution of this file.
    uvicorn.run(app, host="127.0.0.1", port=8000)
