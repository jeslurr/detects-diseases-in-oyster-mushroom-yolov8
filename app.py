"""
app.py - FastAPI Endpoint for Oyster Mushroom Disease Detection
===============================================================
POST /predict — Upload image, get disease detection results.

Response format:
{
    "status": "healthy" | "infected",
    "detections": [
        {"class": "green_mold", "confidence": 0.93, "bbox": [x1, y1, x2, y2]}
    ]
}
"""
import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    print("[INIT] Loading model...")
    try:
        inference.load_model()
        print("[OK] Model loaded and ready!")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("   The API will start but /predict will fail until a model is available.")
    yield
    print("[SHUTDOWN] Shutting down...")


app = FastAPI(
    title="Oyster Mushroom Disease Detection API",
    description=(
        "Detects green_mold (Trichoderma) and black_mold in oyster mushroom images. "
        "Returns 'healthy' if no disease is detected."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Oyster Mushroom Disease Detection",
        "status": "running",
        "model_loaded": inference._model is not None,
        "endpoints": {
            "predict": "POST /predict (upload image file)",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": inference._model is not None,
        "model_path": inference._model_path,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    confidence: float = 0.25,
):
    """
    Predict disease in an oyster mushroom image.

    - **file**: Image file to analyze
    - **confidence**: Detection confidence threshold (0.0-1.0, default 0.25)

    Returns detection results with status (healthy/infected).
    """
    if inference._model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with 03_train.py",
        )

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}",
        )

    # Read image
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Run inference
    try:
        start = time.time()
        result = inference.predict(contents, conf=confidence)
        elapsed = time.time() - start
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    response = result.to_dict()
    response["inference_time_ms"] = round(elapsed * 1000, 2)
    response["image_size"] = {"width": result.image_width, "height": result.image_height}
    response["filename"] = file.filename

    return JSONResponse(content=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
