"""
Cisterna Magna Guidance System - FastAPI Server

This module implements a FastAPI-based web server for ultrasound-guided procedures,
specifically targeting the cisterna magna region. It provides REST endpoints for machine
learning inference on both radiofrequency (RF) data and ultrasound images, as well as
WebSocket support for real-time data streaming from ultrasound probes.

Key Features:
- RF Data Inference: Mock implementation for detecting cisterna magna via RF A-lines.
- Image Localization: PyTorch-based model for predicting (x, y) coordinates in ultrasound images.
- Real-time Angles: WebSocket endpoint for gyroscope/IMU angle updates.
- Static File Serving: Optional SPA (Single Page Application) support for frontend.

Usage:
    Run as a script with optional checkpoint for image model:
    python app.py --checkpoint checkpoints/best_model.pt --port 3000

Endpoints:
- GET /angles: Retrieve latest probe angles.
- POST /predict/rf: Predict on RF data (mocked).
- POST /predict/image: Predict localization on uploaded image (requires checkpoint).
- WS /ws: WebSocket for real-time angle/calibration updates.

Dependencies:
- FastAPI for web framework.
- PyTorch for ML inference.
- PIL for image processing.
- NumPy for numerical operations.
"""

import io
import json
import ssl
import os
import argparse
import asyncio
import uvicorn
import numpy as np

from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from PIL import Image

# ── App setup ─────────────────────────────────────────────────────────────────

# Initialize FastAPI application with a descriptive title for the ultrasound guidance system
app = FastAPI(title="Cisterna Magna Guidance System")

# ── Shared state ──────────────────────────────────────────────────────────────

# Global dictionary to store the latest gyroscope/IMU angles and calibration timestamps.
# This state is shared across WebSocket connections and REST endpoints for real-time updates.
latest_angles: dict = {
    "x": 0.0,  # Current x-axis angle in degrees
    "y": 0.0,  # Current y-axis angle in degrees
    "calibrated_at": None,  # ISO timestamp when calibration was last performed
    "updated_at": None,     # ISO timestamp when angles were last updated
}

# ── ML models ─────────────────────────────────────────────────────────────────

class MockRFModel:
    """
    Stub RF model – swap out for your real ultrasound RF inference call.
    Expected input:  1-D float32 numpy array of raw RF A-lines.
    If your model expects shape (num_lines, samples_per_line), reshape before calling.
    """
    def predict(self, rf_data: np.ndarray) -> dict:
        return {
            "cisterna_magna_detected": bool(np.random.rand() > 0.3),
            "confidence":             float(np.random.rand()),
            "depth_mm":               float(np.random.uniform(40, 80)),
            "tilt_x_deg":             float(np.random.uniform(-15, 15)),
            "tilt_y_deg":             float(np.random.uniform(-15, 15)),
            "classification":         "cisterna_magna" if np.random.rand() > 0.4 else "other",
        }


# ── ML models ─────────────────────────────────────────────────────────────────

# Global variables for the image-based localization model.
# These are initialized when the server starts with a checkpoint.
_image_model     = None   # Instance of UltrasoundLocalizer PyTorch model
_image_transform = None   # Torchvision transform pipeline for preprocessing images
_device          = None   # Torch device (CPU or CUDA) for model inference

# Instantiate the mock RF model for demonstration purposes.
# In production, replace with actual RF inference logic.
rf_model = MockRFModel()


def _load_image_model(checkpoint_path: str):
    """
    Load the UltrasoundLocalizer checkpoint.
    Imports are deferred so the server still starts without a GPU / checkpoint
    when those args are omitted.
    """
    # Import PyTorch and related modules only when needed to avoid startup errors
    # if PyTorch is not installed or GPU is unavailable.
    import torch
    from model import UltrasoundLocalizer          # Custom model definition
    from predict import load_model, get_transform  # Helper functions for loading and preprocessing

    # Update global variables with loaded model components
    global _image_model, _image_transform, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Prefer GPU if available
    _image_model = load_model(checkpoint_path, _device)  # Load model weights from checkpoint
    _image_transform = get_transform(224)  # Create image preprocessing transform (224x224 resize)
    print(f"Image model loaded from {checkpoint_path} on {_device}")  # Log successful loading


# ── Request / response schemas ────────────────────────────────────────────────

# Pydantic models defining the structure of API requests and responses.
# These ensure data validation and automatic documentation generation.

class PredictRFRequest(BaseModel):
    """
    Request schema for RF data prediction endpoint.
    Expects raw radiofrequency A-lines from the ultrasound probe.
    """
    rf_data:  list[float]  # Flattened list of RF signal values (floats)
    metadata: Optional[dict] = None  # Optional probe settings (e.g., gain, frequency)


class PredictRFResponse(BaseModel):
    """
    Response schema for RF prediction results.
    Includes detection, confidence, depth, tilt corrections, and classification.
    """
    cisterna_magna_detected: bool   # Whether cisterna magna was detected
    confidence:              float  # Prediction confidence score (0-1)
    depth_mm:                float  # Estimated depth to cisterna magna in millimeters
    tilt_x_deg:              float  # Recommended x-axis tilt correction in degrees
    tilt_y_deg:              float  # Recommended y-axis tilt correction in degrees
    classification:          str    # Tissue classification label (e.g., "cisterna_magna" or "other")
    timestamp:               str    # ISO 8601 timestamp of prediction


class PredictImageResponse(BaseModel):
    """
    Response schema for image-based localization.
    Returns predicted (x, y) coordinates in degrees for probe positioning.
    """
    x:         float  # Predicted x-coordinate in degrees
    y:         float  # Predicted y-coordinate in degrees
    timestamp: str    # ISO 8601 timestamp of prediction


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/angles")
async def get_angles():
    """
    Retrieve the latest gyroscope/IMU angles from the shared state.
    Used by clients to get current probe orientation for guidance.
    """
    return JSONResponse(content=latest_angles)


@app.post("/predict/rf", response_model=PredictRFResponse)
async def predict_rf(payload: PredictRFRequest):
    """
    RF-data ML endpoint.

    Accepts raw RF data from the ultrasound probe and returns:
      • detection flag + confidence
      • estimated depth to cisterna magna
      • tilt corrections (regression outputs)
      • tissue classification label
    """
    # Validate input: ensure RF data is provided
    if not payload.rf_data:
        raise HTTPException(status_code=422, detail="rf_data must not be empty.")

    # Convert list to NumPy array for model input
    rf_array = np.array(payload.rf_data, dtype=np.float32)

    # Optional: Reshape if model expects 2D input (e.g., multiple A-lines)
    # rf_array = rf_array.reshape(num_lines, samples_per_line)

    # Perform inference using the mock RF model
    result = rf_model.predict(rf_array)

    # Return response with current timestamp
    return PredictRFResponse(**result, timestamp=datetime.now(timezone.utc).isoformat())


@app.post("/predict/image", response_model=PredictImageResponse)
async def predict_image(image: UploadFile = File(...)):
    """
    Image-based localisation endpoint.

    Accepts a multipart image upload (JPEG / PNG), runs the UltrasoundLocalizer,
    and returns denormalised (x, y) pixel coordinates.

    Requires the server to have been started with --checkpoint.
    """
    # Check if image model is loaded; raise error if not
    if _image_model is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. Start the server with --checkpoint <path>.",
        )

    # Import required modules for inference
    import torch
    from dataset import denormalize_x, denormalize_y  # Functions to convert normalized predictions to degrees

    # Read and preprocess the uploaded image
    raw = await image.read()  # Read image bytes
    img = Image.open(io.BytesIO(raw)).convert("RGB")  # Convert to RGB PIL image
    inp = _image_transform(img).unsqueeze(0).to(_device)  # Apply transform and add batch dimension

    # Run inference without gradient computation
    with torch.no_grad():
        pred = _image_model(inp).squeeze(0).cpu()  # Get predictions and move to CPU

    # Denormalize predictions to degrees
    x = denormalize_x(pred[0:1]).item()  # Extract and denormalize x-coordinate
    y = denormalize_y(pred[1:2]).item()  # Extract and denormalize y-coordinate

    # Return response with timestamp
    return PredictImageResponse(x=x, y=y, timestamp=datetime.now(timezone.utc).isoformat())


# Convenience alias so existing clients calling /predict still hit the RF model
@app.post("/predict", response_model=PredictRFResponse)
async def predict_rf_alias(payload: PredictRFRequest):
    """Backwards-compatible alias for /predict/rf."""
    return await predict_rf(payload)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time communication with ultrasound clients.
    Handles incoming messages for angle updates and calibration commands.
    Maintains persistent connection for streaming probe data.
    """
    # Accept the WebSocket connection
    await ws.accept()
    print(f"Client connected from: {ws.client.host}")
    
    try:
        # Main message loop: listen for incoming messages
        while True:
            # Receive raw text message from client
            raw = await ws.receive_text()
            
            try:
                # Parse JSON message
                data = json.loads(raw)
                msg_type = data.get("type")  # Extract message type
                
                # Handle angle update messages
                if msg_type == "angles":
                    # Update global state with new x/y angles
                    latest_angles["x"] = data.get("x", 0.0)
                    latest_angles["y"] = data.get("y", 0.0)
                    # Record timestamp of update
                    latest_angles["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Handle calibration messages
                elif msg_type == "calibrate":
                    # Mark calibration timestamp
                    latest_angles["calibrated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Log received data for debugging
                print(f"Received: {data}")
            
            except json.JSONDecodeError:
                # Handle invalid JSON gracefully
                print(f"Invalid message (not JSON): {raw}")
    
    except WebSocketDisconnect:
        # Handle client disconnection
        print("Client disconnected")


# ── Static file serving (SPA shell) ──────────────────────────────────────────

# Paths for serving a built Single Page Application (SPA) frontend.
# If a 'dist' directory exists (e.g., from a build tool like Vite), serve static assets.
BUILD_PATH = os.path.join(os.path.dirname(__file__), "dist")  # Path to built frontend
INDEX_HTML = os.path.join(BUILD_PATH, "index.html")          # Main HTML file

# Conditionally mount static file routes if frontend build exists
if os.path.exists(INDEX_HTML):
    # Mount assets directory for CSS/JS/images
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(BUILD_PATH, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """
        Catch-all route for SPA routing.
        Serves static files if they exist, otherwise serves index.html for client-side routing.
        """
        file_path = os.path.join(BUILD_PATH, full_path)
        if os.path.isfile(file_path):
            # Serve the requested file directly
            return FileResponse(file_path)
        # Fallback to index.html for SPA routes
        return FileResponse(INDEX_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Command-line argument parsing for server configuration
    parser = argparse.ArgumentParser(description="Cisterna Magna Guidance System")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to UltrasoundLocalizer checkpoint (.pt). "
             "Required to enable POST /predict/image.",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 3000)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Load image model if checkpoint provided
    if args.checkpoint:
        _load_image_model(args.checkpoint)
    else:
        print("No --checkpoint supplied – /predict/image will return 503.")

    # Configure SSL/TLS if certificates are available
    ssl_context = None
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain("cert.pem", "key.pem")
        print(f"TLS enabled – https://{args.host}:{args.port}")
    else:
        print(f"TLS certs not found – http://{args.host}:{args.port} (no TLS)")

    # Print available endpoints for user reference
    print("Endpoints:")
    print("  GET  /angles")
    print("  POST /predict          (RF, backwards-compat alias)")
    print("  POST /predict/rf       (RF data)")
    print("  POST /predict/image    (image upload, requires --checkpoint)")
    print("  WS   /ws")

    # Start the Uvicorn server with configured options
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile="key.pem"  if ssl_context else None,
        ssl_certfile="cert.pem" if ssl_context else None,
    )


