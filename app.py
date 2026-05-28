"""
Cisterna Magna Guidance System — FastAPI Server

Single inference server. Models are hot-swappable at runtime via
POST /model/image/load or POST /model/rf/load without restarting.

Usage:
    python3 app.py [--checkpoint checkpoints/best_model.pt] [--port 3000]

Endpoints:
    GET  /angles              latest probe IMU angles
    POST /predict/rf          RF data inference
    POST /predict/image       image localization (needs a loaded checkpoint)
    POST /predict             backwards-compat alias for /predict/rf
    GET  /model/status        show currently loaded models and metadata
    POST /model/image/load    hot-swap the image model checkpoint at runtime
    WS   /ws                  real-time angle / calibration stream
"""

import asyncio
import io
import json
import os
import socket
import ssl
import argparse
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Cisterna Magna Guidance System")


# ── Validation error handler ──────────────────────────────────────────────────
# FastAPI's default handler crashes with UnicodeDecodeError when the request
# body contains raw binary data (e.g. a PNG sent to a JSON endpoint).

def _sanitize(obj):
    """Recursively replace bytes with a safe placeholder so JSON encoding never fails."""
    if isinstance(obj, bytes):
        return f"<binary {len(obj)} bytes>"
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})

# ── Shared probe state ────────────────────────────────────────────────────────

latest_angles: dict = {
    "x": 0.0,
    "y": 0.0,
    "calibrated_at": None,
    "updated_at": None,
}

# ── Model registry ────────────────────────────────────────────────────────────

_model_lock = asyncio.Lock()

# Image model state
_image_model      = None
_image_transform  = None
_image_device     = None
_image_checkpoint = None   # path of the currently loaded checkpoint
_image_loaded_at  = None   # ISO timestamp

# RF model state (swap MockRFModel for a real one via POST /model/rf/load)
_rf_model         = None
_rf_model_name    = None
_rf_loaded_at     = None


class SweepPredictRequest(BaseModel):
    folder: str   # absolute path to the braindata/ folder

class SweepPredictResponse(BaseModel):
    best_slice_index: int
    best_confidence:  float
    confidences:      list[float]

class MockRFModel:
    """Stub RF model — replace with real inference when available."""
    name = "MockRFModel"

    def predict(self, rf_data: np.ndarray) -> dict:
        return {
            "cisterna_magna_detected": bool(np.random.rand() > 0.3),
            "confidence":             float(np.random.rand()),
            "depth_mm":               float(np.random.uniform(40, 80)),
            "tilt_x_deg":             float(np.random.uniform(-15, 15)),
            "tilt_y_deg":             float(np.random.uniform(-15, 15)),
            "classification":         "cisterna_magna" if np.random.rand() > 0.4 else "other",
        }


def _init_rf_model():
    global _rf_model, _rf_model_name, _rf_loaded_at
    _rf_model      = MockRFModel()
    _rf_model_name = MockRFModel.name
    _rf_loaded_at  = datetime.now(timezone.utc).isoformat()


def _load_image_model_sync(checkpoint_path: str):
    """Load (or reload) the UltrasoundLocalizer from a checkpoint. Thread-safe caller
    must hold _model_lock before calling this."""
    import torch
    from model import UltrasoundLocalizer
    from predict import load_model, get_transform

    global _image_model, _image_transform, _image_device, _image_checkpoint, _image_loaded_at

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    model     = load_model(checkpoint_path, device)
    transform = get_transform(224)

    _image_model      = model
    _image_transform  = transform
    _image_device     = device
    _image_checkpoint = checkpoint_path
    _image_loaded_at  = datetime.now(timezone.utc).isoformat()
    print(f"Image model loaded: {checkpoint_path} on {device}")


# ── Request / response schemas ────────────────────────────────────────────────

class PredictRFRequest(BaseModel):
    rf_data:  list[float]
    metadata: Optional[dict] = None


class PredictRFResponse(BaseModel):
    cisterna_magna_detected: bool
    confidence:              float
    depth_mm:                float
    tilt_x_deg:              float
    tilt_y_deg:              float
    classification:          str
    timestamp:               str


class PredictImageResponse(BaseModel):
    x:         float
    y:         float
    timestamp: str


class LoadImageModelRequest(BaseModel):
    checkpoint: str   # absolute or relative path to .pt file

def _slice_confidence(pred) -> float:
    """
    Extract a scalar 0-1 confidence from a model output tensor.

    If your model has a classification head as a third output (raw logit
    for the "target visible" class), uncomment the first branch — that will
    be used automatically once pred.shape[0] >= 3.

    Default fallback: proximity to the origin.  The image model predicts
    (x, y) tilt angles; a smaller displacement means the probe is more
    on-target, so confidence = 1 / (1 + ||pred||).
    """
    import torch
    if pred.shape[0] >= 3:
        return float(torch.sigmoid(pred[2]).item())
    norm = float(pred[:2].norm().item())
    return 1.0 / (1.0 + norm)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/angles")
async def get_angles():
    return JSONResponse(content=latest_angles)


@app.get("/model/status")
async def model_status():
    return JSONResponse(content={
        "image": {
            "loaded":     _image_model is not None,
            "checkpoint": _image_checkpoint,
            "device":     str(_image_device) if _image_device else None,
            "loaded_at":  _image_loaded_at,
        },
        "rf": {
            "loaded":     _rf_model is not None,
            "name":       _rf_model_name,
            "loaded_at":  _rf_loaded_at,
        },
    })


@app.post("/model/image/load")
async def hot_swap_image_model(body: LoadImageModelRequest):
    """Hot-swap the image model checkpoint without restarting the server."""
    path = body.checkpoint
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {path}")
    async with _model_lock:
        try:
            _load_image_model_sync(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "checkpoint": path, "loaded_at": _image_loaded_at}


@app.post("/predict/rf", response_model=PredictRFResponse)
async def predict_rf(payload: PredictRFRequest):
    if not payload.rf_data:
        raise HTTPException(status_code=422, detail="rf_data must not be empty.")
    rf_array = np.array(payload.rf_data, dtype=np.float32)
    result   = _rf_model.predict(rf_array)
    return PredictRFResponse(**result, timestamp=datetime.now(timezone.utc).isoformat())


@app.post("/predict", response_model=PredictRFResponse)
async def predict_rf_alias(payload: PredictRFRequest):
    """Backwards-compatible alias for /predict/rf."""
    return await predict_rf(payload)


@app.post("/predict/image", response_model=PredictImageResponse)
async def predict_image(image: UploadFile = File(...)):
    async with _model_lock:
        model     = _image_model
        transform = _image_transform
        device    = _image_device

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. POST /model/image/load or start with --checkpoint.",
        )

    import torch
    from dataset import denormalize_x, denormalize_y

    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu()

    x = denormalize_x(pred[0:1]).item()
    y = denormalize_y(pred[1:2]).item()
    return PredictImageResponse(x=x, y=y, timestamp=datetime.now(timezone.utc).isoformat())



@app.post("/sweep_predict", response_model=SweepPredictResponse)
async def sweep_predict(body: SweepPredictRequest):
    """
    Run the image model on every PNG in the sweep folder (sorted alphabetically
    = chronologically).  Returns per-slice confidence scores and the index of
    the best slice.

    The C# client sends:  { "folder": "<abs-path-to-braindata>" }
    It expects back:      { "best_slice_index": N,
                            "best_confidence": 0.95,
                            "confidences": [...] }
    """
    import glob as _glob
    import torch

    folder = body.folder
    if not os.path.isdir(folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    png_files = sorted(_glob.glob(os.path.join(folder, "*.png")))
    if not png_files:
        raise HTTPException(status_code=422,
                            detail=f"No PNG files found in: {folder}")

    # Snapshot model refs (don't hold the lock during inference)
    async with _model_lock:
        model     = _image_model
        transform = _image_transform
        device    = _image_device

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not loaded. POST /model/image/load first.",
        )

    confidences: list[float] = []
    for path in png_files:
        try:
            img = Image.open(path).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).squeeze(0).cpu()
            conf = _slice_confidence(pred)
        except Exception as exc:
            print(f"[sweep_predict] Skipping {os.path.basename(path)}: {exc}")
            conf = 0.0
        confidences.append(conf)

    best_idx = int(np.argmax(confidences))
    return SweepPredictResponse(
        best_slice_index=best_idx,
        best_confidence=float(confidences[best_idx]),
        confidences=[float(c) for c in confidences],
    )

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print(f"[WS] Client connected: {ws.client.host}")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data     = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == "angles":
                    latest_angles["x"]          = data.get("x", 0.0)
                    latest_angles["y"]          = data.get("y", 0.0)
                    latest_angles["updated_at"] = datetime.now(timezone.utc).isoformat()
                    # high-frequency — no print
                elif msg_type == "calibrate":
                    latest_angles["calibrated_at"] = datetime.now(timezone.utc).isoformat()
                    print(f"[WS] Calibrated at {latest_angles['calibrated_at']}")
                else:
                    print(f"[WS] Unknown message type: {msg_type}")
            except json.JSONDecodeError:
                print(f"[WS] Invalid JSON: {raw[:120]}")
    except WebSocketDisconnect:
        print("[WS] Client disconnected")


# ── Static file serving (SPA) ─────────────────────────────────────────────────

BUILD_PATH = os.path.join(os.path.dirname(__file__), "dist")
INDEX_HTML = os.path.join(BUILD_PATH, "index.html")

if os.path.exists(INDEX_HTML):
    app.mount("/assets", StaticFiles(directory=os.path.join(BUILD_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(BUILD_PATH, full_path)
        return FileResponse(file_path if os.path.isfile(file_path) else INDEX_HTML)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cisterna Magna Guidance System")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to UltrasoundLocalizer .pt checkpoint (enables /predict/image).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 3000)))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _init_rf_model()

    if args.checkpoint:
        _load_image_model_sync(args.checkpoint)
    else:
        print("No --checkpoint supplied — /predict/image will return 503 until one is loaded.")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
            _s.connect(("8.8.8.8", 80))
            local_ip = _s.getsockname()[0]
    except Exception:
        local_ip = "localhost"

    ssl_context = None
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain("cert.pem", "key.pem")
        scheme = "https"
    else:
        print("No TLS certs found — running over HTTP (sensors won't work on iPhone).")
        print("Run `bash start.sh` once to generate certs, then use python3 app.py directly.")
        scheme = "http"

    print(f"\nOpen on iPhone: {scheme}://{local_ip}:{args.port}\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile="key.pem"  if ssl_context else None,
        ssl_certfile="cert.pem" if ssl_context else None,
        access_log=False,   # suppress per-request lines; errors still appear
    )
