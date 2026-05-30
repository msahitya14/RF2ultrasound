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
    WS   /ws                  inbound angle / calibration stream (from the iPhone)
    WS   /ws/angles           outbound real-time angle stream (to consumers, e.g. Windows app)
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

# 3D chunk model state (None until trained and loaded via POST /model/chunk/load)
_chunk_model = None


class SweepPredictRequest(BaseModel):
    folder: Optional[str] = None          # scan all *.png in folder (sorted alphabetically)
    files:  Optional[list[str]] = None    # explicit ordered list of absolute PNG paths

class ChunkPredictRequest(BaseModel):
    folder:     Optional[str]       = None  # scan folder for *.png (sorted)
    files:      Optional[list[str]] = None  # explicit ordered slice paths
    chunk_size: int                 = 16    # slices per 3D chunk
    stride:     Optional[int]       = None  # step between chunks; defaults to chunk_size (non-overlapping)

class ChunkResult(BaseModel):
    chunk_index:  int
    start_slice:  int
    end_slice:    int
    confidence:   float
    filenames:    list[str]

class ChunkPredictResponse(BaseModel):
    chunks:     list[ChunkResult]   # sorted best → worst confidence
    model_stub: bool                # True until a real 3D model is loaded

class SliceResult(BaseModel):
    index:      int
    filename:   str
    confidence: float

class SweepPredictResponse(BaseModel):
    slices: list[SliceResult]   # sorted best → worst confidence

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
    Run the image model on a set of PNG slices and return per-slice confidence
    scores plus the index of the best (most on-target) slice.

    Two calling modes:
      { "files": ["abs/path/a.png", "abs/path/b.png", ...] }
          Process exactly these files in the given order (preferred).
          best_slice_index is an index into this list → matches the 3D-view
          slice list built by the C# client during the current session.

      { "folder": "<abs-path>" }
          Scan the folder for *.png (sorted alphabetically).  Legacy mode;
          may include files from previous sessions.

    Confidence proxy (model currently outputs (x, y) tilt angles):
        conf = 1 / (1 + sqrt(x² + y²))
    The slice with the smallest predicted tilt is closest to the target.
    """
    import glob as _glob
    import torch

    # ── Resolve file list ─────────────────────────────────────────────────────
    if body.files is not None:
        # Explicit ordered list from the C# client — use as-is (no sort).
        png_files = [str(p) for p in body.files]
        missing = [p for p in png_files if not os.path.isfile(p)]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"{len(missing)} file(s) not found: {missing[:5]}"
            )
    elif body.folder:
        folder = body.folder
        if not os.path.isdir(folder):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
        png_files = sorted(_glob.glob(os.path.join(folder, "*.png")))
    else:
        raise HTTPException(status_code=422,
                            detail="Provide either 'files' (list of paths) or 'folder'.")

    if not png_files:
        raise HTTPException(status_code=422, detail="No PNG files to process.")

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

    from predict import predict_batch

    CHUNK_SIZE = 32
    confidences: list[float] = []
    for chunk_start in range(0, len(png_files), CHUNK_SIZE):
        chunk = png_files[chunk_start : chunk_start + CHUNK_SIZE]
        try:
            preds = predict_batch(model, transform, chunk, device)
            for pred in preds:
                confidences.append(_slice_confidence(pred))
        except Exception as exc:
            print(f"[sweep_predict] Chunk starting at {chunk_start} failed: {exc}")
            confidences.extend([0.0] * len(chunk))

    ranked = sorted(
        [
            SliceResult(
                index=i,
                filename=os.path.basename(png_files[i]),
                confidence=float(confidences[i]),
            )
            for i in range(len(confidences))
        ],
        key=lambda s: s.confidence,
        reverse=True,
    )
    return SweepPredictResponse(slices=ranked)


@app.post("/predict/chunks", response_model=ChunkPredictResponse)
async def predict_chunks(body: ChunkPredictRequest):
    """
    3D chunk inference — groups consecutive slices into volumetric chunks and
    runs a single forward pass per chunk.

    Each chunk tensor has shape [C, D, H, W] (D = chunk_size slices).
    Chunks are batched into [B, C, D, H, W] for the model forward pass.

    Returns all chunks sorted best → worst confidence.
    model_stub=true means no 3D model is loaded yet — confidences are mock values.
    Swap in a real model via POST /model/chunk/load once trained.
    """
    import glob as _glob
    from predict import predict_3d_chunks

    # ── Resolve file list ─────────────────────────────────────────────────────
    if body.files is not None:
        png_files = [str(p) for p in body.files]
        missing = [p for p in png_files if not os.path.isfile(p)]
        if missing:
            raise HTTPException(status_code=404,
                                detail=f"{len(missing)} file(s) not found: {missing[:5]}")
    elif body.folder:
        if not os.path.isdir(body.folder):
            raise HTTPException(status_code=404, detail=f"Folder not found: {body.folder}")
        png_files = sorted(_glob.glob(os.path.join(body.folder, "*.png")))
    else:
        raise HTTPException(status_code=422, detail="Provide 'files' or 'folder'.")

    if not png_files:
        raise HTTPException(status_code=422, detail="No PNG files to process.")

    chunk_size = body.chunk_size
    stride     = body.stride if body.stride is not None else chunk_size

    if chunk_size < 1 or stride < 1:
        raise HTTPException(status_code=422, detail="chunk_size and stride must be >= 1.")
    if chunk_size > len(png_files):
        raise HTTPException(status_code=422,
                            detail=f"chunk_size ({chunk_size}) exceeds number of slices ({len(png_files)}).")

    # ── Build chunk windows ───────────────────────────────────────────────────
    starts = list(range(0, len(png_files) - chunk_size + 1, stride))
    chunk_windows = [png_files[s : s + chunk_size] for s in starts]

    # ── Snapshot model ref ────────────────────────────────────────────────────
    async with _model_lock:
        model_3d = _chunk_model
        transform = _image_transform
        device    = _image_device

    if transform is None:
        raise HTTPException(status_code=503,
                            detail="Image transform not ready. Load an image model first via POST /model/image/load.")

    # ── Inference ─────────────────────────────────────────────────────────────
    BATCH = 8   # chunks per forward pass (3D tensors are larger than 2D)
    confidences: list[float] = []
    for i in range(0, len(chunk_windows), BATCH):
        batch_chunks = chunk_windows[i : i + BATCH]
        try:
            confs = predict_3d_chunks(model_3d, transform, batch_chunks, device)
            confidences.extend(confs)
        except Exception as exc:
            print(f"[predict_chunks] Batch at {i} failed: {exc}")
            confidences.extend([0.0] * len(batch_chunks))

    # ── Build ranked response ─────────────────────────────────────────────────
    results = [
        ChunkResult(
            chunk_index=i,
            start_slice=starts[i],
            end_slice=starts[i] + chunk_size - 1,
            confidence=confidences[i],
            filenames=[os.path.basename(p) for p in chunk_windows[i]],
        )
        for i in range(len(chunk_windows))
    ]
    ranked = sorted(results, key=lambda c: c.confidence, reverse=True)
    return ChunkPredictResponse(chunks=ranked, model_stub=model_3d is None)


@app.post("/model/chunk/load")
async def load_chunk_model(body: LoadImageModelRequest):
    """
    Hot-swap the 3D chunk model checkpoint at runtime.

    Not yet implemented — define your 3D model class in model_3d.py, then replace
    the 501 below with:
        from model_3d import UltrasoundLocalizer3D
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        m = UltrasoundLocalizer3D(); m.load_state_dict(ckpt["model_state"])
        m.to(device).eval()
        async with _model_lock:
            _chunk_model = m
            _chunk_loaded_at = datetime.now(timezone.utc).isoformat()
        return {"status": "ok", "checkpoint": path, "loaded_at": _chunk_loaded_at}
    """
    raise HTTPException(status_code=501,
                        detail="3D model class not yet defined — train one first.")

# ── WebSocket ─────────────────────────────────────────────────────────────────

class AngleSubscribers:
    """
    Fan-out registry for read-only angle consumers (e.g. the Windows app).

    The iPhone PRODUCES angle samples on /ws; those samples are broadcast to
    every subscriber connected on /ws/angles so consumers get real-time pushes
    instead of polling GET /angles.
    """
    def __init__(self):
        self._subs: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket):
        async with self._lock:
            self._subs.add(ws)

    async def remove(self, ws: WebSocket):
        async with self._lock:
            self._subs.discard(ws)

    async def broadcast(self, message: dict):
        # Snapshot under the lock, send outside it so a slow client can't block others.
        async with self._lock:
            targets = list(self._subs)
        if not targets:
            return
        payload = json.dumps(message)
        dead = []
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._subs.discard(ws)


angle_subscribers = AngleSubscribers()


def _angle_snapshot() -> dict:
    return {
        "x":          latest_angles["x"],
        "y":          latest_angles["y"],
        "updated_at": latest_angles["updated_at"],
    }


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
                    # high-frequency — no print; push to all read-only subscribers
                    await angle_subscribers.broadcast(_angle_snapshot())
                elif msg_type == "calibrate":
                    latest_angles["calibrated_at"] = datetime.now(timezone.utc).isoformat()
                    print(f"[WS] Calibrated at {latest_angles['calibrated_at']}")
                else:
                    print(f"[WS] Unknown message type: {msg_type}")
            except json.JSONDecodeError:
                print(f"[WS] Invalid JSON: {raw[:120]}")
    except WebSocketDisconnect:
        print("[WS] Client disconnected")


@app.websocket("/ws/angles")
async def websocket_angles_subscribe(ws: WebSocket):
    """
    Read-only angle stream for consumers (the Windows app).

    Pushes {x, y, updated_at} to the client whenever the probe angle changes.
    The current value is sent immediately on connect so the client doesn't have
    to wait for the next movement. This endpoint does not expect inbound
    messages — awaiting receive simply lets us detect disconnects promptly.
    """
    await ws.accept()
    await angle_subscribers.add(ws)
    print(f"[WS/angles] Subscriber connected: {ws.client.host}")
    try:
        await ws.send_text(json.dumps(_angle_snapshot()))   # prime with last known angles
        while True:
            await ws.receive_text()   # blocks until the client disconnects
    except WebSocketDisconnect:
        print("[WS/angles] Subscriber disconnected")
    finally:
        await angle_subscribers.remove(ws)


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
