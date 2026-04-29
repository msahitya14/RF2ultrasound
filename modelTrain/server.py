import json
import ssl
import os
import uvicorn
import numpy as np

from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Cisterna Magna Guidance System")

# ── Shared state (mirrors latestAngles in Node version) ───────────────────────

latest_angles: dict = {
    "x": 0.0,
    "y": 0.0,
    "calibrated_at": None,
    "updated_at": None,
}

# ── ML model placeholder ─────────────────────────────────────────────────────
# Replace this stub with your actual model loader, e.g.:
#   import torch; model = torch.load("model.pt")
#   or: from tensorflow import keras; model = keras.models.load_model("model")

class MockModel:
    """Stub – swap out for your real ultrasound RF model."""
    def predict(self, rf_data: np.ndarray) -> dict:
        # Example output shape; adjust to match your model's actual output.
        return {
            "cisterna_magna_detected": bool(np.random.rand() > 0.3),
            "confidence": float(np.random.rand()),
            "depth_mm": float(np.random.uniform(40, 80)),
            "tilt_x_deg": float(np.random.uniform(-15, 15)),
            "tilt_y_deg": float(np.random.uniform(-15, 15)),
            "classification": "cisterna_magna" if np.random.rand() > 0.4 else "other",
        }

model = MockModel()

# ── Request / response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Payload sent to /predict.
    rf_data   – flattened RF A-lines (list of floats) from the ultrasound probe.
    metadata  – optional dict with probe settings, gain, etc.
    """
    rf_data: list[float]
    metadata: Optional[dict] = None


class PredictResponse(BaseModel):
    cisterna_magna_detected: bool
    confidence: float
    depth_mm: float
    tilt_x_deg: float       # regression: how much to tilt probe in X
    tilt_y_deg: float       # regression: how much to tilt probe in Y
    classification: str
    timestamp: str

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/angles")
async def get_angles():
    """Return the latest IMU / gyroscope angles (mirrors Node /angles)."""
    return JSONResponse(content=latest_angles)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """
    Core ML endpoint.

    Accepts raw RF data from the ultrasound probe, runs the model, and
    returns:
      • detection flag + confidence
      • estimated depth to cisterna magna
      • tilt corrections (regression outputs) so the operator knows how
        to reposition the probe / needle
      • tissue classification label
    """
    if not payload.rf_data:
        raise HTTPException(status_code=422, detail="rf_data must not be empty.")

    rf_array = np.array(payload.rf_data, dtype=np.float32)

    # ── Run model ─────────────────────────────────────────────────────────────
    # Replace model.predict() with your real inference call.
    # If your model expects a 2-D array (lines × samples) reshape here:
    #   rf_array = rf_array.reshape(num_lines, samples_per_line)
    result = model.predict(rf_array)

    return PredictResponse(
        **result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

# ── WebSocket (mirrors Node wss on /ws) ──────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client = ws.client.host
    print(f"Client connected from: {client}")
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "angles":
                    latest_angles["x"] = data.get("x", 0.0)
                    latest_angles["y"] = data.get("y", 0.0)
                    latest_angles["updated_at"] = datetime.now(timezone.utc).isoformat()

                elif msg_type == "calibrate":
                    latest_angles["calibrated_at"] = datetime.now(timezone.utc).isoformat()

                print(f"Received: {data}")

            except json.JSONDecodeError:
                print(f"Invalid message: {raw}")

    except WebSocketDisconnect:
        print("Client disconnected")

# ── Static file serving (mirrors Node dist/ serving) ─────────────────────────

BUILD_PATH = os.path.join(os.path.dirname(__file__), "dist")
INDEX_HTML = os.path.join(BUILD_PATH, "index.html")

if os.path.exists(INDEX_HTML):
    app.mount("/assets", StaticFiles(directory=os.path.join(BUILD_PATH, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Let API routes pass through; catch-all serves the SPA shell.
        file_path = os.path.join(BUILD_PATH, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(INDEX_HTML)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = int(os.environ.get("PORT", 3000))

    # TLS – mirrors the key.pem / cert.pem loading in Node
    ssl_context = None
    if os.path.exists("key.pem") and os.path.exists("cert.pem"):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain("cert.pem", "key.pem")
        print(f"Server running on https://{HOST}:{PORT}")
    else:
        print(f"TLS certs not found – running on http://{HOST}:{PORT} (no TLS)")

    print(f"Predict endpoint : /predict  (POST)")
    print(f"Angles endpoint  : /angles   (GET)")
    print(f"WebSocket        : /ws")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        ssl_keyfile="key.pem" if ssl_context else None,
        ssl_certfile="cert.pem" if ssl_context else None,
    )


# Test /angles
# curl http://localhost:3000/angles

# Test /predict
# curl -X POST http://localhost:3000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"rf_data": [0.1, 0.4, 0.9, 0.2, 0.7, 0.3]}'