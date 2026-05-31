# RFGuide

Guided needle placement system for emergency cisterna magna access and therapeutic brain cooling.

---

## How to Run

### Prerequisites (one-time setup)

**1. mkcert** — generates trusted local HTTPS certs (required for iPhone sensor access)
```bash
brew install mkcert
mkcert -install
```

**2. Python dependencies**
```bash
pip3 install -r requirements.txt
```

**3. Generate certs (one-time, or after switching Wi-Fi networks)**
```bash
mkcert -key-file key.pem -cert-file cert.pem "$(ipconfig getifaddr en0)" localhost 127.0.0.1
```

---

### Run

```bash
python3 app.py --checkpoint modelTrain/checkpoints/best_model.pt
```

The server prints the URL to open on your iPhone, e.g.:

```
Open on iPhone: https://192.168.1.5:3000
```

Both devices must be on the **same Wi-Fi network**. HTTPS is required — browsers block gyroscope/orientation sensors over plain HTTP.

> If you switch Wi-Fi networks, re-run the `mkcert` command above to regenerate certs for your new IP.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/angles` | Latest IMU probe angles |
| `GET` | `/model/status` | Show loaded model and metadata |
| `POST` | `/model/image/load` | Hot-swap image checkpoint at runtime |
| `POST` | `/predict/image` | Single-image inference — returns `(x, y)` tilt in degrees |
| `POST` | `/sweep_predict` | Rank all slices in a sweep by model confidence (best → worst) |
| `POST` | `/predict/chunks` | 3D chunk inference — groups consecutive slices into volumetric chunks |
| `POST` | `/model/chunk/load` | Hot-swap 3D chunk model (stub until 3D model is trained) |
| `WS` | `/ws` | Inbound stream from iPhone — sends angle and calibration events |
| `WS` | `/ws/angles` | Outbound angle stream for consumers (e.g. Windows app) |

### `/sweep_predict`

Accepts a folder of PNG slices or an explicit ordered file list. Runs batched inference (32 images per forward pass) and returns all slices ranked by confidence:

```json
{
  "folder": "/abs/path/to/braindata"
}
```

Response:
```json
{
  "slices": [
    { "index": 294, "filename": "frame_...png", "confidence": 0.9987 },
    ...
  ]
}
```

### `/predict/chunks`

Groups consecutive slices into 3D chunks `[C, D, H, W]` and batches them for a single forward pass per batch. Returns chunks ranked by confidence. Uses a mock model until a real 3D checkpoint is trained and loaded.

```json
{
  "folder": "/abs/path/to/braindata",
  "chunk_size": 16,
  "stride": 8
}
```

### WebSocket — `/ws` (iPhone → server)

```json
{ "type": "angles", "x": 12.5, "y": -3.2 }
{ "type": "calibrate" }
```

### WebSocket — `/ws/angles` (server → consumers)

Pushes `{x, y, updated_at}` to all connected subscribers whenever the probe angle changes. Sends the last known value immediately on connect.

---

## RF Reconstruction

Converts raw RF echo data into a convex B-mode image:

```bash
python3 main.py
```

Input: `ae2RF.txt` — Output: `Fixed_Convex_B-mode_Reconstruction.png` + heatmap. Probe settings in `settings.py`.

---

## Training the Image Model

```bash
cd modelTrain
python3 train.py --image_dir images
# Resume:
python3 train.py --image_dir images --resume checkpoints/best_model.pt
```

---

## Repo Structure

```
app.py               FastAPI server — serves UI + all API endpoints
model.py             EfficientNet regression model
dataset.py           Denormalization utilities
predict.py           Inference helpers (predict_single, predict_batch, predict_3d_chunks)
main.py              RF → B-mode reconstruction
settings.py          Probe parameters
ae2RF.txt            Sample RF data
requirements.txt     Python dependencies
dist/                Built frontend (served as SPA)
modelTrain/
  train.py           Two-phase training script
  test_ws.py         WebSocket test client
  checkpoints/       Saved weights + training history
```
