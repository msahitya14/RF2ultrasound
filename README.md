# RFGuide

Guided needle placement system for emergency cisterna magna access and therapeutic brain cooling.

---

## How to Run

### Prerequisites (one-time setup)

**1. Node.js** (v18+)
```bash
brew install node
```

**2. Python 3.10+** — comes with macOS, or install via `brew install python`

**3. mkcert** — generates trusted local HTTPS certs (required for phone sensor access)
```bash
brew install mkcert
mkcert -install
```

**4. Python dependencies**
```bash
pip3 install -r requirements.txt
```

**5. Node dependencies**
```bash
npm install
```

---

### Run

```bash
bash start.sh
```

That's it. The script will:
1. Detect your Mac's local IP address
2. Generate HTTPS certs (`key.pem` / `cert.pem`) trusted by your devices
3. Build the React frontend into `dist/`
4. Start the FastAPI server at `https://<your-ip>:3000`

**Open on your phone:** the script prints the URL, e.g. `https://192.168.1.5:3000`

> Both devices must be on the **same Wi-Fi network**. HTTPS is required — browsers block gyroscope/orientation sensors over plain HTTP.

---

### Run with the image model

```bash
bash start.sh --checkpoint modelTrain/checkpoints/best_model.pt
```

This enables the `/predict/image` endpoint. Without `--checkpoint`, that endpoint returns 503 but everything else works normally.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/angles` | Latest IMU probe angles |
| `WS` | `/ws` | Real-time angle / calibration stream |
| `POST` | `/predict/rf` | RF inference (mock until real model is wired) |
| `POST` | `/predict/image` | Image model inference (needs `--checkpoint`) |
| `GET` | `/model/status` | Show loaded models and metadata |
| `POST` | `/model/image/load` | Hot-swap image checkpoint at runtime |

**WebSocket message format:**
```json
{ "type": "calibrate" }
{ "type": "angles", "x": 12.5, "y": -3.2 }
```

---

## RF Reconstruction

Converts raw RF echo data into a convex B-mode image:

```bash
python3 main.py
```

Input: `ae2RF.txt` (comma-separated RF samples)
Output: `Fixed_Convex_B-mode_Reconstruction.png` + heatmap

Probe settings are in `settings.py`.

---

## Training the Image Model

```bash
cd modelTrain
python3 train.py --image_dir images
# Resume from checkpoint:
python3 train.py --image_dir images --resume checkpoints/best_model.pt
```

Best val error: ~4.51°. Checkpoint: `modelTrain/checkpoints/best_model.pt`

---

## Repo Structure

```
app.py               FastAPI server — serves SPA + all API endpoints
model.py             EfficientNet regression model
dataset.py           Denorm utilities
predict.py           Inference helpers
main.py              RF → B-mode reconstruction
settings.py          Probe parameters (center freq, sector angle, etc.)
ae2RF.txt            Sample RF data
requirements.txt     Python dependencies
src/                 React frontend source
start.sh             One-command launch
modelTrain/
  train.py           Two-phase training script
  test_ws.py         WebSocket test client
  checkpoints/       Saved weights + training history
```
