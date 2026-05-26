# RFGuide

Guided needle placement system for emergency cisterna magna access and therapeutic brain cooling.

---

## How to Run

### Prerequisites (one-time setup)

**1. Node.js** (v18+)
```bash
brew install node
```

**2. mkcert** — generates trusted local HTTPS certs (required for iPhone sensor access)
```bash
brew install mkcert
mkcert -install
```

**3. Python dependencies**
```bash
pip3 install -r requirements.txt
```

**4. Node dependencies**
```bash
npm install
```

**5. Generate certs and build the frontend (one-time, or after switching Wi-Fi networks)**
```bash
mkcert -key-file key.pem -cert-file cert.pem "$(ipconfig getifaddr en0)" localhost 127.0.0.1
npm run build
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

> If you switch Wi-Fi networks, re-run the `mkcert` + `npm run build` commands above to regenerate certs for your new IP.

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
app.py               FastAPI server — serves SPA + all API endpoints
model.py             EfficientNet regression model
dataset.py           Denorm utilities
predict.py           Inference helpers
main.py              RF → B-mode reconstruction
settings.py          Probe parameters
ae2RF.txt            Sample RF data
requirements.txt     Python dependencies
src/                 React frontend source
modelTrain/
  train.py           Two-phase training script
  test_ws.py         WebSocket test client
  checkpoints/       Saved weights + training history
```


<!-- brew install node mkcert && mkcert -install
pip3 install -r requirements.txt
npm install 
bash start.sh
-->
