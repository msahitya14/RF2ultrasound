# RFGuide

Guided needle placement system for emergency cisterna magna access and therapeutic brain cooling. Targets first responders operating within the 5-minute window before irreversible ischemic brain damage.

---

## Clinical Context

Exsanguination, cardiac arrest, and massive stroke cause cerebral ischemia. Selective brain cooling via cisterna magna access extends the survivability window, but the procedure requires ultrasound-guided needle placement that demands specialist training. RFGuide removes that bottleneck — providing real-time ML-driven guidance from raw RF ultrasound data, no image interpretation required.

---

## System Pipeline

```
USB Ultrasound Probe
       │
       ▼
RF Data Acquisition (vendor driver)
       │
       ▼
RF → B-mode Reconstruction  (main.py)
       │
       ▼
ML Inference  (modelTrain/)
       │
   ┌───┴───────────────┐
   ▼                   ▼
Classification      Regression
(target present?)   (tilt angle: x, y)
```

---

## RF Reconstruction

`main.py` converts raw RF echo data into a convex B-mode image.

**Steps:** bandpass filter → Hilbert envelope → time gain compensation → log compression → Cartesian scan conversion → CLAHE + sharpening

**Probe defaults** (`settings.py`):
```python
center_freq      = 3e6       # Hz
fractional_bw    = 0.6
sector_angle_deg = 70
curvature_radius = 30        # mm
```

**Run:**
```bash
pip install -r requirements.txt
python main.py
```

Input: `ae2RF.txt` (comma-separated RF samples)  
Output: `Fixed_Convex_B-mode_Reconstruction.png` + heatmap

---

## ML Model

**Architecture:** EfficientNet-B0 (ImageNet pretrained) → adaptive pool → dropout → FC regression head  
**Outputs:** normalized (x, y) probe orientation angles  
**Bounds:** x ∈ [−180°, +180°], y ∈ [−90°, +90°]

**Label encoding in filenames:**
```
frame_..._x6_710_y6_463.png    →  x = +6.710°, y = +6.463°
frame_..._xm0_644_ym0_457.png  →  x = −0.644°, y = −0.457°
```

**Training** (two-phase):
```bash
cd modelTrain
pip install -r requirements.txt
python train.py --image_dir images
# Resume:
python train.py --image_dir images --resume checkpoints/best_model.pt
# Resume into fine-tune phase:
python train.py --image_dir images --resume checkpoints/best_model.pt --resume_phase finetune
```

Phase 1 — warm-up: backbone frozen, head trained  
Phase 2 — fine-tune: end-to-end

Best val error: ~4.51°  |  Checkpoint: `checkpoints/best_model.pt` (~50 MB)

**Predict:**
```bash
python predict.py --checkpoint checkpoints/best_model.pt --image path/to/image.png
python predict.py --checkpoint checkpoints/best_model.pt --image_dir test_images/
# Folder output: test_images/predictions.json
```

---

## API Servers

### Flask (image only)
```bash
python app.py --checkpoint checkpoints/best_model.pt --port 5001
curl -X POST http://localhost:5001/predict -F "image=@frame.png"
# → {"x": 1.23, "y": -4.56}
```

### FastAPI (full, recommended)
```bash
python app2.py --checkpoint checkpoints/best_model.pt --port 3000
```

| Endpoint | Method | Description |
|---|---|---|
| `/angles` | GET | Current calibrated angles |
| `/predict/image` | POST | Image model inference |
| `/predict/rf` | POST | RF inference (**mock — replace before prod**) |
| `/ws` | WS | Real-time angle stream |

WebSocket messages:
```json
{ "type": "calibrate" }
{ "type": "angles", "x": 12.5, "y": -3.2 }
```

Test: `python test_ws.py`

**Additional dependencies (not in requirements.txt):**
```bash
pip install flask fastapi uvicorn python-multipart websockets
```

---

## Known Limitations

- `/predict/rf` returns mock data — `MockRFModel.predict()` must be replaced with a real RF model
- Reconstruction quality depends on accurate probe settings in `settings.py`
- Image labels are parsed from filenames — naming accuracy is critical
- `norm_stats.json` is from a deprecated normalization approach; fixed physical bounds are used instead

---

## Repo Structure

```
├── main.py              RF → B-mode reconstruction
├── settings.py          Probe parameters
├── ae2RF.txt            Sample RF data
├── requirements.txt     Reconstruction deps
├── finalise.ipynb       Final RF pipeline (notebook)
├── visualise.ipynb      RF exploration
├── clustering.ipynb     Experimental analysis
└── modelTrain/
    ├── model.py         EfficientNet regression model
    ├── dataset.py       Dataset parser + splits
    ├── train.py         Two-phase training
    ├── predict.py       Single/batch inference
    ├── app.py           Flask API
    ├── app2.py          FastAPI server (recommended)
    ├── server.py        Earlier FastAPI stub
    ├── test_ws.py       WebSocket test client
    └── checkpoints/     Saved weights + history
```
