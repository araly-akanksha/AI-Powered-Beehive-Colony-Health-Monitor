# 🐝 Beehive Health Monitor — Domain-Invariant Deep Learning

> **MSc Big Data Analytics | Final Project**  
> **Cross-Hive Generalization via Acoustic Bio-Surveillance & Domain Adversarial Neural Networks (DANN)**

---

## 📌 Executive Summary

Colony health monitoring through acoustic surveillance provides an early non-invasive warning system for queenlessness and hive distress. However, real-world deployment faces a fundamental machine learning hurdle: **acoustic domain shift**. Background hive reverberations, wooden vs. plastic hive box resonances, and diverse microphone hardware (e.g., IoT MEMS vs. studio dynamic mics) cause standard deep learning models to drastically overfit to their training apiary.

This project designs, trains, and benchmark-tests a **3-experiment sequence** across **403,890 audio files from 19 physical hives** (BeeTogether, NUHIVE, SBCM, TBON) to bridge the cross-hive generalization gap.

---

## 🔬 The 3-Experiment Research Progression

| Experiment | Architecture | Key Feature | Within-Hive Acc | Unseen Hive Acc | Generalization Gap | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| **Exp 1: Baseline** | `BaselineCNN` (93k params) | 4-Block Conv2D + Global Average Pooling | **94.8%** | **54.7%** | -40.1% | Overfit to box resonance |
| **Exp 2: Temporal** | `CRNN` (2.59M params) | Conv2D + 2-Layer Bi-GRU + Temporal Attention | **95.2%** | **63.5%** | -31.7% | +8.85% gain via piping rhythm |
| **Exp 3: SOTA Domain Adaptation** | `DANN` (2.65M params) | CRNN Backbone + Gradient Reversal Layer (GRL) | **95.7%** | **64.9%** (Val F1: 0.699) | **-26.6%** | Eliminates hive coloration |

---

## 📊 Master Dataset Structure

Trained on the **BeeTogether Master Dataset** across 19 physical hives:
- **Master Manifest:** 403,890 segments (16.0 kHz mono, 2-second windows with 50% hop).
- **Strict Hive-Disjoint Splits:**
  - **Train (11 Hives, 262,653 files):** `nuhive_1`, `sbcm_1`, `sbcm_4`, `tbon_1`, `tbon_3`, `tbon_4`, `tbon_5`, `hive1`, `cf003`, `queen_other`.
  - **Validation (3 Hives, 38,920 files):** `sbcm_3`, `tbon_2`, `cj001`.
  - **Test (4 Unseen Hives, 102,306 files):** `nuhive_3`, `sbcm_5`, `hive3`, `tbon_6`.

---

## 🏗️ Project Architecture

```
├── config.py              # Central hyperparameters & hive split definitions
├── data.py                # Fast audio segmentation, librosa DSP, BeeDataset
├── models.py              # BaselineCNN, CRNN (Bi-GRU + Attention), DANN (GRL)
├── train.py               # Training loop with Macro F1, DANN alpha schedule
├── predict.py             # Single-clip inference and in-memory model execution
├── beehive_ui.html        # Standalone 60 FPS waterfall & Web Audio synthesizer
│
├── backend/
│   ├── app.py             # FastAPI backend with multi-model caching & DSP metrics
│   └── samples/           # Curated 6-second demo clips (187 KB each)
│
├── frontend/              # Vite + React production dashboard
│   ├── src/               # UI components, model comparison cards, bioacoustic metrics
│   └── public/
│       ├── live.html      # 60 FPS live waterfall sentinel page
│       └── samples/       # Offline / Vercel demo clips
│
└── checkpoints/           # Saved PyTorch model weights (exp_01, exp_02, exp_03)
```

---

## 🚀 Quick Start & Installation

### 1. Python Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the In-Memory FastAPI Backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 3. Run the Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📡 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness check and model cache status |
| `GET` | `/api/samples` | List curated Demo Day benchmark clips |
| `GET` | `/api/samples/audio/{name}` | Stream lightweight demo WAV audio files |
| `POST` | `/api/analyze` | Multi-model evaluation (CNN + CRNN + DANN) with Librosa bioacoustic DSP metrics |
| `POST` | `/predict` | Single-model inference on uploaded `.wav` file |

---

## 🌟 Live Sentinel Showcase (`/live.html`)

For presentations and live demonstrations, the project includes an interactive Sentinel UI (`/live.html`):
- **60 FPS Waterfall Spectrogram:** Continuous real-time mel-spectrogram waterfall canvas.
- **Web Audio Synthesizer:** Authentic procedural audio synthesis of queen piping pulses (~410 Hz) and colony worker drone buzz without requiring external files.
- **Dynamic 34-Band Equalizer:** Live frequency resonance visualization.
- **Cross-Hive Benchmark Presets:** One-click instant demonstration comparing Baseline CNN collapse vs. DANN domain invariance on unseen test colonies.

---

## 📜 License
Academic MSc Project — Department of Big Data Analytics.
