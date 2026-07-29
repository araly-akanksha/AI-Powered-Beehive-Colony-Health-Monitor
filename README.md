# 🐝 Beehive Health Monitor — Deep Learning Audio Classification

> MSc Big Data Analytics | External Project  
> Cross-Hive Generalizable Deep Learning for Beehive Health Monitoring via Acoustic Analysis

---

## What This Project Does

Classifies beehive audio to detect **queen presence / absence** using deep learning on mel-spectrograms. The core scientific contribution is measuring the **cross-hive generalization gap** — how accuracy degrades when the model is tested on hives it has never seen during training.

---

## Project Structure

```
├── config.py          ← ALL hyperparameters in one place
├── data.py            ← Audio loading, segmentation, mel-spectrograms, dataset
├── models.py          ← BaselineCNN + CRNN (both in one file for easy comparison)
├── train.py           ← Training loop, evaluation, cross-hive report
├── predict.py         ← Inference on a single .wav file
│
├── backend/
│   └── app.py         ← FastAPI backend (4 endpoints)
│
├── frontend/          ← Vite + React demo UI
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Dataset EDA (Month 1 Week 2)
│   ├── 02_baseline_cnn.ipynb   ← Baseline CNN results (Month 1 Week 3–4)
│   └── 03_crnn_crosshive.ipynb ← CRNN + generalization gap (Month 2 Week 7–8)
│
├── data/
│   ├── raw/tbon/               ← TBON dataset (download from Kaggle)
│   └── raw/queen_noqueen/      ← Queen/No-Queen dataset (download from Kaggle)
│
└── checkpoints/       ← Saved model weights (gitignored)
```

---

## Dataset Setup (Manual — Required Before Training)

Download both datasets from Kaggle and extract them into the correct folders:

| Dataset | Kaggle URL | Extract to |
|---|---|---|
| To Bee or Not to Bee (TBON) | `kaggle.com/datasets/chrisfilo/to-bee-or-no-to-bee` | `data/raw/tbon/` |
| Queen / No-Queen Audio | `kaggle.com/datasets/harshkumar1711/beehive-audio-dataset-with-queen-and-without-queen` | `data/raw/queen_noqueen/` |

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Month-by-Month Workflow

### Month 1 — Foundation

```bash
# Step 1: Verify config
python config.py

# Step 2: Build manifest from raw audio
python data.py --build

# Step 3: Open notebook 01 — EDA, find hive IDs, update HIVE_SPLITS in config.py
jupyter lab notebooks/01_eda.ipynb

# Step 4: Re-build manifest with correct splits
python data.py --build

# Step 5: Train baseline CNN
python train.py --model baseline_cnn --epochs 30 --experiment exp_01

# Step 6: Open notebook 02 — plot results
jupyter lab notebooks/02_baseline_cnn.ipynb
```

### Month 2 — CRNN + Generalization

```bash
# Train CRNN with cross-hive report at the end
python train.py --model crnn --epochs 50 --experiment exp_02 --cross-hive

# Open notebook 03 — the headline result chart
jupyter lab notebooks/03_crnn_crosshive.ipynb
```

### Month 3 — Demo

```bash
# Test inference on a single clip
python predict.py --audio path/to/clip.wav --checkpoint checkpoints/exp_02_best.pt

# Start backend
cd backend
uvicorn app:app --reload --port 8000

# Start frontend (new terminal)
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## Model Architectures

### BaselineCNN
3 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool) → Global Average Pooling → FC  
~300K parameters. Establishes baseline performance.

### CRNN
Same CNN backbone → 2-layer bidirectional GRU → Attention pooling → FC  
Captures temporal buzzing patterns across the 2-second audio window.

---

## Key Design Decisions

| Decision | Value | Rationale |
|---|---|---|
| Sample rate | 16,000 Hz | Queen piping range 500–2000 Hz — no need for higher |
| Window size | 2 seconds | Short enough to capture events, long enough for patterns |
| Mel bins | 128 | Standard for audio classification |
| Split strategy | **By hive ID** | Never random — this is the scientific point |
| Class imbalance | Weighted sampler + loss | Queen clips are rare |
| Augmentation | Noise, time-stretch, pitch-shift | Improves cross-hive robustness |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model availability check |
| GET | `/samples` | List pre-loaded demo clips |
| POST | `/predict` | Upload `.wav` → prediction JSON |
| POST | `/predict-sample/{name}` | Predict on pre-loaded clip |

---

## The Research Finding

The cross-hive generalization gap — the drop in accuracy from known training hives to
fully unseen test hives — is the headline result. A gap of 10–20% is expected, meaningful,
and is an honest scientific contribution. See `notebooks/03_crnn_crosshive.ipynb` for
the full analysis.
