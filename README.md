# 🐝 Acoustic Hive Sentinel — AI-Powered Beehive Colony Health Monitor

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](YOUR_VERCEL_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/araly-akanksha/Bee-Hive---Deep-Learning-Project)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

**MSc Big Data Analytics — Deep Learning Final Project**

*Non-Invasive Beehive Queen Presence Detection via Domain-Adversarial Neural Networks (DANN)*

</div>

---

## 🌐 Live Demo

> **Try it now — no installation required!**

**👉 [Open the 3D Acoustic Hive Sentinel UI](YOUR_VERCEL_URL)**

The live web dashboard features:
- 🐝 **Interactive 3D Beehive Wireframe** — animates in response to classification confidence
- 🔊 **Real-Time Audio Synthesizer** — hear procedural queen piping & worker bee hum (Web Audio API)
- 📂 **Custom WAV Upload** — upload your own hive recordings for live inference
- 📊 **On-Screen Spectrogram Visualization** — real-time frequency analysis of uploaded audio
- 🌑 **Cyber-Obsidian Dark Theme** — optimized for classroom projectors and tablet displays

---

## 📌 Executive Summary

Honeybee colonies are critical pollinators responsible for over **35% of global food production**. Queen bee loss — if undetected — leads to irreversible colony collapse within 72 hours. Traditional hive inspection requires physically opening the hive, which is labor-intensive, stressful to the bees, and impossible at scale.

This project develops an **end-to-end acoustic monitoring system** that detects queen presence from raw WAV audio files using deep learning — without ever touching the hive.

### 🔑 The Core Research Problem: Cross-Hive Domain Shift
Standard neural networks trained on one set of hives **fail catastrophically** on new hives they've never seen — accuracy drops from 92% → 55% because the model memorizes background hive noise rather than genuine queen piping. This project solves this using **Domain-Adversarial Neural Networks (DANN)**.

---

## 📊 Key Results

| Model | Seen Hive Accuracy | Unseen Hive Accuracy | Generalization Gap | Macro F1 (Unseen) |
|---|:---:|:---:|:---:|:---:|
| Baseline CNN | 92.1% | 55.8% | -36.3% ❌ | 0.512 |
| CRNN + Attention | 95.4% | 70.4% | -25.0% ⚠️ | 0.684 |
| **DANN (Ours)** | **97.4%** | **94.2%** | **-3.2% ✅** | **0.938** |

> **DANN reduces the cross-hive generalization gap by 11× — from 36.3 percentage points to just 3.2 percentage points.**

---

## 🧠 How It Works

### 1. 📂 Dataset — 19 Hives, 26.7 Hours of Audio

- **Sources Combined:** Queen/No-Queen (Kaggle) + TBON Field Archive + BEETogether
- **Raw Files:** 8,054 WAV recordings
- **After Windowing:** **88,010 spectrograms** (2.0-second windows, 50% overlap)
- **Strict Hive-ID Splitting** (no data leakage):

| Split | Hives | Hive IDs |
|---|:---:|---|
| **Training** | 12 hives | `nuhive_1`, `sbcm_1`, `sbcm_4`, `tbon_1`, `tbon_3`, `tbon_4`, `tbon_5`, `hive1`, `hive1_12`, `hive1_31`, `cf003`, `queen_other` |
| **Validation** | 3 hives | `sbcm_3`, `tbon_2`, `cj001` |
| **Blind Test** *(completely unseen)* | 4 hives | `nuhive_3`, `sbcm_5`, `hive3`, `tbon_6` |

### 2. 🎵 Feature Extraction — Log-Mel Spectrograms

Each 2-second audio segment is converted to a `[1, 128, 63]` Log-Mel Spectrogram:
- Sample Rate: **16,000 Hz**
- FFT Window: **1,024 samples (64ms)**
- Mel Bins: **128** (covering 0–8,000 Hz)
- Output shape per segment: **`[1, 128, 63]`**

### 3. 🏗️ Three Model Architectures (Ablation Study)

```
Model 1: Baseline CNN (~93K params)
  → 3× Conv2D Blocks → Global Average Pool → Classifier
  → PROBLEM: 36.3% generalization gap to unseen hives

Model 2: CRNN + Temporal Attention (~2.59M params)
  → CNN Backbone → 2-Layer Bi-GRU → Self-Attention → Classifier
  → BETTER: Gap reduced to 25.0%

Model 3: DANN with GRL (~2.82M params) ← OUR INNOVATION
  → CNN + Bi-GRU + Attention → [shared features]
       ├─► Health Classifier (queen present/absent)  ← minimizes L_y
       └─► Gradient Reversal Layer (GRL) → Domain Classifier (hive ID) ← maximizes L_d
  → BEST: Gap compressed to 3.2% | Unseen Accuracy: 94.2%
```

### 4. 🔄 Gradient Reversal Layer (GRL) — The Key Innovation

The GRL is a custom PyTorch `autograd.Function`:
- **Forward pass:** Identity (passes features unchanged)
- **Backward pass:** Multiplies incoming gradient by **−λ** (reverses direction)

This forces the feature extractor to learn representations that are:
- ✅ **Highly discriminative** for queen presence/absence
- ❌ **NOT discriminative** for which hive the recording came from

Dynamic adversarial schedule (Ganin et al.):
```
λ(p) = 2 / (1 + exp(−10·p)) − 1     where p = current_step / total_steps ∈ [0, 1]
```

---

## 🗂️ Project File Structure

```
Bee Hive - Deep Learning Project/
│
├── config.py                    # All hyperparameters & hive split definitions
├── data.py                      # Audio ingestion, STFT, BeeDataset, augmentation
├── models.py                    # BaselineCNN, CRNN+Attention, DANN with GRL
├── train.py                     # Training loop, DANN alpha schedule, Macro F1
├── predict.py                   # Single-clip inference
├── requirements.txt             # Python dependencies
│
├── backend/
│   ├── app.py                   # FastAPI REST API (in-memory model cache, 14ms latency)
│   └── samples/                 # 6-second demo WAV clips for live demo
│
├── frontend/
│   ├── index.html               # 🌐 3D Sentinel UI (main Vercel homepage)
│   ├── app.html                 # React SPA dashboard
│   ├── src/                     # React components & Tailwind CSS
│   │   ├── components/
│   │   │   ├── Header.jsx       # Navigation with 3D Sentinel link
│   │   │   └── ...
│   ├── public/
│   │   ├── live.html            # Live waterfall spectrogram UI
│   │   └── beehive_ui.html      # Secondary 3D UI route
│   └── vite.config.js           # Multi-page Vite build config
│
├── vercel.json                  # Vercel deployment configuration
├── beehive_ui.html              # Standalone 3D Sentinel (root reference)
└── checkpoints/                 # Saved PyTorch model weights
    ├── baseline_cnn.pt
    ├── crnn_attention.pt
    └── dann.pt
```

---

## 🚀 Quick Start — Run Locally

### Prerequisites
- Python 3.10+ (tested on 3.13)
- Node.js 18+ (for frontend)

### 1. Clone & Install Python Dependencies
```bash
git clone https://github.com/araly-akanksha/Bee-Hive---Deep-Learning-Project.git
cd Bee-Hive---Deep-Learning-Project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
```bash
cd backend
uvicorn app:app --reload --port 8000
```
- 📖 API Docs: `http://localhost:8000/docs`
- 💓 Health Check: `http://localhost:8000/health`

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` — or just use the **[Live Vercel Demo](YOUR_VERCEL_URL)** instead!

---

## 📡 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Liveness check & model cache status |
| `POST` | `/predict?model=dann` | Upload WAV → get queen presence prediction (14ms latency) |
| `GET` | `/api/samples` | List curated benchmark demo clips |
| `POST` | `/api/analyze` | Multi-model evaluation (CNN + CRNN + DANN) with DSP metrics |

**Example inference call:**
```bash
curl -X POST "http://localhost:8000/predict?model=dann" \
     -F "file=@my_hive_recording.wav"

# Response:
# {"prediction": "queen_present", "confidence": 0.962, "model": "dann", "latency_ms": 14.1}
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | Adam | β₁=0.9, β₂=0.999 |
| Learning Rate | 1e-3 | Decayed by ReduceLROnPlateau |
| Batch Size | 32 | WeightedRandomSampler for class balance |
| Max Epochs | 20 | Early stopping patience=5 |
| Samples/Epoch | 40,000 | Balanced (50/50 queen/no-queen) |
| Dropout | 0.3 | Applied after every conv block & GRU layer |
| GRU Hidden Size | 256 | Bidirectional → 512 combined |
| GRU Layers | 2 | With inter-layer dropout |
| Gradient Clipping | 1.0 | Prevents GRU gradient explosion |

```bash
# Train all three models
python train.py --model baseline_cnn
python train.py --model crnn
python train.py --model dann
```

---

## 🌟 What Makes This Project Unique

1. **First DANN applied to beehive acoustics** — novel cross-domain adaptation for apiculture
2. **Strict Hive-ID Splitting** — eliminates the data leakage that plagues most bioacoustic research
3. **IoT Edge-Safe Architecture** — uses `LayerNorm` instead of `BatchNorm` (which crashes at batch_size=1 on IoT devices)
4. **26.7 Hours of Multi-Source Audio** — fused 3 independent datasets across geographic locations
5. **Full Production System** — not just a notebook: FastAPI backend (14ms latency) + 3D web UI on Vercel CDN

---

## 🔬 Academic References

1. Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.* JMLR 17(59).
2. Nolasco et al. (2022). *Towards Detecting Bee Colony Health Status from Audio.* ICASSP 2022.
3. Fanucci et al. (2021). *A Deep Learning Approach for Beehive Monitoring.* IEEE Sensors.
4. McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015.
5. Bahdanau et al. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.

---

## 📜 License & Academic Use

**Academic MSc Project** — Department of Big Data Analytics, 2024–25.  
Open-source for research and educational purposes.  
Please cite this repository if you use this work in your own research.

---

<div align="center">

Made with 🐝 by **Araly Akanksha** | MSc Big Data Analytics  
⭐ Star this repo if you found it helpful!

**[🚀 Try the Live Demo](YOUR_VERCEL_URL)** · **[📂 Browse Code](https://github.com/araly-akanksha/Bee-Hive---Deep-Learning-Project)**

</div>
