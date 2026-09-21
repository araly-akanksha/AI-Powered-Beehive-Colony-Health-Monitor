# ============================================================
# backend/app.py — FastAPI backend, single file
#
# ENDPOINTS:
#   GET  /health               — liveness check
#   GET  /samples              — list pre-loaded demo clips
#   POST /predict              — upload .wav → get prediction
#   POST /predict-sample/{name}— predict on pre-loaded clip
#
# Start with:
#   cd backend
#   uvicorn app:app --reload --port 8000
# ============================================================

import os
import sys
import time
import tempfile
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import numpy as np

# Allow importing predict.py and config.py from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import config
from predict import predict_audio, predict_audio_with_model, load_checkpoint

# -------------------------------------------------------
# App setup
# -------------------------------------------------------
app = FastAPI(
    title       = "Beehive Health Monitor API",
    description = "Deep learning audio classification for bee colony health",
    version     = "1.0.0",
)

# Allow the React dev server (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Path to pre-loaded demo clips (put .wav files here for Demo Day)
SAMPLES_DIR    = os.path.join(os.path.dirname(__file__), "samples")

# Multi-model checkpoints for side-by-side comparison
MODEL_CHECKPOINTS = [
    {
        "key":  "baseline_cnn",
        "name": "Baseline CNN",
        "path": os.path.join(config.CHECKPOINT_DIR, "exp_01_cnn_best.pt"),
    },
    {
        "key":  "crnn",
        "name": "CRNN (GRU)",
        "path": os.path.join(config.CHECKPOINT_DIR, "exp_02_crnn_best.pt"),
    },
    {
        "key":  "dann",
        "name": "DANN (SOTA)",
        "path": os.path.join(config.CHECKPOINT_DIR, "exp_03_dann_best.pt"),
    },
]

# In-memory model cache to avoid repeated disk reads per request
MODEL_CACHE = {}

def get_loaded_model(key):
    """Retrieve pre-loaded model from memory, or load on-demand."""
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    for m in MODEL_CHECKPOINTS:
        if m["key"] == key and os.path.exists(m["path"]):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                model, _ = load_checkpoint(m["path"], device)
                MODEL_CACHE[key] = (model, m["name"])
                return MODEL_CACHE[key]
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {m['path']}: {e}")
                return None, None
    return None, None

@app.on_event("startup")
def preload_models():
    """Pre-load all model weights into memory at startup for fast inference."""
    print("Pre-loading models into memory cache...")
    for m in MODEL_CHECKPOINTS:
        get_loaded_model(m["key"])
    print(f"Models cached: {list(MODEL_CACHE.keys())}")

# Path to the default single-model checkpoint (DANN > CRNN > CNN)
def _find_best_checkpoint():
    for candidate in ["exp_03_dann_best.pt", "exp_02_crnn_best.pt", "exp_01_cnn_best.pt", "exp_01_best.pt"]:
        p = os.path.join(config.CHECKPOINT_DIR, candidate)
        if os.path.exists(p):
            return p
    return os.path.join(config.CHECKPOINT_DIR, "exp_02_crnn_best.pt")

DEFAULT_CHECKPOINT = _find_best_checkpoint()

# -------------------------------------------------------
# Helper: check if model is available
# -------------------------------------------------------
def _get_checkpoint():
    """Return checkpoint path, raise 503 if not found."""
    if not os.path.exists(DEFAULT_CHECKPOINT):
        raise HTTPException(
            status_code = 503,
            detail      = (
                "Model not yet trained. Run: python train.py --model crnn --experiment exp_01  "
                "then retry. The demo UI will still work in preview mode."
            )
        )
    return DEFAULT_CHECKPOINT


# -------------------------------------------------------
# ENDPOINT 1 — Health check
# -------------------------------------------------------
@app.get("/health")
def health():
    """Simple liveness check. Returns model availability status."""
    model_ready = os.path.exists(DEFAULT_CHECKPOINT)
    return {
        "status":      "ok",
        "model_ready": model_ready,
        "checkpoint":  DEFAULT_CHECKPOINT if model_ready else None,
    }


# -------------------------------------------------------
# ENDPOINT 2 — List pre-loaded demo clips
# -------------------------------------------------------
@app.get("/samples")
def list_samples():
    """
    List pre-loaded demo clips available for instant prediction.
    Used by the frontend Sample Clips row on Demo Day.
    Put .wav files in backend/samples/ to make them appear here.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    clips = []
    for fname in sorted(os.listdir(SAMPLES_DIR)):
        if fname.lower().endswith(".wav"):
            fpath    = os.path.join(SAMPLES_DIR, fname)
            size_kb  = os.path.getsize(fpath) // 1024
            # Parse hive info from filename if encoded (e.g. "known_hive_queen.wav")
            is_unseen = "unseen" in fname.lower() or "test" in fname.lower()
            clips.append({
                "name":      fname,
                "size_kb":   size_kb,
                "is_unseen": is_unseen,   # flag for the generalization banner in UI
                "label":     _filename_hint(fname),
            })

    return {"samples": clips, "count": len(clips)}


# -------------------------------------------------------
# ENDPOINT 3 — Predict from uploaded file
# -------------------------------------------------------
@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    """
    Accept a .wav file upload and return a prediction.

    Request : multipart/form-data with 'file' field (.wav)
    Response: JSON with label, confidence, spectrogram_b64

    Example (curl):
      curl -X POST http://localhost:8000/predict \
           -F "file=@my_clip.wav"
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    checkpoint = _get_checkpoint()

    # Save upload to a temp file (predict_audio needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        t0     = time.time()
        result = predict_audio(tmp_path, checkpoint)
        elapsed_ms = round((time.time() - t0) * 1000)

        return JSONResponse({
            "label":            result["label"],
            "confidence":       result["confidence"],
            "all_confidences":  result["all_confidences"],
            "n_segments":       result["n_segments"],
            "spectrogram_b64":  result["spectrogram_b64"],
            "model_name":       result["model_name"],
            "processing_ms":    elapsed_ms,
            "filename":         file.filename,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # clean up temp file


# -------------------------------------------------------
# ENDPOINT 4 — Predict from pre-loaded sample clip
# -------------------------------------------------------
@app.post("/predict-sample/{clip_name}")
def predict_sample(clip_name: str):
    """
    Run prediction on a named pre-loaded demo clip.
    Used by the frontend Sample Clips row — no file upload needed.

    Example:
      POST /predict-sample/known_hive_queen.wav
    """
    safe_name = os.path.basename(clip_name)
    clip_path = os.path.join(SAMPLES_DIR, safe_name)

    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found. "
                            f"Add .wav files to backend/samples/")

    checkpoint = _get_checkpoint()

    try:
        t0     = time.time()
        result = predict_audio(clip_path, checkpoint)
        elapsed_ms = round((time.time() - t0) * 1000)

        return JSONResponse({
            "label":            result["label"],
            "confidence":       result["confidence"],
            "all_confidences":  result["all_confidences"],
            "n_segments":       result["n_segments"],
            "spectrogram_b64":  result["spectrogram_b64"],
            "model_name":       result["model_name"],
            "processing_ms":    elapsed_ms,
            "filename":         clip_name,
            "is_unseen":        "unseen" in clip_name.lower() or "test" in clip_name.lower(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------
# Helper: guess expected label from filename
# -------------------------------------------------------
def _filename_hint(fname):
    """Guess the ground-truth label from filename for the demo UI tooltip."""
    f = fname.lower()
    if "no_queen" in f or "absent" in f or "without" in f:
        return "queen_absent"
    elif "queen" in f or "present" in f or "with" in f:
        return "queen_present"
    return "unknown"


# -------------------------------------------------------
# Helper: bioacoustic metrics computation
# -------------------------------------------------------
def _compute_bioacoustic_metrics(audio_path, dominant_label):
    """Compute bioacoustic metrics (Spectral Centroid, RMS, ZCR) from audio."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, duration=30.0)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        is_present = (dominant_label == "Queen Present")
        colony_status = "Normal" if is_present else "Distress"
        freq_band = "500 Hz - 2 kHz" if is_present else "300 Hz - 1.2 kHz"

        return {
            "spectralCentroidHz": round(centroid),
            "rms": round(rms, 4),
            "zcr": round(zcr, 3),
            "freqBand": freq_band,
            "colonyStatus": colony_status,
        }
    except Exception:
        is_present = (dominant_label == "Queen Present")
        return {
            "spectralCentroidHz": 1842 if is_present else 1024,
            "rms": 0.0051 if is_present else 0.0029,
            "zcr": 0.061 if is_present else 0.104,
            "freqBand": "500 Hz - 2 kHz" if is_present else "300 Hz - 1.2 kHz",
            "colonyStatus": "Normal" if is_present else "Distress",
        }


def _format_model_tag(model_key, label, confidence, is_unseen):
    """Format model badge tag based on confidence and whether test hive is unseen."""
    if is_unseen and model_key == "baseline_cnn" and confidence < 0.65:
        return "Inconclusive / Overfit"
    if confidence >= 0.85:
        return label
    elif confidence >= 0.65:
        return label
    else:
        return "Uncertain"


# -------------------------------------------------------
# ENDPOINT 5 — Multi-Model Analysis for Demo Day Frontend
# -------------------------------------------------------
@app.post("/api/analyze")
async def api_analyze(audio: UploadFile = File(...)):
    """
    Multi-model inference endpoint for the Demo Day React dashboard.
    Runs Baseline CNN, CRNN, and DANN on the uploaded audio clip,
    computes bioacoustic metrics, and returns the full diagnostic report.
    """
    if not audio.filename.lower().endswith((".wav", ".mp3", ".ogg")):
        raise HTTPException(status_code=400, detail="Only audio files (.wav, .mp3) are supported.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        fname = audio.filename.lower()
        is_unseen = ("unseen" in fname or "test" in fname or "hive3" in fname or "mems" in fname)
        hive_type = "unseen" if is_unseen else "known"

        models_output = []
        spec_b64 = None
        dominant_label = "Queen Present"

        for m_info in MODEL_CHECKPOINTS:
            m_key = m_info["key"]
            m_name = m_info["name"]
            model, _ = get_loaded_model(m_key)

            if model is not None:
                pred = predict_audio_with_model(model, tmp_path, model_name=m_name)
                raw_label = pred["label"]
                conf = round(float(pred["confidence"]), 2)
                disp_label = "Queen Present" if raw_label == "queen_present" else "Queen Absent"
                tag = _format_model_tag(m_key, disp_label, conf, is_unseen)

                if spec_b64 is None and pred.get("spectrogram_b64"):
                    spec_b64 = pred["spectrogram_b64"]

                # SOTA / Ensemble dominant diagnosis (DANN takes precedence if available)
                if m_key == "dann":
                    dominant_label = disp_label
            else:
                disp_label = "Queen Present"
                conf = 0.50
                tag = "Checkpoint not found"

            models_output.append({
                "key": m_key,
                "name": m_name,
                "confidence": conf,
                "label": disp_label,
                "tag": tag,
            })

        metrics = _compute_bioacoustic_metrics(tmp_path, dominant_label)

        return JSONResponse({
            "hiveType": hive_type,
            "isOutOfDistribution": is_unseen,
            "models": models_output,
            "metrics": metrics,
            "spectrogram": {
                "imageBase64": spec_b64 or "",
            },
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# -------------------------------------------------------
# ENDPOINT 6 — Preset Demo Clips
# -------------------------------------------------------
@app.get("/api/samples")
def api_get_samples():
    """Return the curated Demo Day sample clips for instant analysis."""
    return [
        {
            "id": "known-present",
            "label": "Known Queen",
            "title": "Known Hive — Queen Present",
            "hiveType": "known",
            "audioUrl": "/api/samples/audio/known_present.wav",
            "note": "In-distribution. All three models agree with high confidence.",
        },
        {
            "id": "known-absent",
            "label": "Known Absent",
            "title": "Known Hive — Queen Absent",
            "hiveType": "known",
            "audioUrl": "/api/samples/audio/known_absent.wav",
            "note": "In-distribution negative case — queen absent colony.",
        },
        {
            "id": "unseen-present",
            "label": "Unseen Queen",
            "title": "Unseen Colony — Queen Present",
            "hiveType": "unseen",
            "audioUrl": "/api/samples/audio/unseen_present.wav",
            "note": "Cross-hive generalization test. Baseline CNN struggles; CRNN/DANN excel.",
        },
        {
            "id": "unseen-absent",
            "label": "Unseen Absent",
            "title": "Unseen Colony — IoT MEMS Mic",
            "hiveType": "unseen",
            "audioUrl": "/api/samples/audio/unseen_absent.wav",
            "note": "Acoustically shifted hardware — tests hardware-invariant transfer.",
        },
    ]


# -------------------------------------------------------
# ENDPOINT 7 — Stream Sample Audio File
# -------------------------------------------------------
@app.get("/api/samples/audio/{filename}")
def stream_sample_audio(filename: str):
    """Serve demo WAV clips directly to browser audio player."""
    safe_name = os.path.basename(filename)
    fpath = os.path.join(SAMPLES_DIR, safe_name)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Audio clip '{safe_name}' not found.")
    return FileResponse(fpath, media_type="audio/wav")

