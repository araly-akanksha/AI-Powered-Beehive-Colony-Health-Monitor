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
from fastapi.responses import JSONResponse

# Allow importing predict.py and config.py from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from predict import predict_audio

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

# Path to the best trained checkpoint (update after training)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints", "exp_01_best.pt"
)

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
    clip_path = os.path.join(SAMPLES_DIR, clip_name)

    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail=f"Sample '{clip_name}' not found. "
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
