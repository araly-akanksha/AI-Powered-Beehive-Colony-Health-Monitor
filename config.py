# ============================================================
# config.py — ALL project hyperparameters in one place
#
# HOW TO USE: Import this file anywhere in the project:
#   import config
#   sr = config.SAMPLE_RATE
# ============================================================

import os

# ----------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------
# AUDIO SETTINGS
# ----------------------------------------------------------
SAMPLE_RATE    = 16000   # Hz — sufficient for bee acoustics (queen piping ~500-2000 Hz)
WINDOW_SEC     = 2       # seconds per audio segment
HOP_SEC        = 1       # hop between segments (50% overlap → more training data)
N_FFT          = 1024    # FFT window size for spectrogram
HOP_LENGTH     = 512     # spectrogram hop length
N_MELS         = 128     # number of mel frequency bins
FMAX           = 8000    # max frequency (Hz) — covers all bee sounds

# Derived constant
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC   # = 32000 samples per segment

# ----------------------------------------------------------
# MODEL SETTINGS
# ----------------------------------------------------------
NUM_CLASSES    = 2       # queen_present=0, queen_absent=1
DROPOUT        = 0.3
# CNN
CNN_CHANNELS   = [32, 64, 128]  # feature maps per conv block
# CRNN (extends CNN)
GRU_HIDDEN     = 256
GRU_LAYERS     = 2
BIDIRECTIONAL  = True

# ----------------------------------------------------------
# TRAINING SETTINGS
# ----------------------------------------------------------
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
EPOCHS         = 50
PATIENCE       = 10      # early stopping patience (epochs without val improvement)
LR_PATIENCE    = 5       # ReduceLROnPlateau patience

# ----------------------------------------------------------
# CLASS LABELS
# ----------------------------------------------------------
# These must match what build_manifest() assigns in data.py
LABEL_MAP = {
    "queen_present": 0,
    "queen_absent":  1,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ----------------------------------------------------------
# HIVE SPLITS — NEVER SPLIT BY RANDOM SEGMENT
#
# Always split by hive ID so the model is tested on fully
# unseen hives. This is the core scientific point of the project.
#
# ⚠ Fill in actual hive IDs after running EDA (notebook 01).
#   Placeholders below — update after you see manifest.csv.
# The cross-hive generalization logic hinges on these splits!
HIVE_SPLITS = {
    # Hives the model sees during training
    "train": ["cf003", "hive1", "hive1 12", "hive1 31", "queen_other"],
    # Held out for hyperparameter tuning
    "val":   ["cj001"],
    # Fully unseen during training — testing the generalization gap!
    "test":  ["hive3"]
}

# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_TBON   = os.path.join(ROOT_DIR, "data", "raw", "tbon")
DATA_RAW_QUEEN  = os.path.join(ROOT_DIR, "data", "raw", "queen_noqueen")
DATA_PROCESSED  = os.path.join(ROOT_DIR, "data", "processed")
MANIFEST_CSV    = os.path.join(DATA_PROCESSED, "manifest.csv")
CHECKPOINT_DIR  = os.path.join(ROOT_DIR, "checkpoints")

# Create directories if they don't exist yet
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------------------------------------
# AUGMENTATION SETTINGS
# ----------------------------------------------------------
AUGMENT_PROB       = 0.5    # probability of applying augmentation per sample
NOISE_SNR_DB       = 20     # signal-to-noise ratio for Gaussian noise
TIME_STRETCH_RATES = [0.9, 1.1]   # slow down or speed up slightly
PITCH_SHIFT_STEPS  = [-2, -1, 1, 2]  # semitones

# ----------------------------------------------------------
# QUICK SANITY CHECK
# Run: python config.py
# ----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Beehive Project — Config Check")
    print("=" * 50)
    print(f"  SAMPLE_RATE   : {SAMPLE_RATE} Hz")
    print(f"  WINDOW_SEC    : {WINDOW_SEC} sec  ({WINDOW_SAMPLES} samples)")
    print(f"  N_MELS        : {N_MELS}")
    print(f"  NUM_CLASSES   : {NUM_CLASSES}  {list(LABEL_MAP.keys())}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print(f"  EPOCHS        : {EPOCHS}")
    print(f"  Train hives   : {HIVE_SPLITS['train']}")
    print(f"  Val hives     : {HIVE_SPLITS['val']}")
    print(f"  Test hives    : {HIVE_SPLITS['test']}  <- unseen!")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print("=" * 50)
    print("Config OK -- all settings loaded.")
