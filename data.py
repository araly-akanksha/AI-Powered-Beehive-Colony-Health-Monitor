# ============================================================
# data.py — Everything about data
#
# SECTIONS (read top to bottom):
#   1. Audio Loading
#   2. Segmentation
#   3. Feature Extraction (Mel-Spectrogram)
#   4. Augmentation
#   5. Dataset Class (BeeDataset)
#   6. DataLoader Factory
#   7. Manifest Builder (build_manifest)
#   8. CLI entry point
#
# CLI usage:
#   python data.py --build --tbon-dir data/raw/tbon --queen-dir data/raw/queen_noqueen
# ============================================================

import os
import argparse
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import config

# Set seeds for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)


# ==============================================================
# SECTION 1 — AUDIO LOADING
# ==============================================================

def load_and_resample(filepath):
    """
    Load a WAV file and resample it to the project sample rate.
    Returns: numpy array (mono waveform), sample_rate
    """
    waveform, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    return waveform, config.SAMPLE_RATE


# ==============================================================
# SECTION 2 — SEGMENTATION
# ==============================================================

def segment_audio(waveform, sr):
    """
    Cut a long audio waveform into fixed-length segments with overlap.

    Example: a 10-sec clip at 16kHz with WINDOW_SEC=2, HOP_SEC=1 →
    produces 9 overlapping 2-sec segments.

    Returns: list of numpy arrays, each of length WINDOW_SAMPLES
    """
    window_samples = config.WINDOW_SAMPLES
    hop_samples    = sr * config.HOP_SEC
    segments       = []

    start = 0
    while start + window_samples <= len(waveform):
        segment = waveform[start : start + window_samples]
        segments.append(segment)
        start += hop_samples

    return segments


# ==============================================================
# SECTION 3 — FEATURE EXTRACTION (Mel-Spectrogram)
# ==============================================================

def to_mel_spectrogram(waveform):
    """
    Convert a raw waveform to a log-scale mel-spectrogram tensor.

    Pipeline:
      waveform → mel filterbank → log amplitude → normalize → tensor

    Output shape: (1, N_MELS, time_frames) — the '1' is the channel
    dimension expected by Conv2d (like a grayscale image).
    """
    # Compute mel-spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y          = waveform,
        sr         = config.SAMPLE_RATE,
        n_fft      = config.N_FFT,
        hop_length = config.HOP_LENGTH,
        n_mels     = config.N_MELS,
        fmax       = config.FMAX,
    )

    # Convert to log scale (dB) — much better for neural nets
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Add channel dimension: (N_MELS, T) → (1, N_MELS, T)
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
    return tensor


# ==============================================================
# SECTION 4 — AUGMENTATION
# ==============================================================

def augment(waveform, sr):
    """
    Apply random audio augmentations to increase dataset diversity.
    Each augmentation is applied with probability AUGMENT_PROB.

    Augmentations:
      - Gaussian noise (simulates hive background noise variation)
      - Time stretch (speed up / slow down slightly)
      - Pitch shift (shift frequency content slightly)

    Returns: augmented waveform (same length as input)
    """
    # Gaussian noise
    if random.random() < config.AUGMENT_PROB:
        # Convert SNR dB → noise amplitude
        signal_power = np.mean(waveform ** 2)
        noise_power  = signal_power / (10 ** (config.NOISE_SNR_DB / 10))
        noise        = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        waveform     = waveform + noise

    # Time stretch
    if random.random() < config.AUGMENT_PROB:
        rate     = random.choice(config.TIME_STRETCH_RATES)
        waveform = librosa.effects.time_stretch(waveform, rate=rate)
        # Re-pad or trim to keep fixed length
        if len(waveform) > config.WINDOW_SAMPLES:
            waveform = waveform[:config.WINDOW_SAMPLES]
        else:
            waveform = np.pad(waveform, (0, config.WINDOW_SAMPLES - len(waveform)))

    # Pitch shift
    if random.random() < config.AUGMENT_PROB:
        steps    = random.choice(config.PITCH_SHIFT_STEPS)
        waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=steps)

    # Guarantee exact WINDOW_SAMPLES length after all augmentations
    if len(waveform) > config.WINDOW_SAMPLES:
        waveform = waveform[:config.WINDOW_SAMPLES]
    elif len(waveform) < config.WINDOW_SAMPLES:
        waveform = np.pad(waveform, (0, config.WINDOW_SAMPLES - len(waveform)))

    return waveform.astype(np.float32)


# ==============================================================
# SECTION 5 — DATASET CLASS
# ==============================================================

class BeeDataset(Dataset):
    """
    PyTorch Dataset for bee audio segments.

    Loads each row from a split CSV (train.csv / val.csv / test.csv),
    reads the audio, segments it on-the-fly, and returns a
    mel-spectrogram tensor + integer label.

    Args:
        csv_path  (str):  Path to train.csv / val.csv / test.csv
        augment   (bool): Whether to apply augmentation (True for train only)
    """

    def __init__(self, csv_path, augment=False, return_domain=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.return_domain = return_domain
        self.samples = []  # list of (filepath, label_idx, offset_sec, domain_idx)
        
        train_hives = config.HIVE_SPLITS.get("train", [])
        self.hive_to_domain = {h: i for i, h in enumerate(train_hives)}

        split_name = os.path.basename(csv_path).replace(".csv", "")
        print(f"  Indexing {split_name} ({len(self.df)} files)...", flush=True)

        debug_limit = int(os.environ.get("DEBUG_LIMIT", 0))
        if debug_limit > 0 and len(self.df) > debug_limit:
            self.df = self.df.sample(n=debug_limit, random_state=42)
            print(f"  [DEBUG_LIMIT ON] Restricted to {debug_limit} files.", flush=True)

        for row in self.df.itertuples(index=False):
            filepath   = getattr(row, "filepath", "")
            label_idx  = config.LABEL_MAP.get(getattr(row, "label", ""), -1)
            duration   = getattr(row, "duration_sec", 0)
            hive_id    = str(getattr(row, "hive_id", ""))
            domain_idx = self.hive_to_domain.get(hive_id, 0)
            if label_idx == -1 or duration < config.WINDOW_SEC:
                continue

            # Calculate offsets
            start_sec = 0
            while start_sec + config.WINDOW_SEC <= duration:
                self.samples.append((filepath, label_idx, start_sec, domain_idx))
                start_sec += config.HOP_SEC

        print(f"  -> {len(self.samples)} segments indexed instantly.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, offset_sec, domain = self.samples[idx]
        
        import torchaudio
        
        frame_offset = int(offset_sec * config.SAMPLE_RATE)
        num_frames = int(config.WINDOW_SEC * config.SAMPLE_RATE)
        
        try:
            # torchaudio.load is extremely fast for reading small chunks from disk
            waveform, sr = torchaudio.load(filepath, frame_offset=frame_offset, num_frames=num_frames)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if needed
            if sr != config.SAMPLE_RATE:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, config.SAMPLE_RATE, dtype=waveform.dtype)
                waveform = resampler(waveform)
            waveform = waveform.squeeze().numpy()
        except Exception:
            # Fallback to librosa if torchaudio fails on some formats
            waveform, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, offset=offset_sec, duration=config.WINDOW_SEC, mono=True)

        # Pad if slightly short
        if len(waveform) < config.WINDOW_SAMPLES:
            waveform = np.pad(waveform, (0, config.WINDOW_SAMPLES - len(waveform)))
        elif len(waveform) > config.WINDOW_SAMPLES:
            waveform = waveform[:config.WINDOW_SAMPLES]

        # Apply augmentation only during training
        if self.augment:
            waveform = augment(waveform, config.SAMPLE_RATE)

        # Convert waveform → mel-spectrogram tensor
        spectrogram = to_mel_spectrogram(waveform)

        if self.return_domain:
            return spectrogram, label, domain

        return spectrogram, label

    def get_labels(self):
        """Return list of all labels — used to compute class weights."""
        return [label for _, label, _, _ in self.samples]


# ==============================================================
# SECTION 6 — DATALOADER FACTORY
# ==============================================================

def get_dataloaders(return_domain=False):
    """
    Build train / val / test DataLoaders from the processed CSV splits.

    Key features:
    - Training uses WeightedRandomSampler to handle class imbalance
    - Validation and test use simple sequential loading (no sampling)
    - Augmentation is ON for train, OFF for val/test
    - If return_domain=True, train_loader yields (specs, labels, domains) for DANN

    Returns: (train_loader, val_loader, test_loader)
    """
    train_csv = os.path.join(config.DATA_PROCESSED, "train.csv")
    val_csv   = os.path.join(config.DATA_PROCESSED, "val.csv")
    test_csv  = os.path.join(config.DATA_PROCESSED, "test.csv")

    train_set = BeeDataset(train_csv, augment=True, return_domain=return_domain)
    val_set   = BeeDataset(val_csv,   augment=False)
    test_set  = BeeDataset(test_csv,  augment=False)

    # --- Class-weighted sampler for training ---
    # Gives rare classes (queen_present) an equal chance of being sampled
    labels       = train_set.get_labels()
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    
    num_samples = getattr(config, "SAMPLES_PER_EPOCH", 40000)
    if num_samples <= 0:
        num_samples = len(sample_weights)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = min(num_samples, len(sample_weights)),
        replacement = True,
    )

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train : {len(train_set)} segments | Class counts: {class_counts}")
    print(f"  Val   : {len(val_set)} segments")
    print(f"  Test  : {len(test_set)} segments")

    return train_loader, val_loader, test_loader


# ==============================================================
# SECTION 7 — MANIFEST BUILDER
# ==============================================================

def build_manifest(tbon_dir=None, queen_dir=None, beetogether_dir=None, output_dir=None):
    """
    Scan raw audio directories and build a master CSV manifest.
    Supports TBON, Queen/No-Queen, and BeeTogether (NUHIVE, SBCM, TBON).

    Output: data/processed/manifest.csv with columns:
      filepath     — absolute path to .wav file
      hive_id      — which hive this clip came from
      label        — 'queen_present' or 'queen_absent'
      duration_sec — clip duration
      dataset      — source dataset name
      split        — 'train', 'val', or 'test' (from HIVE_SPLITS in config.py)

    Also writes: train.csv, val.csv, test.csv (filtered by split)
    """
    if tbon_dir is None:
        tbon_dir = config.DATA_RAW_TBON
    if queen_dir is None:
        queen_dir = config.DATA_RAW_QUEEN
    if beetogether_dir is None:
        beetogether_dir = config.DATA_RAW_BEETOGETHER
    if output_dir is None:
        output_dir = config.DATA_PROCESSED

    records = []
    dfs = []

    # --- Parse TBON dataset ---
    if os.path.isdir(tbon_dir):
        print(f"Scanning TBON: {tbon_dir}")
        for root, dirs, files in os.walk(tbon_dir):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                fpath    = os.path.abspath(os.path.join(root, fname))
                hive_id  = _infer_hive_id_tbon(root, fname)
                label    = _infer_label_tbon(root, fname)
                duration = _get_duration(fpath)
                records.append({
                    "filepath":     fpath,
                    "hive_id":      hive_id,
                    "label":        label,
                    "duration_sec": duration,
                    "dataset":      "tbon",
                })
    else:
        print(f"Notice: TBON directory not found: {tbon_dir} -- skipping")

    # --- Parse Queen/No-Queen dataset ---
    if os.path.isdir(queen_dir):
        print(f"Scanning Queen/No-Queen: {queen_dir}")
        for root, dirs, files in os.walk(queen_dir):
            for fname in files:
                if not fname.lower().endswith(".wav"):
                    continue
                fpath    = os.path.abspath(os.path.join(root, fname))
                hive_id  = _infer_hive_id_queen(root, fname)
                label    = _infer_label_queen(root)
                duration = _get_duration(fpath)
                records.append({
                    "filepath":     fpath,
                    "hive_id":      hive_id,
                    "label":        label,
                    "duration_sec": duration,
                    "dataset":      "queen_noqueen",
                })
    else:
        print(f"Notice: Queen/No-Queen directory not found: {queen_dir} -- skipping")

    if records:
        dfs.append(pd.DataFrame(records))

    # --- Parse BeeTogether dataset ---
    if beetogether_dir and os.path.isdir(beetogether_dir):
        print(f"Scanning BeeTogether: {beetogether_dir}")
        for name in ["NUHIVE", "SBCM", "TBON"]:
            h5_path = os.path.join(beetogether_dir, f"{name}.h5")
            if os.path.exists(h5_path):
                try:
                    df_h5 = pd.read_hdf(h5_path, "bee_audio")
                    df_h5 = df_h5[df_h5["queen presence"].notna()].copy()
                    prefix = os.path.abspath(os.path.join(beetogether_dir, name, name))
                    df_h5["filepath"] = prefix + os.sep + df_h5["file name"]
                    df_h5["hive_id"] = f"{name.lower()}_" + df_h5["hive number"].astype(str)
                    df_h5["label"] = np.where(df_h5["queen presence"] == 1, "queen_present", "queen_absent")
                    df_h5["duration_sec"] = 2.0
                    df_h5["dataset"] = f"beetogether_{name.lower()}"
                    dfs.append(df_h5[["filepath", "hive_id", "label", "duration_sec", "dataset"]])
                    print(f"  Indexed BeeTogether/{name}: {len(df_h5)} audio segments")
                except Exception as e:
                    print(f"  Warning: could not read {h5_path}: {e}")
            else:
                print(f"  Notice: {h5_path} not found")
    else:
        print(f"Notice: BeeTogether directory not found: {beetogether_dir} -- skipping")

    if not dfs:
        print("Warning: No audio files found. Make sure datasets are downloaded to data/raw/")
        return

    manifest = pd.concat(dfs, ignore_index=True)

    # --- Assign splits by hive ID ---
    hive_to_split = {}
    for split, hive_ids in config.HIVE_SPLITS.items():
        for h in hive_ids:
            hive_to_split[h] = split

    manifest["split"] = manifest["hive_id"].map(hive_to_split).fillna("unassigned")

    # Save full manifest
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest saved -> {manifest_path}")
    print(manifest.groupby(["split", "label"]).size().to_string())

    # Save individual split CSVs
    for split in ["train", "val", "test"]:
        split_df   = manifest[manifest["split"] == split]
        split_path = os.path.join(output_dir, f"{split}.csv")
        split_df.to_csv(split_path, index=False)
        print(f"  {split}.csv : {len(split_df)} files")

    unassigned = manifest[manifest["split"] == "unassigned"]
    if len(unassigned) > 0:
        print(f"\n! {len(unassigned)} files have unassigned hive IDs.")
        print("  Unique hive IDs found in data:")
        print(" ", manifest["hive_id"].unique())
        print("  -> Update HIVE_SPLITS in config.py with these IDs.")

    return manifest


def _get_duration(filepath):
    """Get audio duration in seconds without loading the full file."""
    try:
        info = sf.info(filepath)
        return round(info.duration, 2)
    except Exception:
        return -1.0


def _infer_hive_id_tbon(root, fname):
    """Infer hive ID from TBON filename prefix."""
    name = fname.lower()
    if name.startswith("cf") or name.startswith("cj") or name.startswith("gh"):
        return name[:5]
    if name.startswith("hive"):
        return name.split("_")[0]
    return "tbon_other"


def _infer_label_tbon(root, fname):
    """
    Infer queen_present / queen_absent label from TBON file path.
    TBON files often have 'queen' or 'no_queen' in the filename or folder.
    """
    text = (root + fname).lower()
    if "no_queen" in text or "noqueen" in text or "absent" in text or "no-queen" in text:
        return "queen_absent"
    elif "queen" in text:
        return "queen_present"
    return "queen_absent"  # TBON default: most clips are normal hive sound


def _infer_hive_id_queen(root, fname):
    """Infer hive ID from Queen/No-Queen filename prefix."""
    name = fname.lower()
    if name.startswith("cf") or name.startswith("cj") or name.startswith("gh"):
        return name[:5]
    if name.startswith("hive"):
        return name.split("_")[0]
    return "queen_other"


def _infer_label_queen(root):
    """
    Infer label from Queen/No-Queen dataset folder name.
    Expects folders named 'queen_present' / 'queen_absent'.
    """
    root_lower = root.lower()
    if "no_queen" in root_lower or "absent" in root_lower or "without" in root_lower:
        return "queen_absent"
    elif "queen" in root_lower or "present" in root_lower or "with" in root_lower:
        return "queen_present"
    return "queen_absent"


# ==============================================================
# SECTION 8 — CLI ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beehive data pipeline")
    parser.add_argument("--build", action="store_true",
                        help="Build manifest CSV from raw audio directories")
    parser.add_argument("--tbon-dir",        type=str, default=config.DATA_RAW_TBON,
                        help="Path to TBON raw audio directory")
    parser.add_argument("--queen-dir",       type=str, default=config.DATA_RAW_QUEEN,
                        help="Path to Queen/No-Queen raw audio directory")
    parser.add_argument("--beetogether-dir", type=str, default=config.DATA_RAW_BEETOGETHER,
                        help="Path to BeeTogether raw audio directory")
    parser.add_argument("--output-dir",      type=str, default=config.DATA_PROCESSED,
                        help="Where to save manifest and split CSVs")
    args = parser.parse_args()

    if args.build:
        print("Building manifest...")
        build_manifest(args.tbon_dir, args.queen_dir, args.beetogether_dir, args.output_dir)
        print("\nDone! Next steps:")
        print("  1. Open notebooks/01_eda.ipynb to explore your data")
        print("  2. Update HIVE_SPLITS in config.py with actual hive IDs")
        print("  3. Re-run: python data.py --build (to regenerate splits)")
    else:
        parser.print_help()
