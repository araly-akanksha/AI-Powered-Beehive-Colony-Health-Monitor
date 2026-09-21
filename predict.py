# ============================================================
# predict.py — Inference on a single audio file
#
# Loads a trained checkpoint, runs the full pipeline on one .wav,
# prints label + confidence, and saves a spectrogram PNG.
# Also used directly by backend/app.py.
#
# Usage:
#   python predict.py --audio clip.wav --checkpoint checkpoints/exp_01_best.pt
# ============================================================

import os
import argparse
import base64
import io
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import config
from data import load_and_resample, segment_audio, to_mel_spectrogram
from models import get_model


def load_checkpoint(checkpoint_path, device):
    """
    Load a saved model checkpoint.
    Returns: (model, model_name)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt       = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "crnn")
    model      = get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    print(f"Loaded: {model_name}  |  Val F1: {ckpt.get('val_f1', 'N/A')}")
    return model, model_name


def predict_audio_with_model(model, audio_path, model_name="model", device=None):
    """
    Run inference on a single .wav using an already loaded model in memory.
    Bypasses disk checkpoint loading for lightning-fast inference.
    """
    if device is None:
        device = next(model.parameters()).device

    # Load and segment the audio
    waveform, sr = load_and_resample(audio_path)
    segments     = segment_audio(waveform, sr)

    if not segments:
        raise ValueError(f"Audio file too short (< {config.WINDOW_SEC}s): {audio_path}")

    # Run inference on each segment, then average the logits
    all_logits = []
    with torch.no_grad():
        for seg in segments:
            spec   = to_mel_spectrogram(seg).unsqueeze(0).to(device)  # (1,1,N_MELS,T)
            logits = model(spec)                                        # (1, num_classes)
            all_logits.append(logits.cpu())

    # Average over all segments (ensemble)
    avg_logits    = torch.stack(all_logits).mean(dim=0)
    probabilities = torch.softmax(avg_logits, dim=1).squeeze(0).numpy()

    pred_idx      = int(probabilities.argmax())
    pred_label    = config.IDX_TO_LABEL[pred_idx]
    confidence    = float(probabilities[pred_idx])

    # Generate spectrogram visualization of the first segment
    spec_b64 = _spectrogram_to_base64(segments[0])

    return {
        "label":            pred_label,
        "confidence":       round(confidence, 4),
        "all_confidences":  {config.IDX_TO_LABEL[i]: round(float(p), 4)
                             for i, p in enumerate(probabilities)},
        "n_segments":       len(segments),
        "spectrogram_b64":  spec_b64,
        "model_name":       model_name,
    }


def predict_audio(audio_path, checkpoint_path):
    """
    Run inference on a single .wav file from a checkpoint path on disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_checkpoint(checkpoint_path, device)
    return predict_audio_with_model(model, audio_path, model_name=model_name, device=device)


def _spectrogram_to_base64(waveform):
    """
    Generate a mel-spectrogram image and return it as a base64 PNG string.
    This lets the frontend display the spectrogram without saving a file.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y          = waveform,
        sr         = config.SAMPLE_RATE,
        n_fft      = config.N_FFT,
        hop_length = config.HOP_LENGTH,
        n_mels     = config.N_MELS,
        fmax       = config.FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = ax.imshow(mel_db, aspect="auto", origin="lower",
                    cmap="magma", vmin=-80, vmax=0)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel frequency bins")
    ax.set_title("Mel-Spectrogram (what the model sees)")
    plt.colorbar(img, ax=ax, label="dB")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def save_spectrogram_png(audio_path, output_path=None):
    """
    Save a mel-spectrogram PNG for a given audio file.
    Useful for generating example images for the slide deck.
    """
    waveform, sr = load_and_resample(audio_path)
    segments     = segment_audio(waveform, sr)
    if not segments:
        return

    b64 = _spectrogram_to_base64(segments[0])
    png_data = base64.b64decode(b64)

    if output_path is None:
        output_path = os.path.splitext(audio_path)[0] + "_spectrogram.png"

    with open(output_path, "wb") as f:
        f.write(png_data)
    print(f"Spectrogram saved -> {output_path}")


# ==============================================================
# CLI ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bee hive health from audio")
    parser.add_argument("--audio",      type=str, required=True,
                        help="Path to .wav audio file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--save-spec",  action="store_true",
                        help="Save mel-spectrogram as PNG alongside the audio file")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Beehive Prediction")
    print("=" * 50)

    result = predict_audio(args.audio, args.checkpoint)

    print(f"\n  File       : {os.path.basename(args.audio)}")
    print(f"  Prediction : {result['label'].upper().replace('_', ' ')}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"\n  Per-class probabilities:")
    for label, prob in result["all_confidences"].items():
        bar = "█" * int(prob * 30)
        print(f"    {label:20} {prob*100:5.1f}%  {bar}")
    print(f"\n  Segments analysed : {result['n_segments']}")
    print(f"  Model             : {result['model_name']}")

    if args.save_spec:
        save_spectrogram_png(args.audio)

    print("\nDone")
