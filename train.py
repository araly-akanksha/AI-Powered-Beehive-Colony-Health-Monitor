# ============================================================
# train.py — Training loop, evaluation, cross-hive analysis
#
# SECTIONS:
#   1. Setup (device, loss, optimizer)
#   2. train_one_epoch()  — one pass through training data
#   3. evaluate()         — accuracy, F1, confusion matrix
#   4. cross_hive_report()— the headline result: within vs cross-hive gap
#   5. main()             — CLI entry point
#
# Usage:
#   python train.py --model baseline_cnn --epochs 30 --experiment exp_01
#   python train.py --model crnn --epochs 50 --experiment exp_02
# ============================================================

import os
import csv
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
import config
from models import get_model
from data import get_dataloaders, BeeDataset

# Try to import W&B; fall back to CSV logging if not configured
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==============================================================
# SECTION 1 — SETUP HELPERS
# ==============================================================

def get_device():
    """Return GPU if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def get_loss_fn(train_loader):
    """
    Compute class-weighted CrossEntropyLoss.

    Why: If dataset has 80% queen_absent and 20% queen_present,
    the model can reach 80% accuracy by predicting always absent.
    Class weighting penalizes the model more for rare-class mistakes.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    class_counts = np.bincount(all_labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES

    print(f"Class counts  : {class_counts}")
    print(f"Class weights : {class_weights.round(3)}")

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights_tensor)


# ==============================================================
# SECTION 2 — TRAINING: ONE EPOCH
# ==============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full pass through the training data.

    Returns: dict with 'loss' and 'accuracy'
    """
    model.train()
    total_loss    = 0.0
    all_preds     = []
    all_labels    = []

    for batch_idx, (specs, labels) in enumerate(loader):
        specs  = specs.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(specs)              # (batch, num_classes)
        loss   = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        # Accumulate metrics
        total_loss  += loss.item()
        preds        = logits.argmax(dim=1).cpu().numpy()
        all_preds   .extend(preds)
        all_labels  .extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return {"loss": avg_loss, "accuracy": accuracy}


# ==============================================================
# SECTION 3 — EVALUATION
# ==============================================================

def evaluate(model, loader, device, split_name="val"):
    """
    Evaluate model on a DataLoader and return full metrics.

    Returns: dict with accuracy, precision, recall, f1, confusion_matrix
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for specs, labels in loader:
            specs  = specs.to(device)
            logits = model(specs)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds .extend(preds)
            all_labels.extend(labels.numpy())

    acc   = accuracy_score(all_labels, all_preds)
    prec  = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec   = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1    = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm    = confusion_matrix(all_labels, all_preds)

    print(f"\n[{split_name.upper()}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "confusion_matrix": cm.tolist(),
        "all_preds":        all_preds,
        "all_labels":       all_labels,
    }


# ==============================================================
# SECTION 4 — CROSS-HIVE REPORT (The Headline Result)
# ==============================================================

def cross_hive_report(model, device, manifest_csv=None):
    """
    Evaluate model separately on each hive and print the
    within-hive vs cross-hive generalization table.

    This is the core research contribution of this project.
    The gap between known-hive and unseen-hive accuracy is what
    you present on Demo Day as your headline result.

    Returns: DataFrame with per-hive metrics
    """
    if manifest_csv is None:
        manifest_csv = config.MANIFEST_CSV

    if not os.path.exists(manifest_csv):
        print("⚠ manifest.csv not found. Run: python data.py --build first.")
        return None

    manifest = pd.read_csv(manifest_csv)
    model.eval()

    results = []

    for hive_id in manifest["hive_id"].unique():
        hive_rows = manifest[manifest["hive_id"] == hive_id]
        split     = hive_rows["split"].iloc[0]

        # Write a temp CSV for this hive to reuse BeeDataset
        temp_csv = os.path.join(config.DATA_PROCESSED, f"_temp_{hive_id}.csv")
        hive_rows.to_csv(temp_csv, index=False)

        try:
            hive_dataset = BeeDataset(temp_csv, augment=False)
            if len(hive_dataset) == 0:
                continue
            hive_loader = torch.utils.data.DataLoader(
                hive_dataset, batch_size=config.BATCH_SIZE, shuffle=False
            )
            metrics = evaluate(model, hive_loader, device, split_name=hive_id)
            results.append({
                "hive_id":  hive_id,
                "split":    split,
                "n_segments": len(hive_dataset),
                "accuracy": round(metrics["accuracy"] * 100, 2),
                "f1":       round(metrics["f1"] * 100, 2),
            })
        except Exception as e:
            print(f"  Could not evaluate hive {hive_id}: {e}")
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    if not results:
        return None

    df = pd.DataFrame(results).sort_values("split")

    print("\n" + "=" * 60)
    print("CROSS-HIVE GENERALIZATION REPORT")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Summary: average within-hive (train hives) vs unseen (test hives)
    within_acc = df[df["split"] == "train"]["accuracy"].mean()
    unseen_acc = df[df["split"] == "test"]["accuracy"].mean()
    delta      = within_acc - unseen_acc

    print(f"\nWithin-hive accuracy  (train hives) : {within_acc:.2f}%")
    print(f"Cross-hive accuracy   (test hives)  : {unseen_acc:.2f}%")
    print(f"Generalization gap                  : {delta:.2f}% drop")
    print("=" * 60)
    print("-> This gap is your research finding. Present it honestly.")

    # Save report
    report_path = os.path.join(config.DATA_PROCESSED, "cross_hive_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\nReport saved -> {report_path}")

    return df


# ==============================================================
# SECTION 5 — MAIN (CLI ENTRY POINT)
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Train beehive audio classifier")
    parser.add_argument("--model",      type=str, default="baseline_cnn",
                        choices=["baseline_cnn", "crnn"],
                        help="Model architecture to train")
    parser.add_argument("--epochs",     type=int, default=config.EPOCHS)
    parser.add_argument("--lr",         type=float, default=config.LEARNING_RATE)
    parser.add_argument("--experiment", type=str, default="exp_01",
                        help="Experiment name (used for checkpoint filename and W&B run name)")
    parser.add_argument("--cross-hive", action="store_true",
                        help="Run cross-hive report after training")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print(f"Beehive Training — {args.model.upper()} — {args.experiment}")
    print("=" * 55)

    # Setup
    device     = get_device()
    model      = get_model(args.model).to(device)
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    criterion  = get_loss_fn(train_loader).to(device)
    optimizer  = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler  = ReduceLROnPlateau(optimizer, mode="max", patience=config.LR_PATIENCE,
                                   factor=0.5, verbose=True)

    # Resume from checkpoint if given
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # W&B or CSV logging
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(project="beehive-health-monitoring", name=args.experiment,
                       config=vars(args))
        except Exception:
            use_wandb = False
            print("W&B init failed — using CSV logging instead.")

    log_path  = os.path.join(config.DATA_PROCESSED, f"{args.experiment}_log.csv")
    log_rows  = []

    # Training loop
    best_val_f1   = 0.0
    best_ckpt     = os.path.join(config.CHECKPOINT_DIR, f"{args.experiment}_best.pt")
    patience_count = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0        = time.time()
        train_m   = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m     = evaluate(model, val_loader, device, split_name="val")
        epoch_sec = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_m['loss']:.4f} | Train Acc: {train_m['accuracy']:.4f} | "
              f"Val Acc: {val_m['accuracy']:.4f} | Val F1: {val_m['f1']:.4f} | "
              f"{epoch_sec:.1f}s")

        scheduler.step(val_m["f1"])

        # Log metrics
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()
                                   if k != "loss" or True},
               **{f"val_{k}": v for k, v in val_m.items()
                  if k not in ("confusion_matrix", "all_preds", "all_labels")}}
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(log_path, index=False)

        if use_wandb:
            wandb.log(row)

        # Save best checkpoint
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_count = 0
            torch.save({
                "epoch":          epoch,
                "model_name":     args.model,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1":         best_val_f1,
                "val_accuracy":   val_m["accuracy"],
            }, best_ckpt)
            print(f"  * Saved best checkpoint -> {best_ckpt}")
        else:
            patience_count += 1
            if patience_count >= config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.PATIENCE} epochs)")
                break

    # Final evaluation on test set
    print("\n" + "=" * 55)
    print("FINAL TEST SET EVALUATION")
    print("=" * 55)
    best_model = get_model(args.model).to(device)
    ckpt = torch.load(best_ckpt, map_location=device)
    best_model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(best_model, test_loader, device, split_name="test")

    if args.cross_hive:
        cross_hive_report(best_model, device)

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete.")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Training log    : {log_path}")
    print(f"\nNext: python predict.py --audio <clip.wav> --checkpoint {best_ckpt}")


if __name__ == "__main__":
    main()
