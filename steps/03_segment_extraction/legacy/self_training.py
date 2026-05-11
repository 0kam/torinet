"""
Self-training refinement pipeline for bird vocalization segmentation.

Implements iterative self-training to refine TweetyNet pseudo-labels using
signal-processing teacher evaluation and confidence-based label filtering.

Subcommands:
  evaluate-teachers       - Evaluate SP methods against BirdNET-accepted bouts
  generate-refined-labels - Re-generate labels using selected teacher methods
  self-train              - Run multi-round self-training loop

Usage:
  python self_training.py evaluate-teachers [--f1-threshold 0.4]
  python self_training.py generate-refined-labels [--min-votes 2]
  python self_training.py self-train [--rounds 3] [--device cuda]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_opening
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
BOUTS_DIR = NAS_BASE / "segments" / "test_samples_results_bouts"
MODEL_DIR = STEP_DIR / "models"

# Frame parameters (must match prototype_tweetynet.py)
SR = 32000
HOP_LENGTH = 320
FRAME_DUR = HOP_LENGTH / SR  # ~0.01s per frame

# Training parameters
BATCH_SIZE = 32


# ===========================================================================
# Subcommand: evaluate-teachers
# ===========================================================================


def evaluate_teachers(args: argparse.Namespace) -> None:
    """Evaluate signal processing methods against BirdNET-accepted bout regions."""
    from prototype_segmentation import METHOD_FUNCS, METHOD_NAMES, postprocess_segments

    samples_df = pd.read_csv(SAMPLES_CSV)

    # Collect accepted bout regions per recording
    accepted_regions = {}  # {(species, safe_id): {...}}

    for _, row in samples_df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")
        bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
        if not bout_path.exists():
            continue
        with open(bout_path) as f:
            bout_data = json.load(f)
        accepted_notes = []
        for b in bout_data["bouts"]:
            if b.get("birdnet_verdict") == "accept":
                accepted_notes.extend(b.get("notes", [(b["bout_onset"], b["bout_offset"])]))
        accepted = accepted_notes
        if accepted:
            accepted_regions[(species, safe_id)] = {
                "regions": accepted,
                "file_path": row["file_path"],
                "n_frames": bout_data["n_frames"],
            }

    print(f"Found {len(accepted_regions)} recordings with accepted bouts")

    # Evaluate each method
    method_metrics = []
    for method_id, method_func in METHOD_FUNCS.items():
        tp_total, fp_total, fn_total = 0, 0, 0
        n_evaluated = 0

        for (species, safe_id), info in accepted_regions.items():
            audio_path = TEST_SAMPLES_DIR / species / Path(info["file_path"]).name
            if not audio_path.exists():
                continue

            try:
                y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            except Exception:
                continue

            n_frames = info["n_frames"]

            # Method prediction -> frame mask
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_segs = method_func(y, sr)
                segs = postprocess_segments(raw_segs)
            except Exception:
                continue

            method_mask = np.zeros(n_frames, dtype=bool)
            for onset, offset in segs:
                f_start = int(onset * SR / HOP_LENGTH)
                f_end = int(offset * SR / HOP_LENGTH)
                f_start = max(0, min(f_start, n_frames - 1))
                f_end = max(0, min(f_end, n_frames - 1))
                method_mask[f_start:f_end + 1] = True

            # Accepted bout regions -> frame mask
            accept_mask = np.zeros(n_frames, dtype=bool)
            for onset, offset in info["regions"]:
                f_start = int(onset / FRAME_DUR)
                f_end = int(offset / FRAME_DUR)
                f_start = max(0, min(f_start, n_frames - 1))
                f_end = max(0, min(f_end, n_frames - 1))
                accept_mask[f_start:f_end + 1] = True

            # Frame-level TP/FP/FN
            tp_total += (method_mask & accept_mask).sum()
            fp_total += (method_mask & ~accept_mask).sum()
            fn_total += (~method_mask & accept_mask).sum()
            n_evaluated += 1

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        method_metrics.append({
            "method_id": method_id,
            "method_name": METHOD_NAMES[method_id],
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "n_evaluated": n_evaluated,
        })
        print(
            f"  M{method_id:2d} ({METHOD_NAMES[method_id]:30s}): "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} (n={n_evaluated})"
        )

    # Save results
    metrics_df = pd.DataFrame(method_metrics)
    metrics_df.to_csv(STEP_DIR / "teacher_evaluation.csv", index=False)

    # Select teachers (F1 >= threshold)
    threshold = args.f1_threshold
    selected = [m["method_id"] for m in method_metrics if m["f1"] >= threshold]

    with open(STEP_DIR / "selected_teachers.json", "w") as f:
        json.dump(
            {
                "selected_method_ids": selected,
                "f1_threshold": threshold,
                "metrics": method_metrics,
            },
            f,
            indent=2,
        )

    print(f"\nSelected {len(selected)} teachers (F1 >= {threshold}): {selected}")


# ===========================================================================
# Subcommand: generate-refined-labels
# ===========================================================================


def generate_refined_labels(args: argparse.Namespace) -> None:
    """Generate refined pseudo-labels using selected teacher methods."""
    from prototype_segmentation import METHOD_FUNCS, postprocess_segments
    from prototype_tweetynet import compute_mel_spectrogram

    # Load selected teachers
    teachers_path = STEP_DIR / "selected_teachers.json"
    if not teachers_path.exists():
        print("ERROR: Run evaluate-teachers first")
        sys.exit(1)

    with open(teachers_path) as f:
        teachers_data = json.load(f)
    selected_ids = teachers_data["selected_method_ids"]
    min_votes = args.min_votes

    print(f"Using teacher methods: {selected_ids}, min_votes={min_votes}")

    output_dir = STEP_DIR / "refined_pseudo_labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SAMPLES_CSV)

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        original_path = Path(row["file_path"])
        file_path = TEST_SAMPLES_DIR / species / original_path.name

        if not file_path.exists():
            continue

        safe_id = rec_id.replace(":", "_").replace("/", "_")
        label_path = output_dir / species / f"{safe_id}.npz"

        if label_path.exists():
            print(f"  [{i+1}/{len(df)}] EXISTS: {species}/{safe_id}")
            continue

        try:
            y_orig, sr_orig = librosa.load(str(file_path), sr=None, mono=True)
            y, _ = librosa.load(str(file_path), sr=SR, mono=True)
        except Exception as exc:
            print(f"  [{i+1}/{len(df)}] ERROR: {exc}")
            continue

        if len(y) < SR * 0.5:
            continue

        # Compute spectrogram
        S_db = compute_mel_spectrogram(y, SR)
        n_frames = S_db.shape[1]

        # Generate labels with selected methods
        votes = np.zeros(n_frames, dtype=int)
        for method_id in selected_ids:
            method_func = METHOD_FUNCS[method_id]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_segs = method_func(y_orig, sr_orig)
                segs = postprocess_segments(raw_segs)
            except Exception:
                continue

            for onset, offset in segs:
                f_start = int(onset * SR / HOP_LENGTH)
                f_end = int(offset * SR / HOP_LENGTH)
                f_start = max(0, min(f_start, n_frames - 1))
                f_end = max(0, min(f_end, n_frames - 1))
                votes[f_start:f_end + 1] += 1

        active = votes >= min_votes

        # Morphological cleanup
        gap_frames = int(0.05 / FRAME_DUR)
        if gap_frames > 1:
            active = binary_closing(active, structure=np.ones(gap_frames))
        min_dur_frames = int(0.03 / FRAME_DUR)
        if min_dur_frames > 1:
            active = binary_opening(active, structure=np.ones(min_dur_frames))

        # Save
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(label_path),
            labels=active.astype(np.uint8),
            spectrogram=S_db.astype(np.float32),
        )

        n_bird = active.sum()
        ratio = n_bird / len(active) * 100 if len(active) > 0 else 0
        print(
            f"  [{i+1}/{len(df)}] {species}/{safe_id}: "
            f"{n_frames} frames, {n_bird} bird ({ratio:.1f}%)"
        )

    print(f"\nRefined labels saved to {output_dir}")


# ===========================================================================
# Confidence-filtered label generation (helper for self-train)
# ===========================================================================


def _generate_confidence_labels(
    model_path: Path,
    round_idx: int,
    conf_high: float,
    conf_low: float,
    device: torch.device,
) -> Path:
    """Generate confidence-filtered labels from model predictions.

    Frames with P(bird) > conf_high are labeled bird (mask=1).
    Frames with P(bird) < conf_low are labeled background (mask=1).
    Frames in between are masked out (mask=0) and excluded from loss.

    Returns:
        Path to the output label directory.
    """
    from prototype_tweetynet import (
        N_MELS,
        TweetyNet,
        _predict_full_file,
        normalize_spectrogram,
    )

    output_dir = STEP_DIR / f"self_train_labels_r{round_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TweetyNet(num_classes=2, num_freqbins=N_MELS).to(device)
    model.load_state_dict(
        torch.load(str(model_path), map_location=device, weights_only=True)
    )
    model.eval()

    # Find source labels (from previous round or refined)
    if round_idx == 1:
        src_dir = STEP_DIR / "refined_pseudo_labels"
    else:
        src_dir = STEP_DIR / f"self_train_labels_r{round_idx - 1}"

    for npz_path in sorted(src_dir.rglob("*.npz")):
        rel = npz_path.relative_to(src_dir)
        out_path = output_dir / rel

        if out_path.exists():
            continue

        data = np.load(str(npz_path))
        S_db = data["spectrogram"]
        data.close()

        S_norm = normalize_spectrogram(S_db)
        probs = _predict_full_file(model, S_norm, device)

        # Confidence filtering
        labels = np.zeros(len(probs), dtype=np.uint8)
        mask = np.zeros(len(probs), dtype=np.float32)

        labels[probs > conf_high] = 1  # confident bird
        mask[probs > conf_high] = 1.0
        labels[probs < conf_low] = 0  # confident background
        mask[probs < conf_low] = 1.0
        # frames between conf_low and conf_high are masked out (mask=0)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path), labels=labels, spectrogram=S_db, mask=mask
        )

    return output_dir


# ===========================================================================
# Subcommand: self-train
# ===========================================================================


def self_train(args: argparse.Namespace) -> None:
    """Run multi-round self-training loop."""
    from prototype_tweetynet import (
        CONTEXT_FRAMES,
        N_MELS,
        BirdFrameDataset,
        TweetyNet,
        _compute_class_weights_from_files,
        _plot_training_curves,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rounds_config = [
        {"epochs": 30, "lr": 1e-3, "conf_high": None, "conf_low": None},
        {"epochs": 20, "lr": 5e-4, "conf_high": 0.9, "conf_low": 0.1},
        {"epochs": 15, "lr": 2e-4, "conf_high": 0.85, "conf_low": 0.15},
    ]

    n_rounds = min(args.rounds, len(rounds_config))

    # Round 0 uses refined labels
    labels_dir = STEP_DIR / "refined_pseudo_labels"
    if not labels_dir.exists():
        print("ERROR: Run generate-refined-labels first")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prev_model_path = None
    prev_bird_ratio = None

    checkpoint_path = MODEL_DIR / "self_train_checkpoint.json"
    start_round = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_round = ckpt.get("completed_rounds", 0)
        if start_round > 0:
            prev_model_path = Path(ckpt["last_model_path"])
            prev_bird_ratio = ckpt.get("last_bird_ratio")
            print(f"  Resuming from round {start_round} (model: {prev_model_path})")

    for round_idx in range(start_round, n_rounds):
        cfg = rounds_config[round_idx]
        print(f"\n{'='*60}")
        print(f"SELF-TRAINING ROUND {round_idx}")
        print(f"{'='*60}")
        print(f"  Labels: {labels_dir}")
        print(f"  Epochs: {cfg['epochs']}, LR: {cfg['lr']}")

        if round_idx > 0 and cfg["conf_high"] is not None:
            # Generate masked labels from previous round's predictions
            print(
                f"  Generating confidence-filtered labels "
                f"(high>{cfg['conf_high']}, low<{cfg['conf_low']})..."
            )
            labels_dir = _generate_confidence_labels(
                prev_model_path, round_idx, cfg["conf_high"], cfg["conf_low"],
                device,
            )

        # Check bird ratio for collapse prevention
        npz_files = sorted(labels_dir.rglob("*.npz"))
        total_bird = 0
        total_frames = 0
        total_all_frames = 0
        for f in npz_files:
            data = np.load(str(f))
            file_labels = data["labels"]
            total_all_frames += len(file_labels)
            if "mask" in data:
                file_mask = data["mask"]
                total_bird += (file_labels[file_mask > 0] == 1).sum()
                total_frames += (file_mask > 0).sum()
            else:
                total_bird += file_labels.sum()
                total_frames += len(file_labels)
            data.close()

        bird_ratio = total_bird / (total_frames + 1e-8)
        print(f"  Bird frame ratio: {bird_ratio:.3f}")

        if total_all_frames > total_frames:
            mask_ratio = 1.0 - total_frames / total_all_frames
            print(f"  Mask ratio: {mask_ratio:.1%} of frames masked out")
            if mask_ratio > 0.8:
                print("  WARNING: >80% of frames masked out. Consider relaxing confidence thresholds.")

        if prev_bird_ratio is not None:
            change = abs(bird_ratio - prev_bird_ratio) / (prev_bird_ratio + 1e-8)
            if change > 0.5:
                print(
                    f"  WARNING: Bird ratio changed by {change:.1%} (>50%). "
                    f"Aborting self-training."
                )
                break
        prev_bird_ratio = bird_ratio

        # Split into train/val
        species_dirs = sorted([d for d in labels_dir.iterdir() if d.is_dir()])
        np.random.seed(42)
        indices = np.random.permutation(len(species_dirs))
        split_idx = max(1, int(len(species_dirs) * 0.8))
        train_dirs = [species_dirs[i] for i in indices[:split_idx]]
        val_dirs = [species_dirs[i] for i in indices[split_idx:]]

        train_files = []
        for d in train_dirs:
            train_files.extend(sorted(d.glob("*.npz")))
        val_files = []
        for d in val_dirs:
            val_files.extend(sorted(d.glob("*.npz")))

        print(f"  Train: {len(train_files)} files, Val: {len(val_files)} files")

        # Create datasets
        use_mask = round_idx > 0 and cfg["conf_high"] is not None
        train_dataset = BirdFrameDataset(
            train_files, context_frames=CONTEXT_FRAMES, augment=True,
            mask_key="mask" if use_mask else None,
        )
        val_dataset = BirdFrameDataset(
            val_files, context_frames=CONTEXT_FRAMES, augment=False,
            mask_key="mask" if use_mask else None,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        ) if len(val_dataset) > 0 else None

        # Model
        model = TweetyNet(num_classes=2, num_freqbins=N_MELS).to(device)
        if prev_model_path:
            print(f"  Loading weights from {prev_model_path}")
            model.load_state_dict(
                torch.load(str(prev_model_path), map_location=device, weights_only=True)
            )

        # Training setup
        class_weights = _compute_class_weights_from_files(
            train_files, mask_key="mask" if use_mask else None
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        model_name = f"tweetynet_r{round_idx}"
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "val_acc": [],
        }

        for epoch in range(cfg["epochs"]):
            model.train()
            train_losses = []
            for batch in train_loader:
                if use_mask:
                    batch_spec, batch_labels, batch_mask = batch
                    batch_mask = batch_mask.to(device)
                else:
                    batch_spec, batch_labels = batch
                    batch_mask = None

                batch_spec = batch_spec.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_spec)
                loss_per_frame = criterion(logits, batch_labels)  # (B, T)

                if batch_mask is not None:
                    loss = (loss_per_frame * batch_mask).sum() / (
                        batch_mask.sum() + 1e-8
                    )
                else:
                    loss = loss_per_frame.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)

            # Validation
            if val_loader:
                model.eval()
                val_losses = []
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        if use_mask:
                            bs, bl, bm = batch
                            bm = bm.to(device)
                        else:
                            bs, bl = batch
                            bm = None
                        bs, bl = bs.to(device), bl.to(device)
                        logits = model(bs)
                        loss_pf = criterion(logits, bl)
                        if bm is not None:
                            loss = (loss_pf * bm).sum() / (bm.sum() + 1e-8)
                        else:
                            loss = loss_pf.mean()
                        val_losses.append(loss.item())
                        preds = logits.argmax(dim=1)
                        if bm is not None:
                            correct += ((preds == bl) * bm).sum().item()
                            total += bm.sum().item()
                        else:
                            correct += (preds == bl).sum().item()
                            total += bl.numel()

                avg_val = np.mean(val_losses)
                val_acc = correct / (total + 1e-8)
                history["val_loss"].append(avg_val)
                history["val_acc"].append(val_acc)
                scheduler.step(avg_val)

                print(
                    f"  R{round_idx} Epoch {epoch+1}/{cfg['epochs']}: "
                    f"train={avg_train:.4f} val={avg_val:.4f} acc={val_acc:.4f}"
                )

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    patience_counter = 0
                    torch.save(
                        model.state_dict(),
                        MODEL_DIR / f"{model_name}_best.pt",
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            else:
                print(
                    f"  R{round_idx} Epoch {epoch+1}/{cfg['epochs']}: "
                    f"train={avg_train:.4f}"
                )
                torch.save(
                    model.state_dict(), MODEL_DIR / f"{model_name}_best.pt"
                )

        torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_final.pt")
        with open(MODEL_DIR / f"{model_name}_history.json", "w") as f:
            json.dump(history, f, indent=2)
        _plot_training_curves(history, MODEL_DIR / f"{model_name}_curves.png")

        prev_model_path = MODEL_DIR / f"{model_name}_best.pt"
        print(f"  Round {round_idx} complete. Best val_loss: {best_val_loss:.4f}")

        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({
                "completed_rounds": round_idx + 1,
                "last_model_path": str(prev_model_path),
                "last_bird_ratio": float(bird_ratio),
            }, f, indent=2)

    print(f"\nSelf-training complete. Final model: {prev_model_path}")


# ===========================================================================
# Main CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-training refinement pipeline for bird segmentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate-teachers
    sub = subparsers.add_parser(
        "evaluate-teachers",
        help="Evaluate SP methods against BirdNET-accepted bouts",
    )
    sub.add_argument(
        "--f1-threshold", type=float, default=0.4,
        help="F1 threshold for selecting teacher methods",
    )

    # generate-refined-labels
    sub = subparsers.add_parser(
        "generate-refined-labels",
        help="Generate refined pseudo-labels using selected teachers",
    )
    sub.add_argument(
        "--min-votes", type=int, default=2,
        help="Minimum teacher votes to label a frame as bird",
    )

    # self-train
    sub = subparsers.add_parser(
        "self-train",
        help="Run multi-round self-training loop",
    )
    sub.add_argument(
        "--rounds", type=int, default=3,
        help="Number of self-training rounds",
    )
    sub.add_argument(
        "--device", type=str, default="cuda",
        help="Device for training (cuda or cpu)",
    )

    args = parser.parse_args()

    if args.command == "evaluate-teachers":
        evaluate_teachers(args)
    elif args.command == "generate-refined-labels":
        generate_refined_labels(args)
    elif args.command == "self-train":
        self_train(args)


if __name__ == "__main__":
    main()
