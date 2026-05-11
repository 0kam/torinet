"""
TweetyNet-based bird vocalization segmentation prototype.

Implements a lightweight TweetyNet (CNN + BiLSTM) for binary frame-level
classification (bird vs background) using PyTorch directly, without the vak
framework. This avoids the complexity overhead of vak while retaining the
proven TweetyNet architecture.

Workflow:
  1. Generate pseudo-labels from existing signal-processing methods (OR ensemble)
  2. Train TweetyNet on pseudo-labels (self-training / bootstrapping)
  3. Predict frame labels on test samples
  4. Visualize and compare against signal-processing baselines

Usage:
  # Phase 1: Generate pseudo-labels from signal-processing ensemble
  python prototype_tweetynet.py generate-labels

  # Phase 2: Train TweetyNet on pseudo-labels
  python prototype_tweetynet.py train [--epochs 30] [--device cuda]

  # Phase 3: Predict on test samples
  python prototype_tweetynet.py predict [--device cuda]

  # Phase 4: Visualize predictions
  python prototype_tweetynet.py visualize [--species jabwar] [--limit 5]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_opening, label
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
RESULTS_DIR = NAS_BASE / "segments" / "test_samples_results_tweetynet"
LABELS_DIR = STEP_DIR / "pseudo_labels"
MODEL_DIR = STEP_DIR / "models"

# Spectrogram parameters (shared across all stages)
SR = 32000  # Resample all audio to 32kHz
N_FFT = 1024
HOP_LENGTH = 320  # 10ms hop at 32kHz
N_MELS = 128
FMIN = 150
FMAX = 12000
FRAME_DUR = HOP_LENGTH / SR  # ~0.01s per frame

# Training parameters
CONTEXT_FRAMES = 200  # ~2s context window for training patches
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
PATIENCE = 5  # Early stopping patience


# ===========================================================================
# TweetyNet Architecture (lightweight reimplementation)
# ===========================================================================


class TweetyNet(nn.Module):
    """Lightweight TweetyNet for binary frame-level segmentation.

    Architecture: 2x (Conv2d -> ReLU -> MaxPool) -> BiLSTM -> Linear(2)

    Input:  (batch, 1, n_mels, time_steps) mel spectrogram
    Output: (batch, 2, time_steps)  logits for [background, bird]
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_freqbins: int = N_MELS,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: Tuple[int, int] = (5, 5),
        pool_size: Tuple[int, int] = (4, 1),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN block 1
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.pool1 = nn.MaxPool2d(pool_size)

        # CNN block 2
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.pool2 = nn.MaxPool2d(pool_size)

        # Calculate RNN input size after pooling
        freq_after_pool = num_freqbins // (pool_size[0] ** 2)
        rnn_input_size = conv2_filters * max(freq_after_pool, 1)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 1, n_mels, time_steps) mel spectrogram

        Returns:
            (batch, num_classes, time_steps) class logits per frame
        """
        # CNN blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 32, F/8, T)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 64, F/64, T)

        # Reshape for RNN: merge freq and channel dims
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)  # (B, T, C*F)

        # BiLSTM
        x, _ = self.lstm(x)  # (B, T, hidden*2)

        # Output
        x = self.dropout(x)
        x = self.fc(x)  # (B, T, num_classes)
        x = x.permute(0, 2, 1)  # (B, num_classes, T)

        return x


# ===========================================================================
# Spectrogram Computation
# ===========================================================================


def compute_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns:
        (n_mels, time_frames) log-power mel spectrogram
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def normalize_spectrogram(S_db: np.ndarray) -> np.ndarray:
    """Per-file z-score normalization."""
    mean = S_db.mean()
    std = S_db.std() + 1e-8
    return (S_db - mean) / std


# ===========================================================================
# Pseudo-label Generation from Signal-Processing Ensemble
# ===========================================================================


def _generate_ensemble_labels(
    y: np.ndarray, sr: int, method_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Generate frame-level pseudo-labels by OR-ensembling signal-processing methods.

    Uses the methods from prototype_segmentation.py to create initial labels.
    A frame is labeled as 'bird' if >= 2 out of 7 methods detect activity there.

    Args:
        y: Audio waveform
        sr: Sample rate
        method_ids: If provided, only use these method IDs (subset of METHOD_FUNCS)

    Returns:
        Boolean array of shape (n_frames,) where True = bird vocalization
    """
    # Import methods from existing prototype
    from prototype_segmentation import (
        METHOD_FUNCS,
        postprocess_segments,
    )

    n_frames = 1 + int(len(y) / sr * SR / HOP_LENGTH)
    votes = np.zeros(n_frames, dtype=int)

    methods = {mid: METHOD_FUNCS[mid] for mid in method_ids} if method_ids else METHOD_FUNCS
    for method_id, method_func in methods.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_segments = method_func(y, sr)
            segments = postprocess_segments(raw_segments)
        except Exception:
            continue

        # Convert segments to frame-level mask
        for onset, offset in segments:
            frame_start = int(onset * SR / HOP_LENGTH)
            frame_end = int(offset * SR / HOP_LENGTH)
            frame_start = max(0, min(frame_start, n_frames - 1))
            frame_end = max(0, min(frame_end, n_frames - 1))
            votes[frame_start:frame_end + 1] += 1

    # Majority vote: >= 2 methods agree
    active = votes >= 2

    # Morphological cleanup
    gap_frames = int(0.05 / FRAME_DUR)  # 50ms gap fill
    if gap_frames > 1:
        active = binary_closing(active, structure=np.ones(gap_frames))
    min_frames = int(0.03 / FRAME_DUR)  # 30ms minimum duration
    if min_frames > 1:
        active = binary_opening(active, structure=np.ones(min_frames))

    return active


def generate_labels(args: argparse.Namespace) -> None:
    """Generate pseudo-labels for all test samples."""
    df = pd.read_csv(SAMPLES_CSV)
    print(f"Loaded {len(df)} test samples")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        original_path = Path(row["file_path"])
        file_path = TEST_SAMPLES_DIR / species / original_path.name

        if not file_path.exists():
            print(f"  [{i+1}/{len(df)}] SKIP (not found): {file_path}")
            continue

        safe_id = rec_id.replace(":", "_").replace("/", "_")
        label_path = LABELS_DIR / species / f"{safe_id}.npz"

        if label_path.exists():
            print(f"  [{i+1}/{len(df)}] EXISTS: {species}/{safe_id}")
            continue

        try:
            y, file_sr = librosa.load(str(file_path), sr=SR, mono=True)
        except Exception as exc:
            print(f"  [{i+1}/{len(df)}] ERROR loading: {exc}")
            continue

        if len(y) < SR * 0.5:
            print(f"  [{i+1}/{len(df)}] SKIP (too short): {species}/{safe_id}")
            continue

        # Generate pseudo-labels using original sample rate for methods
        y_orig, sr_orig = librosa.load(str(file_path), sr=None, mono=True)
        active = _generate_ensemble_labels(y_orig, sr_orig)

        # Compute mel spectrogram at target SR
        S_db = compute_mel_spectrogram(y, SR)
        n_frames = S_db.shape[1]

        # Align label length to spectrogram frames
        if len(active) > n_frames:
            active = active[:n_frames]
        elif len(active) < n_frames:
            active = np.pad(active, (0, n_frames - len(active)), mode="constant")

        # Save
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(label_path),
            labels=active.astype(np.uint8),
            spectrogram=S_db.astype(np.float32),
        )

        n_bird = active.sum()
        ratio = n_bird / len(active) * 100 if len(active) > 0 else 0
        print(f"  [{i+1}/{len(df)}] {species}/{safe_id}: {n_frames} frames, "
              f"{n_bird} bird ({ratio:.1f}%)")

    print(f"\nPseudo-labels saved to {LABELS_DIR}")


# ===========================================================================
# Dataset
# ===========================================================================


class BirdFrameDataset(Dataset):
    """Dataset for frame-level bird/background classification.

    Extracts random patches of CONTEXT_FRAMES from spectrograms.
    """

    def __init__(
        self,
        npz_files: List[Path],
        context_frames: int = CONTEXT_FRAMES,
        augment: bool = False,
        mask_key: Optional[str] = None,
    ):
        self.context_frames = context_frames
        self.augment = augment
        self.mask_key = mask_key
        self.items: List[Tuple[Path, int]] = []

        # Collect valid start positions
        for npz_path in npz_files:
            data = np.load(str(npz_path))
            n_frames = data["spectrogram"].shape[1]
            data.close()
            if n_frames >= context_frames:
                n_patches = max(1, n_frames // context_frames)
                for _ in range(n_patches):
                    self.items.append((npz_path, n_frames))

        print(f"Dataset: {len(npz_files)} files, {len(self.items)} patches")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        npz_path, n_frames = self.items[idx]
        data = np.load(str(npz_path))
        S_db = data["spectrogram"]
        labels = data["labels"]
        if self.mask_key is not None:
            if self.mask_key in data:
                mask = data[self.mask_key]
            else:
                mask = np.ones(len(labels), dtype=np.float32)
        data.close()

        # Random crop
        max_start = n_frames - self.context_frames
        start = np.random.randint(0, max_start + 1)
        end = start + self.context_frames

        spec_patch = S_db[:, start:end]  # (n_mels, context_frames)
        label_patch = labels[start:end]  # (context_frames,)

        # Normalize patch
        spec_patch = normalize_spectrogram(spec_patch)

        # Augmentation
        if self.augment:
            # Time masking
            if np.random.random() < 0.3:
                t_start = np.random.randint(0, self.context_frames - 10)
                t_len = np.random.randint(5, 20)
                spec_patch[:, t_start:t_start + t_len] = 0.0

            # Frequency masking
            if np.random.random() < 0.3:
                f_start = np.random.randint(0, N_MELS - 10)
                f_len = np.random.randint(5, 15)
                spec_patch[f_start:f_start + f_len, :] = 0.0

            # Additive Gaussian noise
            if np.random.random() < 0.2:
                noise = np.random.randn(*spec_patch.shape) * 0.1
                spec_patch = spec_patch + noise

            # Background noise mix: overlay background region from same file
            # onto bird regions to make model robust to noise
            if np.random.random() < 0.3:
                bg_mask = label_patch == 0
                if bg_mask.sum() > 10:
                    bg_spec = spec_patch[:, bg_mask]
                    # Tile background to match full length
                    n_bg = bg_spec.shape[1]
                    if n_bg > 0:
                        reps = self.context_frames // n_bg + 1
                        bg_tiled = np.tile(bg_spec, (1, reps))[:, :self.context_frames]
                        mix_weight = np.random.uniform(0.1, 0.4)
                        spec_patch = spec_patch * (1 - mix_weight) + bg_tiled * mix_weight

        # Add channel dim: (1, n_mels, context_frames)
        spec_tensor = torch.from_numpy(spec_patch[np.newaxis].astype(np.float32))
        label_tensor = torch.from_numpy(label_patch.astype(np.int64))

        if self.mask_key is not None:
            mask_patch = mask[start:end]
            mask_tensor = torch.from_numpy(mask_patch.astype(np.float32))
            return spec_tensor, label_tensor, mask_tensor

        return spec_tensor, label_tensor


class BirdFullFileDataset(Dataset):
    """Dataset that returns full files (for inference)."""

    def __init__(self, label_dir: Path):
        self.npz_files = sorted(label_dir.rglob("*.npz"))
        print(f"Inference dataset: {len(self.npz_files)} files")

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        npz_path = self.npz_files[idx]
        data = np.load(str(npz_path))
        S_db = normalize_spectrogram(data["spectrogram"])
        labels = data["labels"]
        data.close()

        spec_tensor = torch.from_numpy(S_db[np.newaxis].astype(np.float32))
        label_tensor = torch.from_numpy(labels.astype(np.int64))

        # Extract species/recording_id from path
        file_id = f"{npz_path.parent.name}/{npz_path.stem}"
        return spec_tensor, label_tensor, file_id


# ===========================================================================
# Training
# ===========================================================================


def _compute_class_weights_from_files(
    npz_files: List[Path], mask_key: Optional[str] = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced labels."""
    total = np.array([0, 0], dtype=np.float64)
    for npz_path in npz_files:
        data = np.load(str(npz_path))
        labels = data["labels"]
        if mask_key and mask_key in data:
            mask = data[mask_key] > 0
            labels = labels[mask]
        data.close()
        total[0] += (labels == 0).sum()
        total[1] += (labels == 1).sum()

    if total.sum() == 0:
        return torch.ones(2)

    # Inverse frequency, normalized
    weights = total.sum() / (2.0 * total + 1e-8)
    weights = weights / weights.sum() * 2.0
    print(f"Class distribution: background={total[0]:.0f}, bird={total[1]:.0f}")
    print(f"Class weights: background={weights[0]:.3f}, bird={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32)


def train_model(args: argparse.Namespace) -> None:
    """Train TweetyNet on pseudo-labels."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    labels_dir = Path(args.labels_dir) if args.labels_dir else LABELS_DIR
    model_name = args.model_name

    if not labels_dir.exists() or not any(labels_dir.rglob("*.npz")):
        print(f"ERROR: No pseudo-labels found in {labels_dir}. Run 'generate-labels' first.")
        sys.exit(1)

    # Split into train/val (80/20 by species directory)
    species_dirs = sorted([d for d in labels_dir.iterdir() if d.is_dir()])
    np.random.seed(42)
    np.random.shuffle(species_dirs)
    split_idx = max(1, int(len(species_dirs) * 0.8))
    train_dirs = species_dirs[:split_idx]
    val_dirs = species_dirs[split_idx:]

    # Collect npz files per split
    train_files = []
    for d in train_dirs:
        train_files.extend(sorted(d.glob("*.npz")))
    val_files = []
    for d in val_dirs:
        val_files.extend(sorted(d.glob("*.npz")))

    print(f"Split: {len(train_dirs)} train species ({len(train_files)} files), "
          f"{len(val_dirs)} val species ({len(val_files)} files)")

    # Datasets
    train_dataset = BirdFrameDataset(train_files, augment=True)
    val_dataset = BirdFrameDataset(val_files, augment=False)

    if len(train_dataset) == 0:
        print("ERROR: No training data found.")
        sys.exit(1)

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
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Class-weighted loss
    class_weights = _compute_class_weights_from_files(train_files).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    lr = args.lr if args.lr is not None else LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    # Training loop
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for batch_spec, batch_labels in train_loader:
            batch_spec = batch_spec.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_spec)  # (B, 2, T)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validate
        if val_loader is not None:
            model.eval()
            val_losses = []
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_spec, batch_labels in val_loader:
                    batch_spec = batch_spec.to(device)
                    batch_labels = batch_labels.to(device)

                    logits = model(batch_spec)
                    loss = criterion(logits, batch_labels)
                    val_losses.append(loss.item())

                    preds = logits.argmax(dim=1)
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.numel()

            avg_val_loss = np.mean(val_losses)
            val_acc = correct / total if total > 0 else 0
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_acc={val_acc:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: train_loss={avg_train_loss:.4f}")
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")

    # Save final model and history
    torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_final.pt")
    with open(MODEL_DIR / f"{model_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    _plot_training_curves(history, MODEL_DIR / f"{model_name}_curves.png")

    print(f"\nModels saved to {MODEL_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")


def _plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Plot training loss and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    if history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], "g-")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ===========================================================================
# Prediction
# ===========================================================================


def _predict_full_file(
    model: nn.Module,
    spec: np.ndarray,
    device: torch.device,
    context_frames: int = CONTEXT_FRAMES,
) -> np.ndarray:
    """Predict frame labels for a full spectrogram using sliding window.

    Args:
        model: Trained TweetyNet model
        spec: (n_mels, n_frames) normalized spectrogram
        device: torch device
        context_frames: Window size for chunked prediction

    Returns:
        (n_frames,) predicted labels (0=background, 1=bird)
    """
    model.eval()
    n_frames = spec.shape[1]

    if n_frames <= context_frames:
        # Single chunk
        x = torch.from_numpy(spec[np.newaxis, np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)  # (1, 2, T)
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()  # P(bird)
        return probs

    # Sliding window with overlap
    stride = context_frames // 2
    prob_sum = np.zeros(n_frames, dtype=np.float64)
    count = np.zeros(n_frames, dtype=np.float64)

    for start in range(0, n_frames - context_frames + 1, stride):
        end = start + context_frames
        patch = spec[:, start:end]
        x = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32)).to(device)

        with torch.no_grad():
            logits = model(x)  # (1, 2, T)
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()

        prob_sum[start:end] += probs
        count[start:end] += 1.0

    # Handle tail
    if count[-1] == 0:
        tail_start = n_frames - context_frames
        patch = spec[:, tail_start:n_frames]
        x = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        offset = n_frames - tail_start
        prob_sum[tail_start:n_frames] += probs[:offset]
        count[tail_start:n_frames] += 1.0

    avg_probs = prob_sum / np.maximum(count, 1.0)
    return avg_probs


def _probs_to_segments(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_dur: float = 0.03,
    merge_gap: float = 0.05,
) -> List[Tuple[float, float]]:
    """Convert frame-level probabilities to (onset, offset) segments.

    Args:
        probs: (n_frames,) P(bird) for each frame
        threshold: Classification threshold
        min_dur: Minimum segment duration in seconds
        merge_gap: Maximum gap to merge in seconds

    Returns:
        List of (onset_s, offset_s) tuples
    """
    active = probs >= threshold

    # Morphological cleanup
    gap_frames = int(merge_gap / FRAME_DUR)
    if gap_frames > 1:
        active = binary_closing(active, structure=np.ones(gap_frames))
    min_frames = int(min_dur / FRAME_DUR)
    if min_frames > 1:
        active = binary_opening(active, structure=np.ones(min_frames))

    # Extract segments
    lab_arr, n = label(active)
    segments = []
    for k in range(1, n + 1):
        idx = np.where(lab_arr == k)[0]
        onset = idx[0] * FRAME_DUR
        offset = (idx[-1] + 1) * FRAME_DUR
        if offset - onset >= min_dur:
            segments.append((onset, offset))

    return segments


def predict(args: argparse.Namespace) -> None:
    """Run TweetyNet prediction on all test samples."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Predicting on device: {device}")

    model_path = MODEL_DIR / "tweetynet_best.pt"
    if not model_path.exists():
        model_path = MODEL_DIR / "tweetynet_final.pt"
    if not model_path.exists():
        print("ERROR: No trained model found. Run 'train' first.")
        sys.exit(1)

    # Load model
    model = TweetyNet(num_classes=2, num_freqbins=N_MELS).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Process all test samples
    df = pd.read_csv(SAMPLES_CSV)
    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        original_path = Path(row["file_path"])
        file_path = TEST_SAMPLES_DIR / species / original_path.name

        if not file_path.exists():
            print(f"  [{i+1}/{len(df)}] SKIP: {file_path}")
            continue

        try:
            y, _ = librosa.load(str(file_path), sr=SR, mono=True)
        except Exception as exc:
            print(f"  [{i+1}/{len(df)}] ERROR: {exc}")
            continue

        if len(y) < SR * 0.5:
            continue

        # Compute spectrogram
        S_db = compute_mel_spectrogram(y, SR)
        S_norm = normalize_spectrogram(S_db)

        # Predict
        probs = _predict_full_file(model, S_norm, device)
        segments = _probs_to_segments(probs)

        safe_id = rec_id.replace(":", "_").replace("/", "_")

        # Save predictions
        pred_dir = RESULTS_DIR / species
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(pred_dir / f"{safe_id}_pred.npz"),
            probs=probs.astype(np.float32),
        )

        total_dur = sum(off - on for on, off in segments)
        results.append({
            "recording_id": rec_id,
            "ebird_species_code": species,
            "n_segments": len(segments),
            "total_duration_sec": round(total_dur, 3),
        })

        print(f"  [{i+1}/{len(df)}] {species}/{safe_id}: "
              f"{len(segments)} segments, {total_dur:.1f}s total")

    # Save results CSV
    if results:
        results_df = pd.DataFrame(results)
        results_csv = STEP_DIR / "tweetynet_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults saved to {results_csv}")

        # Summary
        print(f"\n--- Summary ---")
        print(f"Files processed: {len(results)}")
        print(f"Total segments: {results_df['n_segments'].sum()}")
        print(f"Mean segments/file: {results_df['n_segments'].mean():.1f}")


# ===========================================================================
# Visualization
# ===========================================================================


def visualize(args: argparse.Namespace) -> None:
    """Visualize TweetyNet predictions vs pseudo-labels."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_path = MODEL_DIR / "tweetynet_best.pt"
    if not model_path.exists():
        model_path = MODEL_DIR / "tweetynet_final.pt"
    if not model_path.exists():
        print("ERROR: No trained model found. Run 'train' first.")
        sys.exit(1)

    model = TweetyNet(num_classes=2, num_freqbins=N_MELS).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(SAMPLES_CSV)
    if args.species:
        df = df[df["ebird_species_code"] == args.species]
    if args.limit:
        df = df.head(args.limit)

    print(f"Visualizing {len(df)} files...")

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        original_path = Path(row["file_path"])
        file_path = TEST_SAMPLES_DIR / species / original_path.name

        if not file_path.exists():
            continue

        try:
            y, _ = librosa.load(str(file_path), sr=SR, mono=True)
        except Exception:
            continue

        S_db = compute_mel_spectrogram(y, SR)
        S_norm = normalize_spectrogram(S_db)

        # TweetyNet prediction
        probs = _predict_full_file(model, S_norm, device)
        tn_segments = _probs_to_segments(probs)

        # Load pseudo-labels if available
        safe_id = rec_id.replace(":", "_").replace("/", "_")
        label_path = LABELS_DIR / species / f"{safe_id}.npz"
        pseudo_labels = None
        if label_path.exists():
            data = np.load(str(label_path))
            pseudo_labels = data["labels"]
            data.close()

        # Create visualization
        vis_path = RESULTS_DIR / species / f"{safe_id}_tweetynet.png"
        _visualize_comparison(
            S_db, SR, probs, tn_segments, pseudo_labels,
            vis_path, rec_id, len(y) / SR,
        )
        print(f"  [{i+1}/{len(df)}] {species}/{safe_id} -> {vis_path.name}")


def _visualize_comparison(
    S_db: np.ndarray,
    sr: int,
    probs: np.ndarray,
    segments: List[Tuple[float, float]],
    pseudo_labels: Optional[np.ndarray],
    output_path: Path,
    recording_id: str,
    duration: float,
) -> None:
    """Create comparison figure: spectrogram with segment overlays + probability."""
    n_rows = 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 8), sharex=True)

    # Row 1: Mel spectrogram (plain)
    ax = axes[0]
    librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        ax=ax, cmap="gray_r", fmin=FMIN, fmax=FMAX,
    )
    ax.set_title(f"TweetyNet: {recording_id}  ({duration:.1f}s)")
    ax.set_ylabel("Spectrogram")

    # Row 2: TweetyNet probability curve + threshold
    ax = axes[1]
    prob_time = np.linspace(0, duration, len(probs))
    ax.fill_between(prob_time, probs, alpha=0.3, color="steelblue")
    ax.plot(prob_time, probs, color="steelblue", linewidth=0.8, label="P(bird)")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, linewidth=0.7, label="threshold=0.5")
    if pseudo_labels is not None:
        label_time = np.linspace(0, duration, len(pseudo_labels))
        ax.fill_between(label_time, pseudo_labels.astype(float) * 0.95, alpha=0.15,
                        color="orange", label="Pseudo-label")
    ax.set_ylabel("Bird probability")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=7)

    # Row 3: Spectrogram with segment overlays
    ax = axes[2]
    librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        ax=ax, cmap="gray_r", fmin=FMIN, fmax=FMAX,
    )
    for onset, offset in segments:
        ax.axvspan(onset, offset, alpha=0.35, color="limegreen")
    n_segs = len(segments)
    total_dur = sum(off - on for on, off in segments)
    ax.set_ylabel(f"Segments ({n_segs})")
    ax.text(
        0.01, 0.92,
        f"TweetyNet: {n_segs} segments, {total_dur:.1f}s total",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
    )

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ===========================================================================
# Main CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TweetyNet-based bird vocalization segmentation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-labels
    sub = subparsers.add_parser("generate-labels", help="Generate pseudo-labels from SP ensemble")

    # train
    sub = subparsers.add_parser("train", help="Train TweetyNet on pseudo-labels")
    sub.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    sub.add_argument("--device", type=str, default="cuda")
    sub.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    sub.add_argument("--lr", type=float, default=None, help="Override learning rate")
    sub.add_argument("--labels-dir", type=str, default=None, help="Override pseudo-labels directory")
    sub.add_argument("--model-name", type=str, default="tweetynet", help="Base name for saved model files")

    # predict
    sub = subparsers.add_parser("predict", help="Run TweetyNet prediction on test samples")
    sub.add_argument("--device", type=str, default="cuda")

    # visualize
    sub = subparsers.add_parser("visualize", help="Visualize predictions")
    sub.add_argument("--species", type=str, default=None)
    sub.add_argument("--limit", type=int, default=None)
    sub.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "generate-labels":
        generate_labels(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "visualize":
        visualize(args)


if __name__ == "__main__":
    main()
