"""Pre-compute Perch v2 embeddings, SNR, and NDSI for Bird-MixIT sources.

For each separated source WAV, extract non-overlapping 5-second windows and
compute:
  - Perch v2 embedding (1280-d, L2-normalized)
  - SNR (dB) estimated from frame-level RMS
  - NDSI (Normalized Difference Soundscape Index)

Results are saved as .npz files for downstream use (channel selection,
clustering, quality filtering).

Usage:
  python precompute_features.py [--species CODE] [--workers N] [--device cpu/gpu]
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"

SOURCES_DIR = NAS_BASE / "segments" / "birdmixit_sources"
EMBEDDINGS_DIR = NAS_BASE / "segments" / "birdmixit_embeddings"

MIXIT_SR = 22050
PERCH_SR = 32000
PERCH_WINDOW_S = 5.0
PERCH_WINDOW_SAMPLES = int(PERCH_WINDOW_S * PERCH_SR)  # 160000
PERCH_MODEL_URL = "https://tfhub.dev/google/bird-vocalization-classifier/2"

# Minimum window length to process (in seconds)
MIN_WINDOW_S = 1.0
MIN_WINDOW_MIXIT = int(MIN_WINDOW_S * MIXIT_SR)
MIN_WINDOW_PERCH = int(MIN_WINDOW_S * PERCH_SR)

# NDSI STFT parameters (computed at MIXIT_SR)
NDSI_N_FFT = 1024
NDSI_HOP = 512
NDSI_FREQ_RES = MIXIT_SR / NDSI_N_FFT  # ~21.5 Hz/bin

# SNR frame parameters (computed at MIXIT_SR)
SNR_FRAME_LEN = int(0.025 * MIXIT_SR)  # 25 ms
SNR_HOP_LEN = int(0.010 * MIXIT_SR)    # 10 ms


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _flush_print(msg):
    """Print with flush for nohup compatibility."""
    print(msg, flush=True)


def load_metadata(species_filter=None):
    """Load test samples metadata, optionally filtered by species."""
    df = pd.read_csv(SAMPLES_CSV)
    if species_filter:
        df = df[df["ebird_species_code"] == species_filter]
        if df.empty:
            _flush_print(f"ERROR: No recordings found for species '{species_filter}'")
            sys.exit(1)
    return df


# ---------------------------------------------------------------------------
# Perch model
# ---------------------------------------------------------------------------


def _load_perch_model():
    """Load Google Perch v2 model from TensorFlow Hub."""
    import tensorflow_hub as hub

    _flush_print("Loading Perch v2 model from TF Hub...")
    model = hub.load(PERCH_MODEL_URL)
    infer = model.signatures["serving_default"]
    _flush_print("Perch v2 loaded (embedding dim=1280)")
    return infer


def _extract_perch_embedding(infer, audio: np.ndarray) -> np.ndarray:
    """Extract L2-normalized Perch v2 embedding (1280-d) from audio at PERCH_SR.

    Audio is padded/trimmed to 5s (160000 samples).
    """
    import tensorflow as tf

    if len(audio) < PERCH_WINDOW_SAMPLES:
        audio = np.pad(audio, (0, PERCH_WINDOW_SAMPLES - len(audio)))
    audio = audio[:PERCH_WINDOW_SAMPLES]

    inp = tf.constant(audio[np.newaxis].astype(np.float32))
    result = infer(inputs=inp)
    emb = result["output_1"].numpy()[0]  # (1280,)

    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------


def _compute_snr(y: np.ndarray) -> float:
    """Estimate SNR (dB) from audio without a clean reference.

    Uses frame-level RMS with noise floor = median(frame_rms) and
    signal = RMS of frames above median + 2*MAD threshold.
    """
    # Frame-level RMS
    frames = librosa.util.frame(y, frame_length=SNR_FRAME_LEN, hop_length=SNR_HOP_LEN)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))

    if len(frame_rms) == 0:
        return 0.0

    noise_floor = np.median(frame_rms)
    mad = np.median(np.abs(frame_rms - noise_floor))
    threshold = noise_floor + 2.0 * mad

    signal_mask = frame_rms > threshold
    if not np.any(signal_mask):
        return 0.0

    signal_rms = np.sqrt(np.mean(frame_rms[signal_mask] ** 2))
    noise_rms = noise_floor

    snr_db = 20.0 * np.log10(signal_rms / (noise_rms + 1e-10))
    return float(snr_db)


# ---------------------------------------------------------------------------
# NDSI computation
# ---------------------------------------------------------------------------


def _compute_ndsi(y: np.ndarray) -> float:
    """Compute Normalized Difference Soundscape Index.

    biophony:    2-10 kHz
    anthrophony: 1-2 kHz
    NDSI = (bio - anthro) / (bio + anthro + eps)
    """
    S = np.abs(librosa.stft(y, n_fft=NDSI_N_FFT, hop_length=NDSI_HOP)) ** 2

    # Frequency bin boundaries at MIXIT_SR (22050 Hz), n_fft=1024
    # bin_k = k * sr / n_fft
    anthro_lo = int(np.ceil(1000.0 / NDSI_FREQ_RES))   # ~47
    anthro_hi = int(np.floor(2000.0 / NDSI_FREQ_RES))  # ~93
    bio_lo = int(np.ceil(2000.0 / NDSI_FREQ_RES))      # ~93
    bio_hi = int(np.floor(10000.0 / NDSI_FREQ_RES))    # ~465

    # Clamp to valid range
    n_bins = S.shape[0]
    anthro_hi = min(anthro_hi, n_bins - 1)
    bio_hi = min(bio_hi, n_bins - 1)

    sum_anthro = np.sum(S[anthro_lo:anthro_hi + 1, :])
    sum_bio = np.sum(S[bio_lo:bio_hi + 1, :])

    ndsi = (sum_bio - sum_anthro) / (sum_bio + sum_anthro + 1e-10)
    return float(ndsi)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def _process_source_file(
    infer, src_path: Path, out_path: Path, device: str,
    tweetynet=None, tweetynet_device=None,
) -> dict:
    """Process a single Bird-MixIT source file.

    Returns dict with stats or None on error.
    """
    try:
        y_mixit, sr = sf.read(str(src_path), dtype="float32")
    except Exception as e:
        _flush_print(f"  WARN: Cannot read {src_path.name}: {e}")
        return None

    if sr != MIXIT_SR:
        _flush_print(f"  WARN: Unexpected SR {sr} in {src_path.name}, resampling")
        y_mixit = librosa.resample(y_mixit, orig_sr=sr, target_sr=MIXIT_SR)

    # Ensure mono
    if y_mixit.ndim > 1:
        y_mixit = y_mixit.mean(axis=1)

    total_samples_mixit = len(y_mixit)

    # Resample to PERCH_SR for embeddings
    y_perch = librosa.resample(y_mixit, orig_sr=MIXIT_SR, target_sr=PERCH_SR)
    total_samples_perch = len(y_perch)

    # Window over audio in 5-second chunks (at MIXIT_SR for SNR/NDSI, at PERCH_SR for embeddings)
    mixit_window = int(PERCH_WINDOW_S * MIXIT_SR)  # 5s at 22050 = 110250

    embeddings = []
    snr_values = []
    ndsi_values = []
    bird_ratios = []
    window_starts = []  # sample offset at MIXIT_SR

    n_windows = 0
    for start_mixit in range(0, total_samples_mixit, mixit_window):
        end_mixit = start_mixit + mixit_window
        chunk_mixit = y_mixit[start_mixit:end_mixit]

        # Skip windows shorter than 1 second
        if len(chunk_mixit) < MIN_WINDOW_MIXIT:
            break

        # Corresponding Perch chunk
        start_perch = int(start_mixit * PERCH_SR / MIXIT_SR)
        end_perch = start_perch + PERCH_WINDOW_SAMPLES
        chunk_perch = y_perch[start_perch:end_perch]

        if len(chunk_perch) < MIN_WINDOW_PERCH:
            break

        # Perch embedding
        emb = _extract_perch_embedding(infer, chunk_perch)
        embeddings.append(emb)

        # SNR from MIXIT_SR audio
        snr = _compute_snr(chunk_mixit)
        snr_values.append(snr)

        # NDSI from MIXIT_SR audio
        ndsi = _compute_ndsi(chunk_mixit)
        ndsi_values.append(ndsi)

        # TweetyNet bird activity ratio for this window
        if tweetynet is not None and tweetynet_device is not None:
            from birdmixit_pipeline import _tweetynet_predict
            probs = _tweetynet_predict(tweetynet, chunk_mixit, tweetynet_device)
            bird_ratio = float((probs > 0.5).mean())
        else:
            bird_ratio = -1.0  # sentinel: not computed
        bird_ratios.append(bird_ratio)

        window_starts.append(start_mixit)
        n_windows += 1

    if n_windows == 0:
        _flush_print(f"  WARN: No valid windows in {src_path.name}")
        return None

    # Save .npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        embeddings=np.array(embeddings, dtype=np.float32),
        snr=np.array(snr_values, dtype=np.float32),
        ndsi=np.array(ndsi_values, dtype=np.float32),
        bird_ratio=np.array(bird_ratios, dtype=np.float32),
        window_starts=np.array(window_starts, dtype=np.int32),
    )

    return {"n_windows": n_windows, "path": str(out_path)}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_precompute(args):
    """Pre-compute features for all Bird-MixIT sources."""
    t0 = time.time()

    # Configure TF device
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    if args.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            _flush_print(f"Using GPU: {gpus[0].name}")
        else:
            _flush_print("No GPU found, falling back to CPU")
    else:
        _flush_print("Using CPU for TensorFlow")

    # Load metadata
    df = load_metadata(args.species)
    species_list = sorted(df["ebird_species_code"].unique())
    _flush_print(f"Species to process: {len(species_list)}")

    # Load Perch model once
    infer = _load_perch_model()

    # Load TweetyNet for bird activity ratio
    import torch
    from birdmixit_pipeline import _load_tweetynet
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush_print(f"Loading TweetyNet (device={torch_device})...")
    tweetynet = _load_tweetynet(torch_device)

    # Gather all source files
    file_list = []  # (src_path, out_path, species, rec_id, ch)
    for species in species_list:
        sp_df = df[df["ebird_species_code"] == species]
        rec_ids = sp_df["recording_id"].unique()
        for rec_id in rec_ids:
            safe_id = rec_id.replace(":", "_")
            for ch in range(4):
                src_path = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
                out_path = EMBEDDINGS_DIR / species / f"{safe_id}_src{ch}.npz"
                file_list.append((src_path, out_path, species, rec_id, ch))

    _flush_print(f"Total source files to process: {len(file_list)}")

    # Filter: skip already computed (resume support)
    if not args.force:
        remaining = [(s, o, sp, r, c) for s, o, sp, r, c in file_list if not o.exists()]
        skipped_existing = len(file_list) - len(remaining)
        if skipped_existing > 0:
            _flush_print(f"Skipping {skipped_existing} already computed files (use --force to recompute)")
        file_list = remaining

    # Filter: skip missing source files
    missing = [(s, o, sp, r, c) for s, o, sp, r, c in file_list if not s.exists()]
    if missing:
        _flush_print(f"Skipping {len(missing)} missing source files")
        file_list = [(s, o, sp, r, c) for s, o, sp, r, c in file_list if s.exists()]

    _flush_print(f"Files to process: {len(file_list)}")

    if not file_list:
        _flush_print("Nothing to do.")
        return

    # Process sequentially (GPU memory / TF model shared)
    total_files = len(file_list)
    total_windows = 0
    processed = 0
    errors = 0
    current_species = None

    for i, (src_path, out_path, species, rec_id, ch) in enumerate(file_list):
        if species != current_species:
            current_species = species
            _flush_print(f"\n[{species}] Processing sources...")

        result = _process_source_file(
            infer, src_path, out_path, args.device,
            tweetynet=tweetynet, tweetynet_device=torch_device,
        )

        if result is None:
            errors += 1
        else:
            processed += 1
            total_windows += result["n_windows"]

        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            _flush_print(
                f"  Progress: {i + 1}/{total_files} "
                f"({rate:.1f} files/s, {total_windows} windows, {errors} errors)"
            )

    elapsed = time.time() - t0
    _flush_print(f"\n{'=' * 60}")
    _flush_print(f"Pre-compute complete in {elapsed:.1f}s")
    _flush_print(f"  Total files processed: {processed}")
    _flush_print(f"  Total windows:         {total_windows}")
    _flush_print(f"  Skipped (missing):     {len(missing)}")
    _flush_print(f"  Errors:                {errors}")
    _flush_print(f"  Output directory:      {EMBEDDINGS_DIR}")
    _flush_print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute Perch v2 embeddings, SNR, and NDSI for Bird-MixIT sources."
    )
    parser.add_argument(
        "--species", type=str, default=None, help="Filter by species code"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device for TensorFlow (default: gpu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if .npz already exists",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_precompute(args)
