"""
Bird-MixIT source separation + segment extraction pipeline (v3).

Early-routing design: BirdNET is run on every registered species first, then
species are routed to B-1 (BirdNET-sufficient) or B-2 (prototype required).
Perch v2 embeddings and prototype clustering are computed only for B-2 species.

Pipeline stages (in order):
  1. separate                 Bird-MixIT 4-source separation (parallel, CPU)
  2. compute-acoustic-features SNR / NDSI / bird_ratio (librosa + TweetyNet)
  3. birdnet-score             BirdNET v2.4 per 4ch (BirdNET-registered species)
  4. route-species             B-1 vs B-2 routing (writes species_routing.csv)
  5. compute-perch-embeddings  Perch v2 embeddings (B-2 species only)
  6. build-prototypes          HDBSCAN prototype clustering (B-2 species only)
  7. channel-select            Gate + rank (B-1: BirdNET conf, B-2: proto sim)
  8. segment                   TweetyNet segmentation of focal channel
  9. select                    Rank bouts and export final WAVs

Top-level shortcuts:
  select                       runs 3→9 with resume/cache
  all                          runs 1→9 end-to-end

Usage:
  python birdmixit_pipeline.py separate [--species CODE] [--workers N] [--limit N]
  python birdmixit_pipeline.py compute-acoustic-features [--species CODE] [--force]
  python birdmixit_pipeline.py birdnet-score [--species CODE] [--force]
  python birdmixit_pipeline.py route-species [--birdnet-hit-rate-min 0.3] [--birdnet-hit-count-min 10]
  python birdmixit_pipeline.py compute-perch-embeddings [--species CODE] [--force] [--device gpu]
  python birdmixit_pipeline.py build-prototypes [--species CODE] [--force]
  python birdmixit_pipeline.py channel-select [--species CODE] [--force]
  python birdmixit_pipeline.py segment [--species CODE] [--force] [--device cuda]
  python birdmixit_pipeline.py select [--species CODE] [--target-n 75] [--max-per-recording 5]
  python birdmixit_pipeline.py all [--species CODE] [--workers N]
"""

# TF environment: force CPU for TF1 (Bird-MixIT), suppress logs
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Note: CUDA_VISIBLE_DEVICES is set to "-1" only in TF worker processes (see _init_worker)
# to avoid blocking PyTorch CUDA access in the main process.

import argparse
import json
import multiprocessing
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_opening, label

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
MIXIT_MODEL_DIR = NAS_BASE / "models" / "bird_mixit" / "output_sources4"
TWEETYNET_MODEL_PATH = STEP_DIR / "models" / "tweetynet_r2_best.pt"
MIXIT_SR = 22050
# Cap separation input to the first N seconds. Long clips blow up Bird-MixIT's
# TDCN++ intermediate tensors (observed: 45-min input killed worker at ~32 GB
# RSS). 10 min is plenty to harvest 50-100 clean segments per recording.
MAX_SEPARATE_DURATION_S = 10 * 60
SOURCES_DIR = NAS_BASE / "segments" / "birdmixit_sources"
SELECTED_DIR = NAS_BASE / "segments" / "birdmixit_selected"
RESULTS_CSV = STEP_DIR / "birdmixit_pipeline_results.csv"

# Intermediate result directories
ACOUSTIC_FEATURES_DIR = NAS_BASE / "segments" / "acoustic_features"
PERCH_EMBEDDINGS_DIR = NAS_BASE / "segments" / "perch_embeddings"
BIRDNET_SCORES_DIR = NAS_BASE / "segments" / "birdnet_scores"
ROUTING_CSV = NAS_BASE / "segments" / "species_routing.csv"
CHANNEL_SELECT_DIR = NAS_BASE / "segments" / "channel_selection"
SEGMENTS_DIR = NAS_BASE / "segments" / "tweetynet_segments"

# Perch v2 parameters (for species prototype channel selection)
PERCH_SR = 32000
PERCH_WINDOW_S = 5.0
PERCH_WINDOW_SAMPLES = int(PERCH_SR * PERCH_WINDOW_S)  # 160000
PERCH_MODEL_URL = "https://tfhub.dev/google/bird-vocalization-classifier/2"
PROTOTYPES_DIR = STEP_DIR / "species_prototypes"

# 5-second window at MIXIT_SR for SNR/NDSI/bird_ratio computation
MIXIT_WINDOW_SAMPLES = int(PERCH_WINDOW_S * MIXIT_SR)  # 110250
MIN_WINDOW_SAMPLES_MIXIT = MIXIT_SR  # 1 second minimum

# NDSI STFT parameters at MIXIT_SR
NDSI_N_FFT = 1024
NDSI_HOP = 512
NDSI_FREQ_RES = MIXIT_SR / NDSI_N_FFT  # ~21.5 Hz/bin

# SNR frame parameters at MIXIT_SR
SNR_FRAME_LEN = int(0.025 * MIXIT_SR)  # 25 ms
SNR_HOP_LEN = int(0.010 * MIXIT_SR)    # 10 ms

# BirdNET parameters (for species classification channel scoring)
BIRDNET_SR = 48000
BIRDNET_SEGMENT_S = 3.0
BIRDNET_GATE_THRESHOLD = 0.2
BIRDNET_BATCH_SIZE = 20  # recordings per subprocess to avoid FD leaks

# Routing defaults (B-1 vs B-2 thresholds for route-species)
DEFAULT_BIRDNET_HIT_RATE_MIN = 0.3  # fraction of recordings with conf >= gate
DEFAULT_BIRDNET_HIT_COUNT_MIN = 10  # absolute minimum count of such recordings


# ---------------------------------------------------------------------------
# TweetyNet constants (must match prototype_tweetynet.py)
# ---------------------------------------------------------------------------

TWEETYNET_SR = 32000
TWEETYNET_N_FFT = 1024
TWEETYNET_HOP = 320  # 10ms hop at 32kHz
TWEETYNET_N_MELS = 128
TWEETYNET_FMIN = 150
TWEETYNET_FMAX = 12000
TWEETYNET_FRAME_DUR = TWEETYNET_HOP / TWEETYNET_SR  # 0.01s
TWEETYNET_CONTEXT = 200  # frames (~2s)


# ---------------------------------------------------------------------------
# TweetyNet model (lightweight reimplementation, same as prototype_tweetynet)
# ---------------------------------------------------------------------------


class TweetyNet(nn.Module):
    """Lightweight TweetyNet for binary frame-level segmentation."""

    def __init__(
        self,
        num_classes: int = 2,
        num_freqbins: int = TWEETYNET_N_MELS,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size=(5, 5),
        pool_size=(4, 1),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, conv1_filters, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.pool2 = nn.MaxPool2d(pool_size)
        freq_after_pool = num_freqbins // (pool_size[0] ** 2)
        rnn_input_size = conv2_filters * max(freq_after_pool, 1)
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x


def _load_tweetynet(device: torch.device) -> nn.Module:
    """Load the self-trained TweetyNet model."""
    model = TweetyNet(num_classes=2, num_freqbins=TWEETYNET_N_MELS).to(device)
    model.load_state_dict(
        torch.load(str(TWEETYNET_MODEL_PATH), map_location=device, weights_only=True)
    )
    model.eval()
    return model


def _tweetynet_predict(model: nn.Module, audio_22k: np.ndarray, device: torch.device) -> np.ndarray:
    """Run TweetyNet on 22kHz audio from Bird-MixIT source.

    Resamples to 32kHz, computes mel spectrogram, predicts frame-level P(bird).

    Returns:
        (n_frames,) array of P(bird) probabilities at TWEETYNET_SR frame rate.
    """
    # Resample 22050 -> 32000
    y32 = librosa.resample(audio_22k, orig_sr=MIXIT_SR, target_sr=TWEETYNET_SR)

    # Mel spectrogram + normalization
    S = librosa.feature.melspectrogram(
        y=y32, sr=TWEETYNET_SR, n_fft=TWEETYNET_N_FFT, hop_length=TWEETYNET_HOP,
        n_mels=TWEETYNET_N_MELS, fmin=TWEETYNET_FMIN, fmax=TWEETYNET_FMAX, power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mean, std = S_db.mean(), S_db.std() + 1e-8
    spec = (S_db - mean) / std  # (n_mels, n_frames)

    n_frames = spec.shape[1]
    ctx = TWEETYNET_CONTEXT

    if n_frames <= ctx:
        x = torch.from_numpy(spec[np.newaxis, np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
        return F.softmax(logits, dim=1)[0, 1].cpu().numpy()

    # Sliding window with 50% overlap
    stride = ctx // 2
    prob_sum = np.zeros(n_frames, dtype=np.float64)
    count = np.zeros(n_frames, dtype=np.float64)

    for start in range(0, n_frames - ctx + 1, stride):
        end = start + ctx
        patch = spec[:, start:end]
        x = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        prob_sum[start:end] += probs
        count[start:end] += 1.0

    # Handle tail
    if count[-1] == 0:
        tail_start = n_frames - ctx
        patch = spec[:, tail_start:n_frames]
        x = torch.from_numpy(patch[np.newaxis, np.newaxis].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy()
        offset = n_frames - tail_start
        prob_sum[tail_start:n_frames] += probs[:offset]
        count[tail_start:n_frames] += 1.0

    return prob_sum / np.maximum(count, 1.0)


def _probs_to_segments(probs, threshold=0.7, min_dur=0.03, merge_gap=0.05):
    """Convert frame-level P(bird) to (onset, offset) segments in seconds."""
    active = probs >= threshold

    gap_frames = int(merge_gap / TWEETYNET_FRAME_DUR)
    if gap_frames > 1:
        active = binary_closing(active, structure=np.ones(gap_frames))
    min_frames = int(min_dur / TWEETYNET_FRAME_DUR)
    if min_frames > 1:
        active = binary_opening(active, structure=np.ones(min_frames))

    lab_arr, n = label(active)
    segments = []
    for k in range(1, n + 1):
        idx = np.where(lab_arr == k)[0]
        onset = idx[0] * TWEETYNET_FRAME_DUR
        offset = (idx[-1] + 1) * TWEETYNET_FRAME_DUR
        if offset - onset >= min_dur:
            segments.append((round(onset, 4), round(offset, 4)))
    return segments


# ---------------------------------------------------------------------------
# Bout grouping (note-level -> bout-level)
# ---------------------------------------------------------------------------


def _group_notes_into_bouts(notes, max_gap=0.4, max_dur=8.0, max_silence=0.6):
    """Group notes into bouts using gap-based merging + recursive splitting."""
    if not notes:
        return []
    raw_bouts = [[notes[0]]]
    for note in notes[1:]:
        gap = note[0] - raw_bouts[-1][-1][1]
        if gap <= max_gap:
            raw_bouts[-1].append(note)
        else:
            raw_bouts.append([note])
    final_bouts = []
    for bout_notes in raw_bouts:
        final_bouts.extend(_split_bout(bout_notes, max_dur, max_silence))
    return final_bouts


def _split_bout(notes, max_dur, max_silence):
    """Recursively split a bout at the largest gap if constraints violated."""
    bout = _make_bout_dict(notes)
    duration = bout["bout_offset"] - bout["bout_onset"]
    if duration <= max_dur and bout["silence_ratio"] <= max_silence:
        return [bout]
    if len(notes) <= 1:
        onset, offset = notes[0]
        if offset - onset <= max_dur:
            return [bout]
        chunks = []
        t = onset
        while t < offset:
            end = min(t + max_dur, offset)
            if end - t >= 0.1:
                chunks.append(_make_bout_dict([(t, end)]))
            t = end
        return chunks if chunks else [bout]
    gaps = [(notes[i + 1][0] - notes[i][1], i) for i in range(len(notes) - 1)]
    gaps.sort(reverse=True)
    _, split_idx = gaps[0]
    result = []
    result.extend(_split_bout(notes[:split_idx + 1], max_dur, max_silence))
    result.extend(_split_bout(notes[split_idx + 1:], max_dur, max_silence))
    return result


def _make_bout_dict(notes):
    """Create a bout dict from a list of (onset, offset) tuples."""
    bout_onset = notes[0][0]
    bout_offset = notes[-1][1]
    total_note_dur = sum(off - on for on, off in notes)
    bout_dur = bout_offset - bout_onset
    silence_ratio = 1.0 - (total_note_dur / bout_dur) if bout_dur > 0 else 0.0
    return {
        "bout_onset": round(bout_onset, 4),
        "bout_offset": round(bout_offset, 4),
        "notes": [(round(on, 4), round(off, 4)) for on, off in notes],
        "n_notes": len(notes),
        "total_note_duration": round(total_note_dur, 4),
        "silence_ratio": round(silence_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Bioacoustic quality helpers (v3)
# ---------------------------------------------------------------------------


def _bioacoustic_quality(snr_db: float, ndsi: float) -> float:
    """Compute bioacoustic quality score from SNR and NDSI.

    SNR is sigmoid-normalized around 10 dB; NDSI is linearly mapped to [0.3, 1.0]
    (floor at 0.3 accommodates low-frequency species with negative NDSI).
    """
    snr_norm = 1.0 / (1.0 + np.exp(-0.3 * (snr_db - 10.0)))
    ndsi_norm = max((ndsi + 1.0) / 2.0, 0.3)
    return float(snr_norm * ndsi_norm)


def _load_acoustic_features(
    species: str, rec_id: str, ch: int
) -> dict | None:
    """Load precomputed acoustic features (SNR/NDSI/bird_ratio) from npz.

    Returns dict with keys 'snr', 'ndsi', 'bird_ratio', 'window_starts',
    or None if file is missing.
    """
    safe_id = rec_id.replace(":", "_")
    path = ACOUSTIC_FEATURES_DIR / species / f"{safe_id}_src{ch}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "snr": data["snr"],                       # (n_windows,)
        "ndsi": data["ndsi"],                     # (n_windows,)
        "bird_ratio": data["bird_ratio"],         # (n_windows,)
        "window_starts": data["window_starts"],   # (n_windows,)
    }


def _load_perch_embeddings_features(
    species: str, rec_id: str, ch: int
) -> dict | None:
    """Load precomputed Perch v2 embeddings from npz.

    Returns dict with keys 'embeddings' and 'window_starts', or None if
    file is missing.
    """
    safe_id = rec_id.replace(":", "_")
    path = PERCH_EMBEDDINGS_DIR / species / f"{safe_id}_src{ch}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "embeddings": data["embeddings"],        # (n_windows, 1280)
        "window_starts": data["window_starts"],  # (n_windows,)
    }


def _channel_quality_from_features(acoustic: dict | None) -> float:
    """Compute bioacoustic quality for a channel from precomputed features.

    Uses median SNR/NDSI across windows as representative values.
    """
    if acoustic is None or len(acoustic["snr"]) == 0:
        return 0.0
    median_snr = float(np.median(acoustic["snr"]))
    median_ndsi = float(np.median(acoustic["ndsi"]))
    return _bioacoustic_quality(median_snr, median_ndsi)


def _compute_snr_ndsi(audio: np.ndarray, sr: int = MIXIT_SR) -> tuple[float, float]:
    """Compute SNR (dB) and NDSI from audio segment.

    SNR: ratio of signal power (above median+MAD threshold) to noise power.
    NDSI: (biophony - technophony) / (biophony + technophony),
          where biophony = energy in 2-8 kHz, technophony = 0.5-2 kHz.
    """
    if len(audio) < sr // 10:
        return 0.0, 0.0

    # SNR via energy ratio
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    frame_rms = np.array([
        np.sqrt(np.mean(audio[j * hop: j * hop + frame_len] ** 2) + 1e-10)
        for j in range(n_frames)
    ])
    median_rms = np.median(frame_rms)
    mad = np.median(np.abs(frame_rms - median_rms))
    threshold = median_rms + 1.0 * mad
    signal_mask = frame_rms > threshold
    if signal_mask.sum() == 0 or (~signal_mask).sum() == 0:
        snr_db = 0.0
    else:
        signal_power = np.mean(frame_rms[signal_mask] ** 2)
        noise_power = np.mean(frame_rms[~signal_mask] ** 2) + 1e-20
        snr_db = float(10.0 * np.log10(signal_power / noise_power))

    # NDSI via spectral energy bands
    S = np.abs(librosa.stft(audio, n_fft=1024, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    bio_mask = (freqs >= 2000) & (freqs <= 8000)
    tech_mask = (freqs >= 500) & (freqs < 2000)
    biophony = float(np.sum(S[bio_mask, :]))
    technophony = float(np.sum(S[tech_mask, :]))
    denom = biophony + technophony
    ndsi = float((biophony - technophony) / denom) if denom > 1e-10 else 0.0

    return snr_db, ndsi


def _compute_bout_quality(bout_audio: np.ndarray, sr: int = MIXIT_SR) -> float:
    """Compute bioacoustic quality for a bout's concatenated note audio."""
    snr_db, ndsi = _compute_snr_ndsi(bout_audio, sr)
    return _bioacoustic_quality(snr_db, ndsi)


# ---------------------------------------------------------------------------
# Windowed acoustic-feature computation (SNR/NDSI/bird_ratio)
# ---------------------------------------------------------------------------


def _window_snr(y: np.ndarray) -> float:
    """Per-window SNR (dB) from raw audio at MIXIT_SR.

    Frame-level RMS, noise floor = median, signal = RMS of frames above
    (median + 2*MAD). Returns 0.0 when no signal frames are detected.
    """
    frames = librosa.util.frame(y, frame_length=SNR_FRAME_LEN, hop_length=SNR_HOP_LEN)
    if frames.shape[1] == 0:
        return 0.0
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    noise_floor = np.median(frame_rms)
    mad = np.median(np.abs(frame_rms - noise_floor))
    threshold = noise_floor + 2.0 * mad
    signal_mask = frame_rms > threshold
    if not np.any(signal_mask):
        return 0.0
    signal_rms = np.sqrt(np.mean(frame_rms[signal_mask] ** 2))
    return float(20.0 * np.log10(signal_rms / (noise_floor + 1e-10)))


def _window_ndsi(y: np.ndarray) -> float:
    """Per-window NDSI: biophony 2-10 kHz vs anthrophony 1-2 kHz."""
    S = np.abs(librosa.stft(y, n_fft=NDSI_N_FFT, hop_length=NDSI_HOP)) ** 2
    anthro_lo = int(np.ceil(1000.0 / NDSI_FREQ_RES))
    anthro_hi = int(np.floor(2000.0 / NDSI_FREQ_RES))
    bio_lo = int(np.ceil(2000.0 / NDSI_FREQ_RES))
    bio_hi = int(np.floor(10000.0 / NDSI_FREQ_RES))
    n_bins = S.shape[0]
    anthro_hi = min(anthro_hi, n_bins - 1)
    bio_hi = min(bio_hi, n_bins - 1)
    sum_anthro = float(np.sum(S[anthro_lo:anthro_hi + 1, :]))
    sum_bio = float(np.sum(S[bio_lo:bio_hi + 1, :]))
    denom = sum_bio + sum_anthro
    return float((sum_bio - sum_anthro) / (denom + 1e-10))


def _compute_acoustic_features_for_source(
    src_path: Path, tweetynet, tweetynet_device
) -> dict | None:
    """Compute per-window SNR / NDSI / bird_ratio for one separated source WAV.

    Returns dict with numpy arrays (snr, ndsi, bird_ratio, window_starts) or
    None if the file cannot be read or has no valid windows.
    """
    try:
        y, sr = sf.read(str(src_path), dtype="float32")
    except Exception as e:
        _flush_print(f"  WARN: cannot read {src_path.name}: {e}")
        return None
    if sr != MIXIT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MIXIT_SR)
    if y.ndim > 1:
        y = y.mean(axis=1)

    snrs, ndsis, bird_ratios, starts = [], [], [], []
    for start in range(0, len(y), MIXIT_WINDOW_SAMPLES):
        chunk = y[start: start + MIXIT_WINDOW_SAMPLES]
        if len(chunk) < MIN_WINDOW_SAMPLES_MIXIT:
            break
        snrs.append(_window_snr(chunk))
        ndsis.append(_window_ndsi(chunk))
        if tweetynet is not None:
            probs = _tweetynet_predict(tweetynet, chunk, tweetynet_device)
            bird_ratios.append(float((probs > 0.5).mean()))
        else:
            bird_ratios.append(-1.0)
        starts.append(start)

    if not starts:
        return None

    return {
        "snr": np.array(snrs, dtype=np.float32),
        "ndsi": np.array(ndsis, dtype=np.float32),
        "bird_ratio": np.array(bird_ratios, dtype=np.float32),
        "window_starts": np.array(starts, dtype=np.int32),
    }


def run_compute_acoustic_features(args):
    """Stage 2: SNR / NDSI / bird_ratio for every (species, recording, channel).

    Lightweight — only librosa + TweetyNet (PyTorch). Results are saved to
    acoustic_features/{species}/{rec}_src{ch}.npz and consumed by later
    stages (build-prototypes, channel-select) as the source of bioacoustic
    quality features.
    """
    t0 = time.time()
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    force = getattr(args, "force", False)

    # Enumerate all source files up-front for resume support
    tasks = []
    for species in species_codes:
        df_sp = df[df["ebird_species_code"] == species]
        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")
            for ch in range(4):
                src = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
                out = ACOUSTIC_FEATURES_DIR / species / f"{safe_id}_src{ch}.npz"
                if not src.exists():
                    continue
                if out.exists() and not force:
                    continue
                tasks.append((species, rec_id, ch, src, out))

    if not tasks:
        _flush_print("No acoustic features to compute (all up to date).")
        return

    _flush_print(f"Loading TweetyNet for bird_ratio computation...")
    device_name = getattr(args, "device", "cuda")
    device = torch.device(
        device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu"
    )
    tweetynet = _load_tweetynet(device)

    _flush_print(f"Computing acoustic features for {len(tasks)} source files...")

    total = len(tasks)
    n_ok = 0
    n_err = 0
    current_species = None
    for i, (species, rec_id, ch, src, out) in enumerate(tasks):
        if species != current_species:
            current_species = species
            _flush_print(f"\n[{species}]")

        result = _compute_acoustic_features_for_source(src, tweetynet, device)
        if result is None:
            n_err += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out),
            snr=result["snr"],
            ndsi=result["ndsi"],
            bird_ratio=result["bird_ratio"],
            window_starts=result["window_starts"],
        )
        n_ok += 1

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            _flush_print(
                f"  Progress: {i + 1}/{total} ({rate:.1f} files/s, "
                f"{n_err} errors)"
            )

    elapsed = time.time() - t0
    _flush_print(
        f"\nDone: {n_ok} files written in {elapsed:.1f}s ({n_err} errors) "
        f"-> {ACOUSTIC_FEATURES_DIR}"
    )


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
# TF1 session management
# ---------------------------------------------------------------------------


def _create_tf_session():
    """Create TF1 session with Bird-MixIT model. Call once per process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(
            str(MIXIT_MODEL_DIR / "inference.meta")
        )

    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    config.inter_op_parallelism_threads = 1
    config.intra_op_parallelism_threads = 2
    sess = tf.compat.v1.Session(graph=graph, config=config)
    saver.restore(sess, str(MIXIT_MODEL_DIR / "model.ckpt-3223090"))

    input_t = graph.get_tensor_by_name("input_audio/receiver_audio:0")
    output_t = graph.get_tensor_by_name("denoised_waveforms:0")
    return sess, input_t, output_t


def separate_audio(sess, input_t, output_t, audio_22k):
    """Run Bird-MixIT separation. Returns (4, samples) array."""
    inp = audio_22k[np.newaxis, np.newaxis, :].astype(np.float32)
    result = sess.run(output_t, feed_dict={input_t: inp})
    return result[0]  # (4, samples)


# ---------------------------------------------------------------------------
# Channel analysis
# ---------------------------------------------------------------------------


def analyze_channels(sources, sr=MIXIT_SR):
    """Compute per-channel metrics for source selection.

    Returns list of 4 dicts with:
        rms, spectral_centroid, activity_ratio, combined_score.
    Combined = 0.4*norm_rms + 0.3*norm_centroid + 0.3*norm_activity
    """
    n_ch = sources.shape[0]
    metrics = []

    for i in range(n_ch):
        ch = sources[i]
        rms = float(np.sqrt(np.mean(ch**2)) + 1e-10)

        # Spectral centroid (mean over time)
        cent = librosa.feature.spectral_centroid(y=ch, sr=sr, n_fft=1024, hop_length=512)
        spectral_centroid = float(np.mean(cent))

        # Activity ratio: fraction of frames above energy threshold
        frame_len = int(0.025 * sr)  # 25ms
        hop = int(0.010 * sr)  # 10ms
        n_frames = max(1, (len(ch) - frame_len) // hop + 1)
        frame_rms = np.array(
            [
                np.sqrt(np.mean(ch[j * hop : j * hop + frame_len] ** 2) + 1e-10)
                for j in range(n_frames)
            ]
        )
        median_rms = np.median(frame_rms)
        mad = np.median(np.abs(frame_rms - median_rms))
        threshold = median_rms + 1.0 * mad
        activity_ratio = float(np.mean(frame_rms > threshold))

        metrics.append(
            {
                "rms": rms,
                "spectral_centroid": spectral_centroid,
                "activity_ratio": activity_ratio,
            }
        )

    # Normalize and compute combined score
    rms_vals = np.array([m["rms"] for m in metrics])
    cent_vals = np.array([m["spectral_centroid"] for m in metrics])
    act_vals = np.array([m["activity_ratio"] for m in metrics])

    def _norm(arr):
        rng = arr.max() - arr.min()
        if rng < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - arr.min()) / rng

    rms_n = _norm(rms_vals)
    cent_n = _norm(cent_vals)
    act_n = _norm(act_vals)

    for i in range(n_ch):
        metrics[i]["combined_score"] = float(
            0.4 * rms_n[i] + 0.3 * cent_n[i] + 0.3 * act_n[i]
        )

    return metrics


# ---------------------------------------------------------------------------
# Perch v2 species prototype
# ---------------------------------------------------------------------------


def _load_perch_model():
    """Load Google Perch v2 model from TensorFlow Hub."""
    import tensorflow_hub as hub

    _flush_print("Loading Perch v2 model from TF Hub...")
    model = hub.load(PERCH_MODEL_URL)
    infer = model.signatures["serving_default"]
    _flush_print("Perch v2 loaded (embedding dim=1280)")
    return infer


def _extract_perch_outputs(
    infer, audio: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract both embedding and classification logits from Perch v2.

    Audio is padded/trimmed to 5s (160000 samples at PERCH_SR).

    Returns:
        (embedding, logits) where embedding is L2-normalized (1280,)
        and logits are raw classification outputs (10932,).
    """
    import tensorflow as tf

    if len(audio) < PERCH_WINDOW_SAMPLES:
        audio = np.pad(audio, (0, PERCH_WINDOW_SAMPLES - len(audio)))
    audio = audio[:PERCH_WINDOW_SAMPLES]

    inp = tf.constant(audio[np.newaxis].astype(np.float32))
    result = infer(inputs=inp)
    logits = result["output_0"].numpy()[0]  # (10932,)
    emb = result["output_1"].numpy()[0]     # (1280,)

    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb, logits


def _extract_perch_embedding(infer, audio: np.ndarray) -> np.ndarray:
    """Extract L2-normalized Perch v2 embedding (1280-d) from audio at PERCH_SR.

    Audio is padded/trimmed to 5s (160000 samples).
    """
    emb, _ = _extract_perch_outputs(infer, audio)
    return emb


def _extract_recording_embeddings(infer, audio_path: str, sr: int = PERCH_SR) -> list[np.ndarray]:
    """Extract Perch embeddings from non-overlapping 5s windows of a recording.

    Returns list of L2-normalized 1280-d vectors.
    """
    y, _ = librosa.load(audio_path, sr=sr)
    if len(y) < sr:  # skip < 1s
        return []

    embeddings = []
    for start in range(0, len(y), PERCH_WINDOW_SAMPLES):
        chunk = y[start : start + PERCH_WINDOW_SAMPLES]
        if len(chunk) < sr:  # skip < 1s tail
            break
        emb = _extract_perch_embedding(infer, chunk)
        embeddings.append(emb)
    return embeddings


def _compute_perch_embeddings_for_source(
    infer, src_path: Path
) -> dict | None:
    """Compute Perch v2 embeddings for every 5s window of a separated source.

    Returns dict {embeddings, window_starts} (window_starts in MIXIT_SR samples
    for consistency with acoustic_features) or None on error.
    """
    try:
        y, sr = sf.read(str(src_path), dtype="float32")
    except Exception as e:
        _flush_print(f"  WARN: cannot read {src_path.name}: {e}")
        return None
    if sr != MIXIT_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=MIXIT_SR)
    if y.ndim > 1:
        y = y.mean(axis=1)

    y_perch = librosa.resample(y, orig_sr=MIXIT_SR, target_sr=PERCH_SR)

    embs, starts = [], []
    for start in range(0, len(y), MIXIT_WINDOW_SAMPLES):
        chunk_mixit = y[start: start + MIXIT_WINDOW_SAMPLES]
        if len(chunk_mixit) < MIN_WINDOW_SAMPLES_MIXIT:
            break
        p_start = int(start * PERCH_SR / MIXIT_SR)
        chunk_perch = y_perch[p_start: p_start + PERCH_WINDOW_SAMPLES]
        if len(chunk_perch) < MIXIT_SR:  # <1s tail
            break
        embs.append(_extract_perch_embedding(infer, chunk_perch))
        starts.append(start)

    if not starts:
        return None
    return {
        "embeddings": np.array(embs, dtype=np.float32),
        "window_starts": np.array(starts, dtype=np.int32),
    }


def run_compute_perch_embeddings(args):
    """Stage 5: Perch v2 embeddings, restricted to B-2 species by default.

    Pass --all-species to force embedding every species (e.g. for research).
    Output: perch_embeddings/{species}/{rec}_src{ch}.npz with keys
    'embeddings' (n_windows, 1280) and 'window_starts' (n_windows,).
    """
    t0 = time.time()
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    force = getattr(args, "force", False)
    include_all = getattr(args, "all_species", False)

    if not include_all:
        routing = _load_species_routing()
        if routing is None:
            _flush_print(
                "ERROR: species_routing.csv not found. Run 'route-species' "
                "first, or pass --all-species to embed everything."
            )
            sys.exit(1)
        b2_set = set(routing[routing["route"] == "B2"]["species_code"].tolist())
        species_codes = [s for s in species_codes if s in b2_set]
        _flush_print(
            f"Routing-restricted: {len(species_codes)} B-2 species selected"
        )
        if not species_codes:
            _flush_print("No B-2 species — nothing to do.")
            return

    # Enumerate all source files
    tasks = []
    for species in species_codes:
        df_sp = df[df["ebird_species_code"] == species]
        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")
            for ch in range(4):
                src = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
                out = PERCH_EMBEDDINGS_DIR / species / f"{safe_id}_src{ch}.npz"
                if not src.exists():
                    continue
                if out.exists() and not force:
                    continue
                tasks.append((species, rec_id, ch, src, out))

    if not tasks:
        _flush_print("No Perch embeddings to compute (all up to date).")
        return

    # Configure TF
    device_name = getattr(args, "device", "gpu")
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    if device_name == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        _flush_print(
            f"TF GPU devices: {[g.name for g in gpus] or 'none (CPU fallback)'}"
        )

    infer = _load_perch_model()

    _flush_print(f"Computing Perch embeddings for {len(tasks)} source files...")
    total = len(tasks)
    n_ok = 0
    n_err = 0
    current_species = None
    for i, (species, rec_id, ch, src, out) in enumerate(tasks):
        if species != current_species:
            current_species = species
            _flush_print(f"\n[{species}]")
        result = _compute_perch_embeddings_for_source(infer, src)
        if result is None:
            n_err += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out),
            embeddings=result["embeddings"],
            window_starts=result["window_starts"],
        )
        n_ok += 1
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            _flush_print(
                f"  Progress: {i + 1}/{total} ({rate:.1f} files/s, "
                f"{n_err} errors)"
            )

    elapsed = time.time() - t0
    _flush_print(
        f"\nDone: {n_ok} files written in {elapsed:.1f}s ({n_err} errors) "
        f"-> {PERCH_EMBEDDINGS_DIR}"
    )


def run_build_prototypes(args):
    """Stage 6: build multi-prototype species representations via HDBSCAN.

    Only processes B-2 species (those that need prototypes for channel
    selection). Pass --all-species to override routing and build prototypes
    for every species.

    For each eligible species:
      1. Load Perch v2 embeddings (perch_embeddings/) and acoustic features
         (acoustic_features/) for all 4 channels of all recordings
      2. Filter windows where bird_ratio > 0 (TweetyNet detected bird frames);
         fall back to SNR+NDSI if bird_ratio is unavailable
      3. HDBSCAN clustering on L2-normalized embeddings
      4. Select top-K clusters (size >= 5% of total, cumulative coverage
         <= 90%, max K=4)
      5. Save cluster centroids as multi-prototype to
         species_prototypes/{species_code}.npz
    """
    from sklearn.cluster import HDBSCAN

    MAX_K = 4               # max number of prototypes per species
    MIN_CLUSTER_FRAC = 0.05 # cluster must be >= 5% of total embeddings
    COVERAGE_TARGET = 0.90  # stop adding clusters after 90% coverage
    SNR_THRESHOLD = 5.0     # minimum SNR in dB (fallback filter)
    NDSI_THRESHOLD = -0.5   # minimum NDSI (fallback filter)
    BIRD_RATIO_MIN = 0.0    # require bird_ratio > 0

    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    include_all = getattr(args, "all_species", False)

    if not include_all:
        routing = _load_species_routing()
        if routing is None:
            _flush_print(
                "ERROR: species_routing.csv not found. Run 'route-species' "
                "first, or pass --all-species to build prototypes everywhere."
            )
            sys.exit(1)
        b2_set = set(routing[routing["route"] == "B2"]["species_code"].tolist())
        species_codes = [s for s in species_codes if s in b2_set]
        _flush_print(
            f"Routing-restricted: {len(species_codes)} B-2 species selected"
        )
        if not species_codes:
            _flush_print("No B-2 species — nothing to do.")
            return

    PROTOTYPES_DIR.mkdir(parents=True, exist_ok=True)

    _flush_print(f"Building prototypes for {len(species_codes)} species "
                 f"(filter: bird_ratio>{BIRD_RATIO_MIN})...")

    stats = {"ok": 0, "few": 0, "skip": 0}

    for species in species_codes:
        out_path = PROTOTYPES_DIR / f"{species}.npz"
        if out_path.exists() and not getattr(args, "force", False):
            _flush_print(f"  {species}: EXISTS (skip)")
            stats["ok"] += 1
            continue

        df_sp = df[df["ebird_species_code"] == species]
        all_embs = []

        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            # Load Perch embeddings + acoustic features from all 4 channels
            for ch in range(4):
                perch = _load_perch_embeddings_features(species, rec_id, ch)
                if perch is None:
                    continue
                acoustic = _load_acoustic_features(species, rec_id, ch)
                embs = perch["embeddings"]
                if acoustic is not None and len(acoustic["bird_ratio"]) == len(embs):
                    # Primary filter: TweetyNet bird_ratio
                    mask = acoustic["bird_ratio"] > BIRD_RATIO_MIN
                elif acoustic is not None and len(acoustic["snr"]) == len(embs):
                    # Fallback: SNR + NDSI
                    mask = (acoustic["snr"] > SNR_THRESHOLD) & (
                        acoustic["ndsi"] > NDSI_THRESHOLD
                    )
                else:
                    # No acoustic features — accept all windows
                    mask = np.ones(len(embs), dtype=bool)
                if mask.any():
                    all_embs.extend(embs[mask])

        if len(all_embs) < 5:
            if all_embs:
                proto = np.mean(all_embs, axis=0)
                proto /= np.linalg.norm(proto) + 1e-8
                np.savez(out_path, prototypes=proto[np.newaxis],
                         n_embeddings=len(all_embs), n_prototypes=1, method="mean")
                _flush_print(f"  {species}: {len(all_embs)} embeddings (mean, no clustering)")
                stats["few"] += 1
            else:
                _flush_print(f"  {species}: no audio files found")
                stats["skip"] += 1
            continue

        X = np.stack(all_embs)
        n = len(X)

        # HDBSCAN with adaptive parameters
        min_cluster_size = max(5, min(30, round(0.03 * n)))
        min_samples = max(3, min(15, round(0.5 * min_cluster_size)))

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
        )
        labels = clusterer.fit_predict(X)

        unique_labels = set(labels) - {-1}
        if not unique_labels:
            proto = np.median(X, axis=0)
            proto /= np.linalg.norm(proto) + 1e-8
            np.savez(out_path, prototypes=proto[np.newaxis],
                     n_embeddings=n, n_prototypes=1, method="median-fallback")
            _flush_print(f"  {species}: {n} embs, all noise → median fallback")
            stats["few"] += 1
            continue

        # Rank clusters by size (descending), filter by min fraction
        cluster_sizes = {l: (labels == l).sum() for l in unique_labels}
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])

        selected_protos = []
        coverage = 0
        n_non_noise = sum(cluster_sizes.values())

        for cl_label, cl_size in sorted_clusters:
            if cl_size / n < MIN_CLUSTER_FRAC and len(selected_protos) > 0:
                break
            cl_embs = X[labels == cl_label]
            centroid = np.mean(cl_embs, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            selected_protos.append(centroid)
            coverage += cl_size / n_non_noise
            if len(selected_protos) >= MAX_K or coverage >= COVERAGE_TARGET:
                break

        prototypes_arr = np.stack(selected_protos)
        n_noise = (labels == -1).sum()
        np.savez(out_path, prototypes=prototypes_arr,
                 n_embeddings=n, n_prototypes=len(selected_protos), method="hdbscan")

        _flush_print(
            f"  {species}: {n} embs → {len(unique_labels)} clusters, "
            f"{len(selected_protos)} prototypes, noise={n_noise}"
        )
        stats["ok"] += 1

    _flush_print(
        f"\nDone: {stats['ok']} prototypes built, {stats['few']} few-data, "
        f"{stats['skip']} skipped → {PROTOTYPES_DIR}"
    )


def _load_species_prototypes(species_codes: list[str]) -> dict[str, np.ndarray]:
    """Load precomputed species prototypes from disk.

    Returns dict mapping species_code -> array of shape (K, 1280) with
    K L2-normalized prototype vectors. Missing prototypes are silently skipped.
    """
    prototypes = {}
    for sp in species_codes:
        p = PROTOTYPES_DIR / f"{sp}.npz"
        if p.exists():
            data = np.load(p)
            if "prototypes" in data:
                prototypes[sp] = data["prototypes"]
            elif "prototype" in data:
                # Legacy single-prototype format
                prototypes[sp] = data["prototype"][np.newaxis]
    return prototypes




def _birdnet_worker(args_dict: dict) -> list[float]:
    """Run BirdNET inference in an isolated subprocess to avoid FD leaks.

    BirdNET's TFLite interpreter leaks file descriptors. Running in a subprocess
    ensures all FDs are released when the process exits.
    """
    import birdnet as _bn

    sources = args_dict["sources"]
    target_sciname = args_dict["target_sciname"]

    model = _bn.load("acoustic", "2.4", "tf")
    lookup = {}
    for idx, name in enumerate(model._species_list):
        lookup[name.split("_")[0]] = idx

    target_idx = lookup.get(target_sciname)
    if target_idx is None:
        return [0.0] * sources.shape[0]

    segment_samples = int(BIRDNET_SEGMENT_S * BIRDNET_SR)

    audio_arrays = []
    ch_seg_counts = []
    for ch_idx in range(sources.shape[0]):
        ch_48k = librosa.resample(sources[ch_idx], orig_sr=MIXIT_SR, target_sr=BIRDNET_SR)

        count = 0
        for start in range(0, len(ch_48k), segment_samples):
            seg = ch_48k[start : start + segment_samples]
            if len(seg) < BIRDNET_SR:  # skip < 1s tail
                break
            if len(seg) < segment_samples:
                seg = np.pad(seg, (0, segment_samples - len(seg)))
            audio_arrays.append((seg, BIRDNET_SR))
            count += 1
        ch_seg_counts.append(count)

    if not audio_arrays:
        return [0.0] * sources.shape[0]

    result = model.predict_arrays(
        audio_arrays,
        top_k=10,
        default_confidence_threshold=0.0,
        n_workers=1,
        batch_size=len(audio_arrays),
        device="CPU",
    )

    ch_scores = []
    offset = 0
    for ch_idx in range(sources.shape[0]):
        max_score = 0.0
        for seg_i in range(ch_seg_counts[ch_idx]):
            inp_idx = offset + seg_i
            n_segs = result.species_probs.shape[1]
            for seg_idx in range(n_segs):
                seg_ids = result.species_ids[inp_idx, seg_idx, :]
                seg_probs = result.species_probs[inp_idx, seg_idx, :]
                mask = seg_ids == target_idx
                if mask.any():
                    prob = float(seg_probs[mask].max())
                    if prob > max_score:
                        max_score = prob
        ch_scores.append(max_score)
        offset += ch_seg_counts[ch_idx]

    return ch_scores


def _birdnet_subprocess_target(args_dict, queue):
    """Subprocess entry point for BirdNET inference."""
    try:
        result = _birdnet_worker(args_dict)
        queue.put(result)
    except Exception:
        queue.put([0.0] * 4)


def _birdnet_score_4ch(
    sources: np.ndarray,
    target_sciname: str,
) -> list[float]:
    """Score all 4 channels using BirdNET in an isolated subprocess.

    Runs BirdNET in a subprocess to avoid TFLite file descriptor leaks.
    Uses ALL non-overlapping 3s segments (no subsampling).
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_birdnet_subprocess_target,
        args=({"sources": sources, "target_sciname": target_sciname}, q),
    )
    p.start()
    try:
        result = q.get(timeout=120)
    except Exception:
        result = [0.0] * sources.shape[0]
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
    return result


def _birdnet_batch_subprocess_target(args_dict, queue):
    """Subprocess entry point for batched BirdNET inference across recordings."""
    try:
        import birdnet as _bn

        target_sciname = args_dict["target_sciname"]
        rec_entries = args_dict["rec_entries"]  # list of (rec_id, sources_4ch)

        model = _bn.load("acoustic", "2.4", "tf")
        lookup = {}
        for idx, name in enumerate(model._species_list):
            lookup[name.split("_")[0]] = idx

        target_idx = lookup.get(target_sciname)
        if target_idx is None:
            queue.put({rec_id: [0.0] * 4 for rec_id, _ in rec_entries})
            return

        segment_samples = int(BIRDNET_SEGMENT_S * BIRDNET_SR)
        results = {}

        for rec_id, sources in rec_entries:
            audio_arrays = []
            ch_seg_counts = []
            for ch_idx in range(sources.shape[0]):
                ch_48k = librosa.resample(sources[ch_idx], orig_sr=MIXIT_SR, target_sr=BIRDNET_SR)

                # Use ALL non-overlapping 3s segments (no subsampling)
                count = 0
                for start in range(0, len(ch_48k), segment_samples):
                    seg = ch_48k[start: start + segment_samples]
                    if len(seg) < BIRDNET_SR:  # skip < 1s tail
                        break
                    if len(seg) < segment_samples:
                        seg = np.pad(seg, (0, segment_samples - len(seg)))
                    audio_arrays.append((seg, BIRDNET_SR))
                    count += 1
                ch_seg_counts.append(count)

            if not audio_arrays:
                results[rec_id] = [0.0] * sources.shape[0]
                continue

            result = model.predict_arrays(
                audio_arrays,
                top_k=10,
                default_confidence_threshold=0.0,
                n_workers=1,
                batch_size=len(audio_arrays),
                device="CPU",
            )

            ch_scores = []
            offset = 0
            for ch_idx in range(sources.shape[0]):
                max_score = 0.0
                for seg_i in range(ch_seg_counts[ch_idx]):
                    inp_idx = offset + seg_i
                    n_segs = result.species_probs.shape[1]
                    for seg_idx in range(n_segs):
                        seg_ids = result.species_ids[inp_idx, seg_idx, :]
                        seg_probs = result.species_probs[inp_idx, seg_idx, :]
                        mask = seg_ids == target_idx
                        if mask.any():
                            prob = float(seg_probs[mask].max())
                            if prob > max_score:
                                max_score = prob
                ch_scores.append(max_score)
                offset += ch_seg_counts[ch_idx]

            results[rec_id] = ch_scores

        queue.put(results)
    except Exception:
        queue.put({rec_id: [0.0] * 4 for rec_id, _ in args_dict["rec_entries"]})


def _run_birdnet_species(
    species_code: str,
    valid_recs: list[tuple],
    taxonomy_maps,
) -> dict[str, list[float]]:
    """Run BirdNET on ALL recordings of a species in batched subprocess calls.

    Processes recordings in batches of BIRDNET_BATCH_SIZE to avoid FD leaks.
    Returns dict mapping rec_id -> list of 4 channel confidence scores.
    """
    sciname = taxonomy_maps.species_to_birdnet_sciname.get(species_code)
    if not sciname:
        return {row["recording_id"]: [0.0] * 4 for row, _ in valid_recs}

    all_results: dict[str, list[float]] = {}

    # Batch recordings to limit FD usage per subprocess
    for batch_start in range(0, len(valid_recs), BIRDNET_BATCH_SIZE):
        batch = valid_recs[batch_start: batch_start + BIRDNET_BATCH_SIZE]

        # Load sources for this batch
        rec_entries = []
        for row, src_paths in batch:
            rec_id = row["recording_id"]
            try:
                sources = np.stack(
                    [librosa.load(str(p), sr=MIXIT_SR, mono=True)[0] for p in src_paths]
                )
                rec_entries.append((rec_id, sources))
            except Exception:
                all_results[rec_id] = [0.0] * 4

        if not rec_entries:
            continue

        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_birdnet_batch_subprocess_target,
            args=({"target_sciname": sciname, "rec_entries": rec_entries}, q),
        )
        p.start()
        try:
            batch_results = q.get(timeout=300)  # 5min for batch
        except Exception:
            batch_results = {rec_id: [0.0] * 4 for rec_id, _ in rec_entries}
        p.join(timeout=10)
        if p.is_alive():
            p.kill()

        all_results.update(batch_results)

    return all_results


def _save_birdnet_scores(species_code: str, valid_recs: list[tuple],
                         scores: dict[str, list[float]]) -> Path:
    """Save BirdNET scores for a species to CSV."""
    BIRDNET_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row, _ in valid_recs:
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")
        sc = scores.get(rec_id, [0.0] * 4)
        rows.append({
            "recording_id": rec_id,
            "safe_id": safe_id,
            "ch0_conf": round(sc[0], 6),
            "ch1_conf": round(sc[1], 6),
            "ch2_conf": round(sc[2], 6),
            "ch3_conf": round(sc[3], 6),
        })
    out_path = BIRDNET_SCORES_DIR / f"{species_code}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _load_birdnet_scores(species_code: str) -> dict[str, list[float]] | None:
    """Load saved BirdNET scores from CSV. Returns None if file missing."""
    path = BIRDNET_SCORES_DIR / f"{species_code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    result = {}
    for _, r in df.iterrows():
        result[r["recording_id"]] = [
            r["ch0_conf"], r["ch1_conf"], r["ch2_conf"], r["ch3_conf"]
        ]
    return result


def run_birdnet_score(args):
    """Save BirdNET scores for all recordings to per-species CSV files.

    Output: birdnet_scores/{species}.csv with columns:
        recording_id, safe_id, ch0_conf, ch1_conf, ch2_conf, ch3_conf
    """
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    force = getattr(args, "force", False)

    # Load taxonomy maps
    try:
        from taxonomy_maps import build_taxonomy_maps
        taxonomy_maps = build_taxonomy_maps(species_codes)
    except Exception as e:
        _flush_print(f"ERROR: taxonomy_maps load failed ({e})")
        sys.exit(1)

    BIRDNET_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    _flush_print(f"Running BirdNET scoring for {len(species_codes)} species...")

    done = 0
    skipped = 0
    for species in species_codes:
        out_path = BIRDNET_SCORES_DIR / f"{species}.csv"
        if out_path.exists() and not force:
            _flush_print(f"  {species}: EXISTS (skip)")
            skipped += 1
            continue

        # Check BirdNET coverage
        coverage = taxonomy_maps.classifier_coverage.get(species, "none")
        if coverage not in ("both", "birdnet"):
            _flush_print(f"  {species}: no BirdNET coverage (skip)")
            skipped += 1
            continue

        df_sp = df[df["ebird_species_code"] == species]
        valid_recs = []
        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")
            src_dir = SOURCES_DIR / species
            src_paths = [src_dir / f"{safe_id}_src{i}.wav" for i in range(4)]
            if all(p.exists() for p in src_paths):
                valid_recs.append((row, src_paths))

        if not valid_recs:
            _flush_print(f"  {species}: no source files")
            continue

        t0 = time.time()
        scores = _run_birdnet_species(species, valid_recs, taxonomy_maps)
        _save_birdnet_scores(species, valid_recs, scores)
        elapsed = time.time() - t0

        n_above = sum(
            1 for sc in scores.values() if max(sc) >= BIRDNET_GATE_THRESHOLD
        )
        _flush_print(
            f"  {species}: {len(valid_recs)} recs, "
            f"{n_above} above gate ({elapsed:.1f}s)"
        )
        done += 1

    _flush_print(
        f"\nDone: {done} species scored, {skipped} skipped "
        f"-> {BIRDNET_SCORES_DIR}"
    )


# ---------------------------------------------------------------------------
# Species routing (B-1 vs B-2)
# ---------------------------------------------------------------------------


def _load_species_routing() -> pd.DataFrame | None:
    """Load the B-1/B-2 routing CSV. Returns None if it does not exist."""
    if not ROUTING_CSV.exists():
        return None
    return pd.read_csv(ROUTING_CSV)


def run_route_species(args):
    """Stage 4: decide B-1 (BirdNET-sufficient) vs B-2 (prototype required).

    B-1 criteria (all must hold):
      - species is registered in BirdNET (classifier_coverage in birdnet/both)
      - BirdNET hit rate >= --birdnet-hit-rate-min among valid recordings
      - absolute hit count >= --birdnet-hit-count-min

    Otherwise B-2. Writes species_routing.csv with full provenance so that
    later stages can inspect WHY a species ended up on a given route.
    """
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    hit_rate_min = getattr(args, "birdnet_hit_rate_min", DEFAULT_BIRDNET_HIT_RATE_MIN)
    hit_count_min = getattr(args, "birdnet_hit_count_min", DEFAULT_BIRDNET_HIT_COUNT_MIN)

    try:
        from taxonomy_maps import build_taxonomy_maps
        taxonomy_maps = build_taxonomy_maps(species_codes)
    except Exception as e:
        _flush_print(f"ERROR: taxonomy_maps load failed ({e})")
        sys.exit(1)

    rows = []
    for species in species_codes:
        df_sp = df[df["ebird_species_code"] == species]
        n_recs = len(df_sp)
        coverage = taxonomy_maps.classifier_coverage.get(species, "none")
        birdnet_registered = coverage in ("both", "birdnet")

        bn_scores = _load_birdnet_scores(species) if birdnet_registered else None
        if bn_scores:
            n_valid = len(bn_scores)
            n_above = sum(
                1 for sc in bn_scores.values()
                if max(sc) >= BIRDNET_GATE_THRESHOLD
            )
            hit_rate = n_above / n_valid if n_valid else 0.0
        else:
            n_valid = 0
            n_above = 0
            hit_rate = 0.0

        if not birdnet_registered:
            route, reason = "B2", "birdnet_unregistered"
        elif bn_scores is None:
            route, reason = "B2", "birdnet_not_scored"
        elif hit_rate < hit_rate_min:
            route, reason = "B2", "birdnet_low_hit_rate"
        elif n_above < hit_count_min:
            route, reason = "B2", "birdnet_low_hit_count"
        else:
            route, reason = "B1", "birdnet_sufficient"

        rows.append({
            "species_code": species,
            "route": route,
            "reason": reason,
            "birdnet_coverage": coverage,
            "n_recordings": n_recs,
            "n_valid_scored": n_valid,
            "n_above_gate": n_above,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_threshold": hit_rate_min,
            "hit_count_threshold": hit_count_min,
        })

    out_df = pd.DataFrame(rows)
    ROUTING_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(ROUTING_CSV, index=False)

    n_b1 = int((out_df["route"] == "B1").sum())
    n_b2 = int((out_df["route"] == "B2").sum())
    _flush_print(
        f"Routing: {n_b1} B-1 (BirdNET) + {n_b2} B-2 (prototype) "
        f"-> {ROUTING_CSV}"
    )
    if n_b2:
        reason_counts = (
            out_df[out_df["route"] == "B2"]["reason"].value_counts().to_dict()
        )
        reason_str = ", ".join(f"{k}={v}" for k, v in reason_counts.items())
        _flush_print(f"  B-2 reasons: {reason_str}")


def run_channel_select(args):
    """Stage 7: per-recording focal-channel selection (gate + rank).

    Requires species_routing.csv. For each recording:
      - B-1 route: gate by BirdNET conf >= 0.2, rank by conf * bioacoustic_quality
      - B-2 route: gate by max prototype cosine similarity >= 0.3, rank by
        similarity * bioacoustic_quality
    Quality comes from the acoustic_features/ npz (median SNR/NDSI).

    Output: channel_selection/{species}.csv with columns:
        recording_id, safe_id, focal_channel, species_score, method
    """
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    force = getattr(args, "force", False)

    routing = _load_species_routing()
    if routing is None:
        _flush_print(
            "ERROR: species_routing.csv not found. Run 'route-species' first."
        )
        sys.exit(1)
    route_map = dict(zip(routing["species_code"], routing["route"]))

    # Prototypes (only needed for B-2 species but cheap to load all at once)
    prototypes = _load_species_prototypes(species_codes)

    CHANNEL_SELECT_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_SIM_THRESHOLD = 0.3

    _flush_print(f"Running channel selection for {len(species_codes)} species...")

    done = 0
    skipped = 0
    for species in species_codes:
        out_path = CHANNEL_SELECT_DIR / f"{species}.csv"
        if out_path.exists() and not force:
            _flush_print(f"  {species}: EXISTS (skip)")
            skipped += 1
            continue

        route = route_map.get(species)
        if route is None:
            _flush_print(f"  {species}: not in routing CSV (skip)")
            skipped += 1
            continue

        df_sp = df[df["ebird_species_code"] == species]
        valid_recs = []
        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")
            src_dir = SOURCES_DIR / species
            src_paths = [src_dir / f"{safe_id}_src{i}.wav" for i in range(4)]
            if all(p.exists() for p in src_paths):
                valid_recs.append((row, src_paths))

        if not valid_recs:
            _flush_print(f"  {species}: no source files")
            continue

        birdnet_scores: dict[str, list[float]] | None = None
        sp_protos: np.ndarray | None = None

        if route == "B1":
            birdnet_scores = _load_birdnet_scores(species)
            if birdnet_scores is None:
                _flush_print(
                    f"  {species}: route=B1 but BirdNET scores missing (skip)"
                )
                skipped += 1
                continue
        else:  # B-2
            sp_protos = prototypes.get(species)
            if sp_protos is None:
                _flush_print(
                    f"  {species}: route=B2 but prototype missing (skip)"
                )
                skipped += 1
                continue

        rows = []
        for row, src_paths in valid_recs:
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")

            try:
                acoustic_per_ch = [
                    _load_acoustic_features(species, rec_id, ch)
                    for ch in range(4)
                ]
                ch_qualities = [
                    _channel_quality_from_features(a) for a in acoustic_per_ch
                ]

                # Gate: determine passing channels
                passing_channels: list[tuple[int, float]] = []

                if route == "B1":
                    bn_scores = birdnet_scores.get(rec_id, [0.0] * 4)
                    for ch in range(4):
                        if bn_scores[ch] >= BIRDNET_GATE_THRESHOLD:
                            passing_channels.append((ch, bn_scores[ch]))
                else:  # B-2
                    for ch in range(4):
                        perch = _load_perch_embeddings_features(
                            species, rec_id, ch
                        )
                        if perch is None or len(perch["embeddings"]) == 0:
                            continue
                        sims = perch["embeddings"] @ sp_protos.T
                        max_sim = float(np.max(sims))
                        if max_sim >= PROTO_SIM_THRESHOLD:
                            passing_channels.append((ch, max_sim))

                if not passing_channels:
                    continue

                # Rank: gate_score * bioacoustic_quality
                best_ch = -1
                best_rank = -1.0
                best_gate_score = 0.0
                for ch, gate_score in passing_channels:
                    q = ch_qualities[ch]
                    rank = gate_score * max(q, 0.01)
                    if rank > best_rank:
                        best_rank = rank
                        best_ch = ch
                        best_gate_score = gate_score

                rows.append({
                    "recording_id": rec_id,
                    "safe_id": safe_id,
                    "focal_channel": best_ch,
                    "species_score": round(best_gate_score, 4),
                    "method": route,
                })

            except Exception as e:
                _flush_print(f"    WARN: {species}/{safe_id}: {e}")
                continue

        if not rows:
            _flush_print(f"  {species}: [{route}] no channels selected")
            continue

        ch_df = pd.DataFrame(rows)
        ch_df.to_csv(out_path, index=False)
        ch_dist = ch_df["focal_channel"].value_counts().to_dict()
        ch_str = "/".join(f"ch{k}:{v}" for k, v in sorted(ch_dist.items()))
        _flush_print(
            f"  {species}: [{route}] {ch_str}, "
            f"{len(rows)} recordings selected"
        )
        done += 1

    _flush_print(
        f"\nDone: {done} species, {skipped} skipped "
        f"-> {CHANNEL_SELECT_DIR}"
    )


def _load_channel_selection(species_code: str) -> pd.DataFrame | None:
    """Load saved channel selection CSV. Returns None if missing."""
    path = CHANNEL_SELECT_DIR / f"{species_code}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def run_segment(args):
    """Run TweetyNet segmentation using saved channel selection results.

    Output: tweetynet_segments/{species}.csv with columns:
        recording_id, safe_id, focal_channel, bout_idx, bout_onset, bout_offset,
        bout_duration, n_notes, total_note_dur, silence_ratio, notes_json
    """
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    force = getattr(args, "force", False)

    # Load TweetyNet model
    device_name = getattr(args, "device", "cpu")
    device = torch.device(
        device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu"
    )
    _flush_print(f"Loading TweetyNet from {TWEETYNET_MODEL_PATH.name} (device={device})...")
    tweetynet = _load_tweetynet(device)

    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    _flush_print(f"Running TweetyNet segmentation for {len(species_codes)} species...")

    done = 0
    skipped = 0
    for species in species_codes:
        out_path = SEGMENTS_DIR / f"{species}.csv"
        if out_path.exists() and not force:
            _flush_print(f"  {species}: EXISTS (skip)")
            skipped += 1
            continue

        # Load channel selection
        ch_sel = _load_channel_selection(species)
        if ch_sel is None:
            _flush_print(f"  {species}: no channel selection (skip)")
            skipped += 1
            continue

        rows = []
        for _, sel_row in ch_sel.iterrows():
            rec_id = sel_row["recording_id"]
            safe_id = sel_row["safe_id"]
            focal_ch = int(sel_row["focal_channel"])
            src_path = SOURCES_DIR / species / f"{safe_id}_src{focal_ch}.wav"

            if not src_path.exists():
                continue

            try:
                sources_focal = librosa.load(str(src_path), sr=MIXIT_SR, mono=True)[0]
                probs = _tweetynet_predict(tweetynet, sources_focal, device)
                segments = _probs_to_segments(probs)
                bouts = _group_notes_into_bouts(segments)

                for bout_idx, bout in enumerate(bouts):
                    bout_dur = bout["bout_offset"] - bout["bout_onset"]
                    rows.append({
                        "recording_id": rec_id,
                        "safe_id": safe_id,
                        "focal_channel": focal_ch,
                        "bout_idx": bout_idx,
                        "bout_onset": bout["bout_onset"],
                        "bout_offset": bout["bout_offset"],
                        "bout_duration": round(bout_dur, 4),
                        "n_notes": bout["n_notes"],
                        "total_note_dur": bout["total_note_duration"],
                        "silence_ratio": bout["silence_ratio"],
                        "notes_json": json.dumps(bout["notes"]),
                    })

            except Exception as e:
                _flush_print(f"    WARN: {species}/{safe_id}: {e}")
                continue

        if not rows:
            _flush_print(f"  {species}: no bouts found")
            continue

        seg_df = pd.DataFrame(rows)
        seg_df.to_csv(out_path, index=False)
        _flush_print(
            f"  {species}: {len(ch_sel)} recordings -> {len(rows)} bouts"
        )
        done += 1

        import gc
        gc.collect()

    _flush_print(
        f"\nDone: {done} species segmented, {skipped} skipped "
        f"-> {SEGMENTS_DIR}"
    )


def _load_segments(species_code: str) -> pd.DataFrame | None:
    """Load saved TweetyNet segment CSV. Returns None if missing."""
    path = SEGMENTS_DIR / f"{species_code}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Energy-based VAD
# ---------------------------------------------------------------------------


def extract_segments_vad(audio, sr=MIXIT_SR, min_dur=0.05, merge_gap=0.2):
    """Energy-based VAD. Returns list of (onset, offset) tuples in seconds.

    1. RMS envelope (25ms frame, 10ms hop)
    2. Threshold: > median + 2*MAD
    3. Binary closing (merge gaps < merge_gap)
    4. Binary opening (remove < min_dur)
    5. Find contiguous regions
    """
    frame_len = int(0.025 * sr)  # 25ms
    hop = int(0.010 * sr)  # 10ms
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)

    if n_frames <= 1:
        return []

    frame_rms = np.array(
        [
            np.sqrt(np.mean(audio[j * hop : j * hop + frame_len] ** 2) + 1e-10)
            for j in range(n_frames)
        ]
    )

    # Threshold: median + 2*MAD
    median_rms = np.median(frame_rms)
    mad = np.median(np.abs(frame_rms - median_rms))
    threshold = median_rms + 2.0 * mad

    binary = frame_rms > threshold

    # Binary closing: merge gaps shorter than merge_gap
    frame_dur = hop / sr
    gap_frames = int(merge_gap / frame_dur)
    if gap_frames > 1:
        binary = binary_closing(binary, structure=np.ones(gap_frames))

    # Binary opening: remove segments shorter than min_dur
    min_frames = int(min_dur / frame_dur)
    if min_frames > 1:
        binary = binary_opening(binary, structure=np.ones(min_frames))

    # Find contiguous regions
    lab_arr, n = label(binary)
    segments = []
    for k in range(1, n + 1):
        idx = np.where(lab_arr == k)[0]
        onset = idx[0] * frame_dur
        offset = (idx[-1] + 1) * frame_dur
        if offset - onset >= min_dur:
            segments.append((round(onset, 4), round(offset, 4)))

    return segments


# ===========================================================================
# Separate command (multiprocessing)
# ===========================================================================

_worker_state = None


def _init_worker():
    """Pool initializer: create TF session for this worker (CPU only)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    global _worker_state
    _worker_state = _create_tf_session()


def _separate_worker(task):
    """Process one recording. Uses global _worker_state.

    Args:
        task: dict with keys species, recording_id, audio_path, out_dir

    Returns:
        dict with species, recording_id, status, elapsed
    """
    global _worker_state
    if _worker_state is None:
        _worker_state = _create_tf_session()

    sess, input_t, output_t = _worker_state
    species = task["species"]
    rec_id = task["recording_id"]
    audio_path = task["audio_path"]
    out_dir = Path(task["out_dir"])
    safe_id = rec_id.replace(":", "_")

    t0 = time.time()
    try:
        # Load and resample to 22050 Hz, capped to prevent OOM on long clips.
        y, _ = librosa.load(
            str(audio_path), sr=MIXIT_SR, mono=True, duration=MAX_SEPARATE_DURATION_S
        )
        if len(y) == 0:
            return {"species": species, "recording_id": rec_id, "status": "empty", "elapsed": 0.0}

        # Run separation
        sources = separate_audio(sess, input_t, output_t, y)

        # Save 4 source WAVs
        out_dir.mkdir(parents=True, exist_ok=True)
        for ch_idx in range(sources.shape[0]):
            out_path = out_dir / f"{safe_id}_src{ch_idx}.wav"
            sf.write(str(out_path), sources[ch_idx], MIXIT_SR)

        elapsed = time.time() - t0
        return {"species": species, "recording_id": rec_id, "status": "ok", "elapsed": elapsed}

    except Exception as e:
        elapsed = time.time() - t0
        return {"species": species, "recording_id": rec_id, "status": f"error: {e}", "elapsed": elapsed}


def _sources_exist(species, rec_id):
    """Check if all 4 source WAVs already exist for a recording."""
    safe_id = rec_id.replace(":", "_")
    out_dir = SOURCES_DIR / species
    return all((out_dir / f"{safe_id}_src{i}.wav").exists() for i in range(4))


def run_separate(args):
    """Run Bird-MixIT separation on all recordings."""
    df = load_metadata(args.species)

    # Build task list
    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name

        if not audio_path.exists():
            continue

        # Resume: skip if all 4 sources exist
        if _sources_exist(species, rec_id):
            skipped += 1
            continue

        tasks.append(
            {
                "species": species,
                "recording_id": rec_id,
                "audio_path": str(audio_path),
                "out_dir": str(SOURCES_DIR / species),
            }
        )

    if args.limit:
        tasks = tasks[: args.limit]

    total = len(tasks)
    if total == 0:
        _flush_print(f"No recordings to separate (skipped {skipped} already done).")
        return

    n_workers = args.workers if args.workers else min(os.cpu_count() // 2, 8)
    n_workers = max(1, n_workers)

    _flush_print(
        f"Separating {total} recordings with {n_workers} workers "
        f"(skipped {skipped} already done)..."
    )

    t_start = time.time()
    ctx = multiprocessing.get_context("spawn")
    done = 0
    errors = 0

    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_separate_worker, tasks):
            done += 1
            status = result["status"]
            elapsed = result["elapsed"]
            rec_label = f"{result['species']}/{result['recording_id'].replace(':', '_')}"

            if status == "ok":
                _flush_print(
                    f"  [{done}/{total}] {rec_label} -> 4 sources ({elapsed:.1f}s)"
                )
            else:
                errors += 1
                _flush_print(f"  [{done}/{total}] {rec_label} -> {status}")

    total_time = time.time() - t_start
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    _flush_print(
        f"Done: {total} recordings separated in {mins}m {secs}s"
        + (f" ({errors} errors)" if errors else "")
    )


# ===========================================================================
# Select command (single process)
# ===========================================================================


def run_select(args):
    """Stage 9 (+ orchestration): rank bouts and export WAVs.

    Calls ensure-helpers for every upstream stage so the pipeline is
    self-healing: missing intermediates are computed automatically, and
    existing outputs are reused unless --force is given.

    Upstream order (matches numbered stages in the module docstring):
      3. birdnet-score
      4. route-species
      5. compute-perch-embeddings (B-2 species only)
      6. build-prototypes         (B-2 species only)
      7. channel-select
      8. segment
      9. (this function) rank + export
    """
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    target_n = args.target_n
    max_per_rec = args.max_per_recording
    force = getattr(args, "force", False)

    _flush_print("=== Stage: acoustic features (SNR/NDSI/bird_ratio) ===")
    _ensure_acoustic_features(args, force)

    _flush_print("=== Stage: BirdNET scores ===")
    _ensure_birdnet_scores(args, force)

    _flush_print("=== Stage: Species routing ===")
    _ensure_routing(args, force)

    _flush_print("=== Stage: Perch embeddings (B-2 species) ===")
    _ensure_perch_embeddings(args, force)

    _flush_print("=== Stage: Prototypes (B-2 species) ===")
    _ensure_prototypes(args, force)

    _flush_print("=== Stage: Channel selection ===")
    _ensure_channel_selection(args, force)

    _flush_print("=== Stage: TweetyNet segmentation ===")
    _ensure_segments(args, force)

    _flush_print("=== Stage: Rank bouts and export ===")

    # Resume support: load existing results and skip completed species
    all_selected = []
    completed_species = set()
    if RESULTS_CSV.exists() and not force:
        existing = pd.read_csv(RESULTS_CSV)
        completed_species = set(existing["species_code"].unique())
        if completed_species:
            for sp in completed_species:
                all_selected.append(existing[existing["species_code"] == sp])
            _flush_print(f"Resuming: {len(completed_species)} species already done, "
                         f"{len(species_codes) - len(completed_species)} remaining")

    for species in species_codes:
        if species in completed_species:
            continue
        df_sp = df[df["ebird_species_code"] == species]

        # Load intermediate results
        ch_sel = _load_channel_selection(species)
        seg_df = _load_segments(species)

        if ch_sel is None or ch_sel.empty:
            _flush_print(f"  {species}: no channel selection")
            continue
        if seg_df is None or seg_df.empty:
            _flush_print(f"  {species}: no segments")
            continue

        # Merge channel selection with segments
        ch_sel_map = {}
        for _, r in ch_sel.iterrows():
            ch_sel_map[r["recording_id"]] = {
                "focal_channel": int(r["focal_channel"]),
                "species_score": float(r["species_score"]),
                "method": r["method"],
            }

        # Build sciname map from metadata
        sciname_map = {}
        for _, row in df_sp.iterrows():
            sciname_map[row["recording_id"]] = row["scientific_name"]

        # Pre-load BirdNET scores for this species (once, not per-bout)
        has_b1 = any(v["method"] == "B1" for v in ch_sel_map.values())
        bn_scores_cache = _load_birdnet_scores(species) if has_b1 else None

        species_bouts = []

        for _, seg_row in seg_df.iterrows():
            rec_id = seg_row["recording_id"]
            safe_id = seg_row["safe_id"]
            focal_ch = int(seg_row["focal_channel"])
            bout_idx = int(seg_row["bout_idx"])

            ch_info = ch_sel_map.get(rec_id)
            if ch_info is None:
                continue

            species_score = ch_info["species_score"]
            mode_tag = ch_info["method"]

            # Compute bout quality from audio
            src_path = SOURCES_DIR / species / f"{safe_id}_src{focal_ch}.wav"
            if not src_path.exists():
                continue

            try:
                notes = json.loads(seg_row["notes_json"])
                sources_focal = librosa.load(str(src_path), sr=MIXIT_SR, mono=True)[0]

                note_audio_parts = []
                for note_on, note_off in notes:
                    s0 = int(note_on * MIXIT_SR)
                    s1 = int(note_off * MIXIT_SR)
                    part = sources_focal[s0:s1]
                    if len(part) > 0:
                        note_audio_parts.append(part)

                if not note_audio_parts:
                    continue

                note_audio = np.concatenate(note_audio_parts)
                bout_snr, bout_ndsi = _compute_snr_ndsi(note_audio, MIXIT_SR)
                bout_quality = _bioacoustic_quality(bout_snr, bout_ndsi)
                rank_score = species_score * max(bout_quality, 0.01)

                # BirdNET confidence for this recording
                birdnet_conf = 0.0
                if mode_tag == "B1" and bn_scores_cache and rec_id in bn_scores_cache:
                    birdnet_conf = max(bn_scores_cache[rec_id])

                species_bouts.append({
                    "species_code": species,
                    "scientific_name": sciname_map.get(rec_id, ""),
                    "recording_id": rec_id,
                    "safe_id": safe_id,
                    "focal_channel": focal_ch,
                    "channel_scores": str(bn_scores_cache.get(rec_id, [0.0]*4)) if bn_scores_cache else "[]",
                    "species_score": round(species_score, 4),
                    "birdnet_conf": round(birdnet_conf, 4),
                    "bout_snr": round(bout_snr, 2),
                    "bout_ndsi": round(bout_ndsi, 4),
                    "bout_quality": round(bout_quality, 4),
                    "rank_score": round(rank_score, 6),
                    "bout_idx": bout_idx,
                    "bout_onset": seg_row["bout_onset"],
                    "bout_offset": seg_row["bout_offset"],
                    "bout_duration": seg_row["bout_duration"],
                    "n_notes": seg_row["n_notes"],
                    "total_note_dur": seg_row["total_note_dur"],
                    "silence_ratio": seg_row["silence_ratio"],
                    "notes_json": seg_row["notes_json"],
                })

            except Exception as e:
                _flush_print(f"    WARN: {species}/{safe_id}: {e}")
                continue

        if not species_bouts:
            _flush_print(f"  {species}: no bouts found")
            continue

        bout_df = pd.DataFrame(species_bouts)
        bout_df = bout_df.sort_values("rank_score", ascending=False)
        selected = []
        rec_counts: dict[str, int] = {}
        for _, r in bout_df.iterrows():
            rid = r["recording_id"]
            rec_counts.setdefault(rid, 0)
            if rec_counts[rid] < max_per_rec:
                selected.append(r)
                rec_counts[rid] += 1
            if len(selected) >= target_n:
                break

        selected_df = pd.DataFrame(selected)
        selected_df["method"] = "bird-mixit+tweetynet"

        # Export selected bouts as WAVs (bout_onset to bout_offset)
        # Clean old WAVs first to avoid stale files from prior runs
        out_dir = SELECTED_DIR / species
        if out_dir.exists():
            for old_wav in out_dir.glob("*.wav"):
                old_wav.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        exported = 0

        for _, bout_row in selected_df.iterrows():
            safe_id = bout_row["safe_id"]
            focal_ch = bout_row["focal_channel"]
            src_path = SOURCES_DIR / species / f"{safe_id}_src{focal_ch}.wav"

            if not src_path.exists():
                continue

            try:
                y, _ = librosa.load(str(src_path), sr=MIXIT_SR, mono=True)
                start_s = int(bout_row["bout_onset"] * MIXIT_SR)
                end_s = int(bout_row["bout_offset"] * MIXIT_SR)
                y_bout = y[start_s:end_s]

                if len(y_bout) == 0:
                    continue

                bout_idx = int(bout_row["bout_idx"])
                out_path = out_dir / f"{safe_id}_b{bout_idx:03d}.wav"
                sf.write(str(out_path), y_bout, MIXIT_SR)
                exported += 1
            except Exception:
                pass

        all_selected.append(selected_df)
        mode_tag = ch_sel["method"].iloc[0] if len(ch_sel) > 0 else "?"
        ch_dist = bout_df["focal_channel"].value_counts().to_dict()
        ch_str = "/".join(f"ch{k}:{v}" for k, v in sorted(ch_dist.items()))
        avg_sp_score = bout_df["species_score"].mean()
        avg_bq = bout_df["bout_quality"].mean()
        _flush_print(
            f"  {species}: [{mode_tag}] {ch_str}, "
            f"sp_score={avg_sp_score:.3f}, bq={avg_bq:.3f}, "
            f"{len(bout_df)} bouts -> {len(selected_df)} selected"
        )

        import gc
        gc.collect()

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(RESULTS_CSV, index=False)
        total_bouts = len(result)
        _flush_print(f"\nDone: {total_bouts} bouts -> {RESULTS_CSV.name}")
    else:
        _flush_print("No bouts selected.")


def _ensure_acoustic_features(args, force):
    """Ensure SNR/NDSI/bird_ratio npz exist for every (species, rec, ch)."""
    df = load_metadata(args.species)
    missing = 0
    for _, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")
        for ch in range(4):
            src = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
            out = ACOUSTIC_FEATURES_DIR / species / f"{safe_id}_src{ch}.npz"
            if src.exists() and (not out.exists() or force):
                missing += 1
                break
        if missing:
            break

    if not missing:
        _flush_print("  acoustic features up to date (cached)")
        return

    run_compute_acoustic_features(argparse.Namespace(
        species=args.species,
        force=force,
        device=getattr(args, "device", "cuda"),
    ))


def _ensure_birdnet_scores(args, force):
    """Ensure BirdNET scores exist for every BirdNET-registered species."""
    run_birdnet_score(argparse.Namespace(
        species=args.species,
        force=force,
    ))


def _ensure_routing(args, force):
    """Ensure species_routing.csv exists."""
    if ROUTING_CSV.exists() and not force:
        _flush_print(f"  routing cached: {ROUTING_CSV.name}")
        return
    run_route_species(argparse.Namespace(
        species=args.species,
        birdnet_hit_rate_min=getattr(
            args, "birdnet_hit_rate_min", DEFAULT_BIRDNET_HIT_RATE_MIN
        ),
        birdnet_hit_count_min=getattr(
            args, "birdnet_hit_count_min", DEFAULT_BIRDNET_HIT_COUNT_MIN
        ),
    ))


def _ensure_perch_embeddings(args, force):
    """Ensure Perch embeddings exist for all B-2 source files."""
    routing = _load_species_routing()
    if routing is None:
        _flush_print("  routing missing — skipping Perch embeddings")
        return
    b2_species = routing[routing["route"] == "B2"]["species_code"].tolist()
    if not b2_species:
        _flush_print("  no B-2 species — skipping Perch embeddings")
        return
    run_compute_perch_embeddings(argparse.Namespace(
        species=args.species,
        force=force,
        device=getattr(args, "device", "gpu"),
        all_species=False,
    ))


def _ensure_prototypes(args, force):
    """Ensure prototypes exist for B-2 species."""
    routing = _load_species_routing()
    if routing is None:
        _flush_print("  routing missing — skipping prototypes")
        return
    b2_species = routing[routing["route"] == "B2"]["species_code"].tolist()
    if not b2_species:
        _flush_print("  no B-2 species — skipping prototypes")
        return
    run_build_prototypes(argparse.Namespace(
        species=args.species,
        force=force,
        all_species=False,
    ))


def _ensure_channel_selection(args, force):
    """Ensure channel selection exists for all species."""
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    missing = [
        sp for sp in species_codes
        if force or not (CHANNEL_SELECT_DIR / f"{sp}.csv").exists()
    ]
    if not missing:
        _flush_print(f"  all {len(species_codes)} species have channel selection (cached)")
        return
    run_channel_select(argparse.Namespace(
        species=args.species,
        force=force,
    ))


def _ensure_segments(args, force):
    """Ensure TweetyNet segments exist for all species."""
    df = load_metadata(args.species)
    species_codes = sorted(df["ebird_species_code"].unique())
    missing = [
        sp for sp in species_codes
        if force or not (SEGMENTS_DIR / f"{sp}.csv").exists()
    ]
    if not missing:
        _flush_print(f"  all {len(species_codes)} species have segments (cached)")
        return
    run_segment(argparse.Namespace(
        species=args.species,
        force=force,
        device=getattr(args, "device", "cuda"),
    ))


# ===========================================================================
# CLI
# ===========================================================================


def _add_common_species_arg(p):
    p.add_argument(
        "--species", type=str, default=None, help="Filter by species code"
    )


def _add_force_arg(p):
    p.add_argument(
        "--force", action="store_true", help="Re-run even if output exists"
    )


def _add_routing_threshold_args(p):
    p.add_argument(
        "--birdnet-hit-rate-min",
        type=float,
        default=DEFAULT_BIRDNET_HIT_RATE_MIN,
        help=(
            "Min fraction of recordings with BirdNET conf >= gate to route B-1 "
            f"(default: {DEFAULT_BIRDNET_HIT_RATE_MIN})"
        ),
    )
    p.add_argument(
        "--birdnet-hit-count-min",
        type=int,
        default=DEFAULT_BIRDNET_HIT_COUNT_MIN,
        help=(
            "Min absolute count of recordings above gate to route B-1 "
            f"(default: {DEFAULT_BIRDNET_HIT_COUNT_MIN})"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bird-MixIT source separation + segment extraction pipeline (v3)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. separate
    sep = sub.add_parser("separate", help="Run Bird-MixIT 4-source separation")
    _add_common_species_arg(sep)
    sep.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    sep.add_argument("--limit", type=int, default=None, help="Max recordings to process")

    # 2. compute-acoustic-features
    caf = sub.add_parser(
        "compute-acoustic-features",
        help="Compute SNR/NDSI/bird_ratio per 5s window (librosa + TweetyNet)",
    )
    _add_common_species_arg(caf)
    _add_force_arg(caf)
    caf.add_argument("--device", type=str, default="cuda", help="TweetyNet device (cuda/cpu)")

    # 3. birdnet-score
    bns = sub.add_parser("birdnet-score", help="BirdNET 4ch scores for registered species")
    _add_common_species_arg(bns)
    _add_force_arg(bns)

    # 4. route-species
    rs = sub.add_parser(
        "route-species",
        help="Decide B-1 (BirdNET) vs B-2 (prototype) per species",
    )
    _add_common_species_arg(rs)
    _add_routing_threshold_args(rs)

    # 5. compute-perch-embeddings
    cpe = sub.add_parser(
        "compute-perch-embeddings",
        help="Compute Perch v2 embeddings for B-2 species (or all with flag)",
    )
    _add_common_species_arg(cpe)
    _add_force_arg(cpe)
    cpe.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="TF device")
    cpe.add_argument(
        "--all-species", action="store_true",
        help="Ignore routing and embed every species",
    )

    # 6. build-prototypes
    bp = sub.add_parser("build-prototypes", help="HDBSCAN prototypes for B-2 species")
    _add_common_species_arg(bp)
    _add_force_arg(bp)
    bp.add_argument(
        "--all-species", action="store_true",
        help="Ignore routing and build prototypes for every species",
    )

    # 7. channel-select
    chs = sub.add_parser("channel-select", help="Per-recording focal-channel gate+rank")
    _add_common_species_arg(chs)
    _add_force_arg(chs)

    # 8. segment
    seg_cmd = sub.add_parser("segment", help="TweetyNet segmentation of focal channels")
    _add_common_species_arg(seg_cmd)
    _add_force_arg(seg_cmd)
    seg_cmd.add_argument("--device", type=str, default="cuda", help="TweetyNet device")

    # 9. select (orchestration + ranking + export)
    sel = sub.add_parser(
        "select",
        help="Run stages 3-9 with caching, rank bouts, export WAVs",
    )
    _add_common_species_arg(sel)
    sel.add_argument("--target-n", type=int, default=75, help="Target segments per species")
    sel.add_argument("--max-per-recording", type=int, default=5, help="Max segments per recording")
    sel.add_argument("--device", type=str, default="cuda", help="TweetyNet device")
    _add_routing_threshold_args(sel)
    _add_force_arg(sel)

    # all: stages 1 through 9
    a = sub.add_parser("all", help="Full pipeline: separate -> acoustic -> ... -> select")
    _add_common_species_arg(a)
    a.add_argument("--workers", type=int, default=None, help="Workers for Bird-MixIT")
    a.add_argument("--limit", type=int, default=None, help="Max recordings to separate")
    a.add_argument("--target-n", type=int, default=75, help="Target segments per species")
    a.add_argument("--max-per-recording", type=int, default=5, help="Max segments per recording")
    a.add_argument("--device", type=str, default="cuda", help="TweetyNet device")
    _add_routing_threshold_args(a)
    _add_force_arg(a)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "separate":
        run_separate(args)

    elif args.command == "compute-acoustic-features":
        run_compute_acoustic_features(args)

    elif args.command == "birdnet-score":
        run_birdnet_score(args)

    elif args.command == "route-species":
        run_route_species(args)

    elif args.command == "compute-perch-embeddings":
        run_compute_perch_embeddings(args)

    elif args.command == "build-prototypes":
        run_build_prototypes(args)

    elif args.command == "channel-select":
        run_channel_select(args)

    elif args.command == "segment":
        run_segment(args)

    elif args.command == "select":
        run_select(args)

    elif args.command == "all":
        run_separate(args)
        run_compute_acoustic_features(args)
        run_birdnet_score(args)
        run_route_species(args)
        run_compute_perch_embeddings(
            argparse.Namespace(
                species=args.species,
                force=args.force,
                device="gpu",
                all_species=False,
            )
        )
        run_build_prototypes(
            argparse.Namespace(
                species=args.species,
                force=args.force,
                all_species=False,
            )
        )
        run_select(args)


if __name__ == "__main__":
    main()
