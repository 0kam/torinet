"""
鳥類発声セグメンテーション手法のプロトタイプ比較。

7つのセグメンテーション手法をテスト用音声ファイルに適用し、
結果を可視化・比較する。

手法:
  1. Band Energy Hysteresis VAD
  2. PCEN Mask Connected Components
  3. Flux-Anchor Boundary Estimation
  6. REPET-lite Foreground Extraction
  8. 2-State Gaussian HMM (Viterbi)
  12. Spectral Entropy Change Point
  14. Dual-Axis Median Clipping (Lasseck)

使い方:
  python prototype_segmentation.py                    # 全ファイル処理
  python prototype_segmentation.py --species brebul1  # 1種のみ
  python prototype_segmentation.py --limit 10         # 先頭10ファイルのみ
  python prototype_segmentation.py --methods 1,6,14   # 指定手法のみ
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    gaussian_filter,
    gaussian_filter1d,
    grey_opening,
    label,
    median_filter,
    percentile_filter,
    uniform_filter,
)
from scipy.signal import find_peaks, medfilt, savgol_filter

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
RESULTS_DIR = NAS_BASE / "segments" / "test_samples_results_v2"
RESULTS_CSV = STEP_DIR / "prototype_results_v2.csv"

METHOD_NAMES = {
    1: "Band Energy Hysteresis",
    2: "PCEN Connected Components",
    3: "Flux-Anchor Boundary",
    6: "REPET-lite Foreground",
    8: "2-State HMM (Viterbi)",
    12: "Spectral Entropy",
    14: "Median Clipping (Lasseck)",
}

METHOD_COLORS = {
    1: "red",
    2: "blue",
    3: "green",
    6: "cyan",
    8: "magenta",
    12: "gold",
    14: "darkorange",
}


# ---------------------------------------------------------------------------
# Post-processing (common)
# ---------------------------------------------------------------------------

def postprocess_segments(
    segments: List[Tuple[float, float]],
    merge_gap: float = 0.05,
    min_duration: float = 0.02,
    max_duration: float = 30.0,
) -> List[Tuple[float, float]]:
    """Merge nearby segments, remove too-short and too-long ones."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s[0])

    # Merge close segments
    merged: List[Tuple[float, float]] = [segments[0]]
    for onset, offset in segments[1:]:
        if onset - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
        else:
            merged.append((onset, offset))

    # Filter by duration
    result = []
    for onset, offset in merged:
        dur = offset - onset
        if min_duration <= dur <= max_duration:
            result.append((onset, offset))
    return result


def _labels_to_segments(active: np.ndarray, hop: int, sr: int) -> List[Tuple[float, float]]:
    """Convert boolean activation array to (onset, offset) list in seconds."""
    lab_arr, n = label(active)
    segments = []
    for k in range(1, n + 1):
        idx = np.where(lab_arr == k)[0]
        segments.append((idx[0] * hop / sr, (idx[-1] + 1) * hop / sr))
    return segments


# ---------------------------------------------------------------------------
# Method 1: Band Energy Hysteresis VAD
# ---------------------------------------------------------------------------

def method_band_energy_hysteresis(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via band-limited energy with hysteresis thresholding."""
    hop, n_fft = 256, 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freq >= 300) & (freq <= 10000)
    e = np.log1p(S[band].sum(axis=0))
    e = median_filter(e, size=5)
    noise = percentile_filter(e, percentile=20, size=201)
    hi = noise + 1.2
    lo = noise + 0.6

    # Hysteresis loop
    active = np.zeros_like(e, dtype=bool)
    on = False
    for i, v in enumerate(e):
        if not on and v > hi[i]:
            on = True
        elif on and v < lo[i]:
            on = False
        active[i] = on

    return _labels_to_segments(active, hop, sr)


# ---------------------------------------------------------------------------
# Method 2: PCEN Mask Connected Components
# ---------------------------------------------------------------------------

def method_pcen_connected_components(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via PCEN-normalised mel spectrogram + connected components."""
    hop = 128
    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmin=150, fmax=12000, hop_length=hop, power=1.0,
    )
    P = librosa.pcen(M * (2**16), sr=sr, hop_length=hop)
    Z = np.log1p(P)
    thr = np.percentile(Z, 75, axis=1, keepdims=True) + 0.3
    mask = Z > thr
    mask = binary_opening(mask, structure=np.ones((2, 2)))
    mask = binary_closing(mask, structure=np.ones((3, 3)))
    lab_arr, n = label(mask)

    segments = []
    for k in range(1, n + 1):
        coords = np.where(lab_arr == k)
        t_min = coords[1].min()
        t_max = coords[1].max()
        onset = t_min * hop / sr
        offset = (t_max + 1) * hop / sr
        segments.append((onset, offset))

    return postprocess_segments(segments, merge_gap=0.0, min_duration=0.0, max_duration=1e6)


# ---------------------------------------------------------------------------
# Method 3: Flux-Anchor Boundary Estimation
# ---------------------------------------------------------------------------

def method_flux_anchor(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via spectral flux onset detection + energy valley search."""
    hop, n_fft = 128, 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-9)
    flux = np.r_[
        0.0,
        np.sqrt(np.sum(np.maximum(0, np.diff(Sn, axis=1)) ** 2, axis=0)),
    ]
    flux = savgol_filter(flux, 11, 2)
    base = percentile_filter(flux, percentile=50, size=201)
    mad = median_filter(np.abs(flux - base), size=101) + 1e-6
    peaks, _ = find_peaks(flux, height=base + 2.0 * mad, distance=3)

    if len(peaks) == 0:
        return []

    energy = np.log1p(S.sum(axis=0))
    energy_smooth = median_filter(energy, size=5)

    segments = []
    search_radius = int(0.15 * sr / hop)

    for pk in peaks:
        start = max(0, pk - search_radius)
        region_before = energy_smooth[start : pk + 1]
        onset_idx = start + np.argmin(region_before) if len(region_before) > 0 else pk

        end = min(len(energy_smooth), pk + search_radius + 1)
        region_after = energy_smooth[pk:end]
        offset_idx = pk + np.argmin(region_after) if len(region_after) > 0 else pk

        onset = onset_idx * hop / sr
        offset = (offset_idx + 1) * hop / sr
        if offset > onset:
            segments.append((onset, offset))

    return segments


# ---------------------------------------------------------------------------
# Method 6: REPET-lite Foreground Extraction
# ---------------------------------------------------------------------------

def method_repet_lite(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via temporal median background subtraction (REPET-lite)."""
    hop, n_fft = 256, 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = S[(freq >= 300) & (freq <= 10000)]

    # Background estimation via temporal median
    Bg = median_filter(S, size=(1, 121))
    Fg = np.maximum(S - Bg, 0.0)

    e = librosa.power_to_db(Fg.sum(axis=0) + 1e-12)
    n0 = np.percentile(e, 25)
    on_thr, off_thr = n0 + 7.0, n0 + 4.0

    # Hysteresis
    active = np.zeros_like(e, dtype=bool)
    st = False
    for i, v in enumerate(e):
        st = (v > on_thr) or (st and v > off_thr)
        active[i] = st

    return _labels_to_segments(active, hop, sr)


# ---------------------------------------------------------------------------
# Method 8: 2-State Gaussian HMM (Viterbi)
# ---------------------------------------------------------------------------

def method_hmm_viterbi(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via self-adapting 2-state Gaussian HMM with Viterbi decoding."""
    hop, n_fft = 256, 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    S = S[(freq >= 300) & (freq <= 10000)]

    E = np.log1p(S.sum(axis=0))
    F = librosa.feature.spectral_flatness(S=S + 1e-8)[0]
    # Align lengths
    min_len = min(len(E), len(F))
    E, F = E[:min_len], F[:min_len]
    dE = np.r_[0, np.maximum(0, np.diff(E))]
    X = np.c_[E, -np.log(F + 1e-8), dE]

    # Self-adaptive: estimate parameters from the recording itself
    q = np.quantile(E, [0.35, 0.75])
    z0 = X[E < q[0]]  # silence/noise frames
    z1 = X[E > q[1]]  # vocalization frames

    if len(z0) < 5 or len(z1) < 5:
        # Fallback: not enough data, return empty
        return []

    mu = np.stack([z0.mean(0), z1.mean(0)])
    var = np.stack([z0.var(0) + 1e-4, z1.var(0) + 1e-4])

    # Emission log-probabilities
    logp = np.stack([
        -0.5 * np.sum((X - mu[k]) ** 2 / var[k] + np.log(var[k]), axis=1)
        for k in [0, 1]
    ], axis=1)

    # Transition matrix (prefer staying in same state)
    A = np.log(np.array([[0.995, 0.005], [0.01, 0.99]]))

    # Viterbi
    T = len(X)
    dp = np.zeros((T, 2))
    bp = np.zeros((T, 2), dtype=int)
    dp[0] = logp[0]
    for t in range(1, T):
        for k in [0, 1]:
            v = dp[t - 1] + A[:, k]
            bp[t, k] = np.argmax(v)
            dp[t, k] = v[bp[t, k]] + logp[t, k]

    # Backtrace
    st = np.zeros(T, dtype=int)
    st[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        st[t] = bp[t + 1, st[t + 1]]

    return _labels_to_segments(st == 1, hop, sr)


# ---------------------------------------------------------------------------
# Method 12: Spectral Entropy Change Point
# ---------------------------------------------------------------------------

def method_spectral_entropy(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via spectral entropy (low entropy = tonal bird sounds)."""
    hop, n_fft = 256, 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freq >= 500) & (freq <= 12000)
    S_band = S[band, :]

    # Normalize each frame to probability distribution
    S_norm = S_band / (S_band.sum(axis=0, keepdims=True) + 1e-10)

    # Spectral entropy per frame
    H = -np.sum(S_norm * np.log2(S_norm + 1e-10), axis=0)
    H_max = np.log2(S_band.shape[0])
    H_norm = H / H_max  # 0=pure tone, 1=white noise

    H_smooth = medfilt(H_norm, kernel_size=11)

    # Bird frames = low entropy
    med = np.median(H_smooth)
    mad_val = np.median(np.abs(H_smooth - med))
    threshold = med - 2.5 * mad_val * 1.4826

    detected = H_smooth < threshold

    # Merge gaps
    gap_frames = int(0.1 * sr / hop)
    if gap_frames > 1:
        detected = binary_closing(detected, structure=np.ones(gap_frames))

    return _labels_to_segments(detected, hop, sr)


# ---------------------------------------------------------------------------
# Method 14: Dual-Axis Median Clipping (Lasseck)
# ---------------------------------------------------------------------------

def method_median_clipping(y: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Detect vocalisations via Lasseck-style dual-axis median clipping."""
    hop, n_fft = 512, 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freq >= 500) & (freq <= 12000)
    S_band = S_db[band, :]

    clip_factor = 3.0

    # Row-wise: use MAD for robust spread estimation
    row_median = np.median(S_band, axis=1, keepdims=True)
    row_mad = np.median(np.abs(S_band - row_median), axis=1, keepdims=True) * 1.4826 + 1e-6
    row_mask = S_band > (row_median + clip_factor * row_mad)

    # Column-wise: same approach
    col_median = np.median(S_band, axis=0, keepdims=True)
    col_mad = np.median(np.abs(S_band - col_median), axis=0, keepdims=True) * 1.4826 + 1e-6
    col_mask = S_band > (col_median + clip_factor * col_mad)

    # AND condition
    binary_mask = row_mask & col_mask

    # Morphological cleanup
    struct_2d = np.ones((3, 5))
    binary_mask = binary_closing(binary_mask, structure=struct_2d)

    # Remove small blobs
    labels_2d, n_blobs = label(binary_mask)
    min_area = 20
    for i in range(1, n_blobs + 1):
        if np.sum(labels_2d == i) < min_area:
            binary_mask[labels_2d == i] = False

    # Project to time axis
    activity = binary_mask.any(axis=0)

    # Merge gaps
    gap_frames = int(0.2 * sr / hop)
    if gap_frames > 1:
        activity = binary_closing(activity, structure=np.ones(gap_frames))

    return _labels_to_segments(activity, hop, sr)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_FUNCS = {
    1: method_band_energy_hysteresis,
    2: method_pcen_connected_components,
    3: method_flux_anchor,
    6: method_repet_lite,
    8: method_hmm_viterbi,
    12: method_spectral_entropy,
    14: method_median_clipping,
}

ALL_METHOD_IDS = sorted(METHOD_FUNCS.keys())


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_comparison(
    y: np.ndarray,
    sr: int,
    all_segments: dict,
    output_path: Path,
    recording_id: str,
    active_methods: List[int],
) -> None:
    """Create a comparison figure with spectrogram + segment overlays for each method."""
    n_rows = 1 + len(active_methods)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    # Compute mel spectrogram for display
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmin=150, fmax=12000, hop_length=512,
    )
    S_db = librosa.power_to_db(S_mel, ref=np.max)
    duration = len(y) / sr

    # Row 1: plain spectrogram
    ax0 = axes[0]
    librosa.display.specshow(
        S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
        ax=ax0, cmap="gray_r", fmin=150, fmax=12000,
    )
    ax0.set_ylabel("Spectrogram")
    ax0.set_title(f"{recording_id}  ({duration:.1f}s, sr={sr})")

    # Rows 2+: spectrogram with segment overlays
    for row_idx, method_id in enumerate(active_methods):
        ax = axes[row_idx + 1]
        librosa.display.specshow(
            S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
            ax=ax, cmap="gray_r", fmin=150, fmax=12000,
        )
        color = METHOD_COLORS[method_id]
        segments = all_segments.get(method_id, [])
        for onset, offset in segments:
            ax.axvspan(onset, offset, alpha=0.3, color=color)
        n_segs = len(segments)
        ax.set_ylabel(f"M{method_id} ({n_segs})")
        # Add method name as text inside the plot
        ax.text(
            0.01, 0.92, METHOD_NAMES[method_id],
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
        )

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(
    row: pd.Series,
    active_methods: List[int],
) -> List[dict]:
    """Process a single audio file with all active methods. Returns result rows."""
    recording_id = row["recording_id"]
    species_code = row["ebird_species_code"]

    # Resolve file path from test_samples copy
    original_path = Path(row["file_path"])
    if not original_path.is_absolute():
        original_path = NAS_BASE / original_path
    file_path = TEST_SAMPLES_DIR / species_code / original_path.name

    if not file_path.exists():
        print(f"  WARNING: File not found: {file_path}")
        return []

    # Load audio
    try:
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception as exc:
        print(f"  WARNING: Failed to load {file_path}: {exc}")
        return []

    if len(y) < sr * 0.1:
        print(f"  WARNING: Too short ({len(y)/sr:.2f}s), skipping: {file_path}")
        return []

    # Run each method
    all_segments: dict = {}
    results = []

    for method_id in active_methods:
        method_func = METHOD_FUNCS[method_id]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_segments = method_func(y, sr)
            segments = postprocess_segments(raw_segments)
        except Exception as exc:
            print(f"  WARNING: Method {method_id} failed on {recording_id}: {exc}")
            segments = []

        all_segments[method_id] = segments

        total_dur = sum(off - on for on, off in segments)
        mean_dur = total_dur / len(segments) if segments else 0.0

        safe_id = recording_id.replace(":", "_").replace("/", "_")
        vis_path = RESULTS_DIR / species_code / f"{safe_id}_comparison.png"

        results.append({
            "recording_id": recording_id,
            "ebird_species_code": species_code,
            "method": method_id,
            "n_segments": len(segments),
            "total_duration_sec": round(total_dur, 3),
            "mean_segment_sec": round(mean_dur, 3),
            "visualization_path": str(vis_path),
        })

    # Create comparison visualization
    safe_id = recording_id.replace(":", "_").replace("/", "_")
    vis_path = RESULTS_DIR / species_code / f"{safe_id}_comparison.png"
    try:
        visualize_comparison(y, sr, all_segments, vis_path, recording_id, active_methods)
    except Exception as exc:
        print(f"  WARNING: Visualization failed for {recording_id}: {exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype comparison of bird vocalization segmentation methods.",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Process only this eBird species code (e.g. brebul1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of method IDs to run (e.g. 1,6,14).",
    )
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        active_methods = sorted(int(m) for m in args.methods.split(","))
        invalid = [m for m in active_methods if m not in METHOD_FUNCS]
        if invalid:
            print(f"ERROR: Unknown method IDs: {invalid}. Valid: {ALL_METHOD_IDS}")
            sys.exit(1)
    else:
        active_methods = ALL_METHOD_IDS

    # Load test samples
    if not SAMPLES_CSV.exists():
        print(f"ERROR: Test samples CSV not found: {SAMPLES_CSV}")
        sys.exit(1)

    df = pd.read_csv(SAMPLES_CSV)
    print(f"Loaded {len(df)} test samples from {SAMPLES_CSV}")

    # Filter by species
    if args.species:
        df = df[df["ebird_species_code"] == args.species]
        if df.empty:
            print(f"ERROR: No samples found for species '{args.species}'.")
            sys.exit(1)
        print(f"Filtered to {len(df)} samples for species '{args.species}'")

    # Limit
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} files")

    print(f"Running methods: {active_methods}")
    print(f"Output directory: {RESULTS_DIR}")
    print()

    # Process all files
    all_results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        print(f"[{i + 1}/{total}] {species} / {rec_id}")

        results = process_file(row, active_methods)
        all_results.extend(results)

    # Save summary CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved {len(results_df)} result rows to {RESULTS_CSV}")

        # Print summary statistics
        print("\n--- Summary ---")
        for method_id in active_methods:
            mdf = results_df[results_df["method"] == method_id]
            n_files = len(mdf)
            total_segs = mdf["n_segments"].sum()
            mean_segs = mdf["n_segments"].mean()
            mean_dur = mdf["mean_segment_sec"].mean()
            print(
                f"  M{method_id:2d} ({METHOD_NAMES[method_id]:30s}): "
                f"{n_files} files, {total_segs:6d} segments, "
                f"{mean_segs:5.1f} segs/file, {mean_dur:.3f}s mean dur"
            )
    else:
        print("\nNo results produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
