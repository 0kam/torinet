"""
PANNs framewise output を使った鳥類音声セグメンテーション実験。

PANNs (Pretrained Audio Neural Networks) の Cnn14_DecisionLevelMax モデルから
framewise_output を取得し、AudioSet の鳥関連クラスの確信度を合算して
鳥音活性区間を検出する。ヒステリシス閾値処理 + PCEN エネルギー谷への
境界スナップで精密化。

使い方:
  python prototype_panns.py                    # 全ファイル処理
  python prototype_panns.py --species brebul1  # 1種のみ
  python prototype_panns.py --limit 5          # 先頭5ファイルのみ
  python prototype_panns.py --on-threshold 0.4 # 閾値調整
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
import torch
from scipy.ndimage import label

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
RESULTS_DIR = NAS_BASE / "segments" / "test_samples_results_panns"
RESULTS_CSV = STEP_DIR / "prototype_results_panns.csv"

# PANNs sample rate (fixed by the model)
PANNS_SR = 32000

# AudioSet bird-related class indices
# Gathered from class_labels_indices.csv
BIRD_CLASS_INDICES = [
    98,   # Fowl
    99,   # Chicken, rooster
    101,  # Crowing, cock-a-doodle-doo
    102,  # Turkey
    103,  # Gobble
    104,  # Duck
    105,  # Quack
    106,  # Goose
    111,  # Bird
    112,  # Bird vocalization, bird call, bird song
    113,  # Chirp, tweet
    114,  # Squawk
    115,  # Pigeon, dove
    116,  # Coo
    117,  # Crow
    118,  # Caw
    119,  # Owl
    121,  # Bird flight, flapping wings
    473,  # Flap
    503,  # Chirp tone
]


# ---------------------------------------------------------------------------
# PANNs model wrapper
# ---------------------------------------------------------------------------

class PANNsDetector:
    """Wrapper around PANNs SoundEventDetection for bird activity detection."""

    def __init__(self, device: str = "cuda"):
        from panns_inference import SoundEventDetection

        self.sed = SoundEventDetection(device=device)
        self.device = self.sed.device
        self.bird_indices = np.array(BIRD_CLASS_INDICES)

    def get_bird_activity(
        self, y: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, float]:
        """Compute framewise bird activity from audio.

        Args:
            y: Audio waveform (mono).
            sr: Sample rate of the input audio.

        Returns:
            bird_activity: 1-D array of bird confidence per frame.
            frame_duration: Duration of each frame in seconds.
        """
        # Resample to PANNs expected rate if needed
        if sr != PANNS_SR:
            y_32k = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SR)
        else:
            y_32k = y

        # PANNs expects (batch, samples) as float32
        audio_input = y_32k[np.newaxis, :].astype(np.float32)

        # Run SED inference
        framewise_output = self.sed.inference(audio_input)
        # framewise_output shape: (1, n_frames, 527)
        fw = framewise_output[0]  # (n_frames, 527)

        # Sum bird-related class probabilities per frame.
        # No cap: allow values > 1.0 to preserve dynamic range for
        # focal recordings where multiple bird classes fire.
        bird_activity = fw[:, self.bird_indices].sum(axis=1)

        # Compute frame duration
        # PANNs model: hop_size=320, sr=32000 => base frame = 10ms
        # But the model applies pooling, so actual frame count differs.
        # Compute empirically from output shape.
        audio_duration = len(y_32k) / PANNS_SR
        n_frames = len(bird_activity)
        frame_duration = audio_duration / n_frames

        return bird_activity, frame_duration


# ---------------------------------------------------------------------------
# Segment detection from bird activity curve
# ---------------------------------------------------------------------------

def hysteresis_threshold(
    activity: np.ndarray,
    on_threshold: float = 0.3,
    off_threshold: float = 0.15,
) -> np.ndarray:
    """Apply hysteresis thresholding to activity curve.

    Returns a boolean array indicating active frames.
    """
    active = np.zeros(len(activity), dtype=bool)
    on = False
    for i, v in enumerate(activity):
        if not on and v >= on_threshold:
            on = True
        elif on and v < off_threshold:
            on = False
        active[i] = on
    return active


def snap_boundaries_to_energy_valleys(
    segments: List[Tuple[float, float]],
    y: np.ndarray,
    sr: int,
    search_window: float = 0.05,
) -> List[Tuple[float, float]]:
    """Snap segment boundaries to nearby PCEN energy valleys.

    Uses PCEN-normalized energy to find the lowest-energy point near
    each boundary, yielding cleaner onset/offset times.
    """
    if not segments:
        return segments

    hop = 128
    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=64, fmin=150, fmax=12000, hop_length=hop, power=1.0,
    )
    P = librosa.pcen(M * (2**16), sr=sr, hop_length=hop)
    energy = np.log1p(P).sum(axis=0)

    search_frames = max(1, int(search_window * sr / hop))
    n_energy = len(energy)

    snapped = []
    for onset, offset in segments:
        # Convert to frame indices
        onset_frame = int(onset * sr / hop)
        offset_frame = int(offset * sr / hop)

        # Search for energy valley near onset
        search_start = max(0, onset_frame - search_frames)
        search_end = min(n_energy, onset_frame + search_frames + 1)
        if search_end > search_start:
            valley_idx = search_start + np.argmin(energy[search_start:search_end])
            onset_snapped = valley_idx * hop / sr
        else:
            onset_snapped = onset

        # Search for energy valley near offset
        search_start = max(0, offset_frame - search_frames)
        search_end = min(n_energy, offset_frame + search_frames + 1)
        if search_end > search_start:
            valley_idx = search_start + np.argmin(energy[search_start:search_end])
            offset_snapped = valley_idx * hop / sr
        else:
            offset_snapped = offset

        # Ensure onset < offset
        if onset_snapped < offset_snapped:
            snapped.append((onset_snapped, offset_snapped))
        else:
            snapped.append((onset, offset))

    return snapped


def compute_adaptive_thresholds(
    bird_activity: np.ndarray,
    on_threshold: float,
    off_threshold: float,
) -> Tuple[float, float]:
    """Compute adaptive thresholds based on the activity distribution.

    For focal recordings where bird activity is consistently high,
    fixed thresholds produce one giant segment. Instead, we place
    thresholds relative to the recording's own distribution so that
    inter-vocalization dips create segment boundaries.

    Returns adjusted (on_threshold, off_threshold).
    """
    p10 = np.percentile(bird_activity, 10)
    p90 = np.percentile(bird_activity, 90)
    dynamic_range = p90 - p10

    # If the minimum activity is already above the on_threshold,
    # the entire recording would be one segment. Adapt.
    if p10 >= on_threshold:
        # Place thresholds relative to the activity distribution
        adapted_off = p10 + 0.3 * dynamic_range
        adapted_on = p10 + 0.5 * dynamic_range
        return adapted_on, adapted_off

    return on_threshold, off_threshold


def detect_segments(
    bird_activity: np.ndarray,
    frame_duration: float,
    y: np.ndarray,
    sr: int,
    on_threshold: float = 0.3,
    off_threshold: float = 0.15,
    merge_gap: float = 0.15,
    min_duration: float = 0.05,
    max_duration: float = 60.0,
    snap_boundaries: bool = True,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Detect bird vocalization segments from PANNs bird activity curve.

    Pipeline:
      1. Adaptive threshold computation
      2. Hysteresis thresholding on bird_activity
      3. Convert boolean frames to (onset, offset) pairs
      4. Merge nearby segments
      5. Filter by duration
      6. Snap boundaries to PCEN energy valleys

    Returns:
        (segments, actual_on_threshold, actual_off_threshold)
    """
    # Step 1: Adaptive threshold computation
    on_threshold, off_threshold = compute_adaptive_thresholds(
        bird_activity, on_threshold, off_threshold,
    )

    # Step 2: Hysteresis thresholding
    active = hysteresis_threshold(bird_activity, on_threshold, off_threshold)

    # Step 2: Convert to segments
    lab_arr, n = label(active)
    raw_segments: List[Tuple[float, float]] = []
    for k in range(1, n + 1):
        idx = np.where(lab_arr == k)[0]
        onset = idx[0] * frame_duration
        offset = (idx[-1] + 1) * frame_duration
        raw_segments.append((onset, offset))

    if not raw_segments:
        return [], on_threshold, off_threshold

    # Step 4: Merge close segments
    raw_segments.sort(key=lambda s: s[0])
    merged: List[Tuple[float, float]] = [raw_segments[0]]
    for onset, offset in raw_segments[1:]:
        if onset - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], offset))
        else:
            merged.append((onset, offset))

    # Step 5: Filter by duration
    filtered = [
        (on, off)
        for on, off in merged
        if min_duration <= (off - on) <= max_duration
    ]

    # Step 6: Snap boundaries to energy valleys
    if snap_boundaries and filtered:
        filtered = snap_boundaries_to_energy_valleys(filtered, y, sr)

    return filtered, on_threshold, off_threshold


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_result(
    y: np.ndarray,
    sr: int,
    bird_activity: np.ndarray,
    frame_duration: float,
    segments: List[Tuple[float, float]],
    output_path: Path,
    recording_id: str,
    on_threshold: float,
    off_threshold: float,
) -> None:
    """Create visualization with spectrogram, bird activity curve, and detected segments."""
    duration = len(y) / sr
    time_axis = np.arange(len(bird_activity)) * frame_duration

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    # Row 1: Mel spectrogram
    S_mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmin=150, fmax=12000, hop_length=512,
    )
    S_db = librosa.power_to_db(S_mel, ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
        ax=axes[0], cmap="gray_r", fmin=150, fmax=12000,
    )
    axes[0].set_ylabel("Spectrogram")
    axes[0].set_title(f"PANNs Bird Detection: {recording_id}  ({duration:.1f}s, sr={sr})")

    # Row 2: Bird activity curve with thresholds
    axes[1].plot(time_axis, bird_activity, color="steelblue", linewidth=0.8, label="Bird activity")
    axes[1].axhline(y=on_threshold, color="red", linestyle="--", linewidth=0.7, alpha=0.7, label=f"ON thr={on_threshold:.3f}")
    axes[1].axhline(y=off_threshold, color="orange", linestyle="--", linewidth=0.7, alpha=0.7, label=f"OFF thr={off_threshold:.3f}")
    axes[1].fill_between(time_axis, bird_activity, alpha=0.2, color="steelblue")
    axes[1].set_ylabel("Bird confidence")
    y_max = max(bird_activity.max() * 1.05, 1.02)
    axes[1].set_ylim(-0.02, y_max)
    axes[1].legend(loc="upper right", fontsize=7)

    # Row 3: Spectrogram with segment overlays
    librosa.display.specshow(
        S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
        ax=axes[2], cmap="gray_r", fmin=150, fmax=12000,
    )
    for onset, offset in segments:
        axes[2].axvspan(onset, offset, alpha=0.35, color="limegreen")
    n_segs = len(segments)
    total_dur = sum(off - on for on, off in segments)
    axes[2].set_ylabel(f"Segments ({n_segs})")
    axes[2].text(
        0.01, 0.92,
        f"PANNs SED: {n_segs} segments, {total_dur:.1f}s total",
        transform=axes[2].transAxes, fontsize=8, va="top",
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
    detector: PANNsDetector,
    on_threshold: float,
    off_threshold: float,
    merge_gap: float,
    min_duration: float,
    max_duration: float,
) -> dict:
    """Process a single audio file. Returns a result dict or None on failure."""
    recording_id = row["recording_id"]
    species_code = row["ebird_species_code"]

    # Resolve file path
    original_path = Path(row["file_path"])
    file_path = TEST_SAMPLES_DIR / species_code / original_path.name

    if not file_path.exists():
        print(f"  WARNING: File not found: {file_path}")
        return None

    # Load audio
    try:
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception as exc:
        print(f"  WARNING: Failed to load {file_path}: {exc}")
        return None

    if len(y) < sr * 0.1:
        print(f"  WARNING: Too short ({len(y)/sr:.2f}s), skipping: {file_path}")
        return None

    # Get bird activity from PANNs
    try:
        bird_activity, frame_duration = detector.get_bird_activity(y, sr)
    except Exception as exc:
        print(f"  WARNING: PANNs inference failed for {recording_id}: {exc}")
        return None

    print(f"    PANNs: {len(bird_activity)} frames, {frame_duration*1000:.1f}ms/frame, "
          f"peak={bird_activity.max():.3f}, mean={bird_activity.mean():.3f}")

    # Detect segments
    segments, actual_on, actual_off = detect_segments(
        bird_activity, frame_duration, y, sr,
        on_threshold=on_threshold,
        off_threshold=off_threshold,
        merge_gap=merge_gap,
        min_duration=min_duration,
        max_duration=max_duration,
    )

    if actual_on != on_threshold or actual_off != off_threshold:
        print(f"    Adaptive thresholds: ON={actual_on:.3f}, OFF={actual_off:.3f}")

    total_dur = sum(off - on for on, off in segments)
    mean_dur = total_dur / len(segments) if segments else 0.0

    # Visualize
    safe_id = recording_id.replace(":", "_").replace("/", "_")
    vis_path = RESULTS_DIR / species_code / f"{safe_id}_panns.png"
    try:
        visualize_result(
            y, sr, bird_activity, frame_duration, segments,
            vis_path, recording_id, actual_on, actual_off,
        )
    except Exception as exc:
        print(f"  WARNING: Visualization failed for {recording_id}: {exc}")

    return {
        "recording_id": recording_id,
        "ebird_species_code": species_code,
        "n_segments": len(segments),
        "total_duration_sec": round(total_dur, 3),
        "mean_segment_dur": round(mean_dur, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PANNs-based bird vocalization segmentation prototype.",
    )
    parser.add_argument(
        "--species", type=str, default=None,
        help="Process only this eBird species code (e.g. brebul1).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N files.",
    )
    parser.add_argument(
        "--on-threshold", type=float, default=0.3,
        help="Hysteresis ON threshold for bird activity (default: 0.3).",
    )
    parser.add_argument(
        "--off-threshold", type=float, default=0.15,
        help="Hysteresis OFF threshold for bird activity (default: 0.15).",
    )
    parser.add_argument(
        "--merge-gap", type=float, default=0.15,
        help="Maximum gap (sec) to merge adjacent segments (default: 0.15).",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.05,
        help="Minimum segment duration in seconds (default: 0.05).",
    )
    parser.add_argument(
        "--max-duration", type=float, default=60.0,
        help="Maximum segment duration in seconds (default: 60.0).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for PANNs inference: cuda or cpu (default: cuda).",
    )
    args = parser.parse_args()

    # Load test samples
    if not SAMPLES_CSV.exists():
        print(f"ERROR: Test samples CSV not found: {SAMPLES_CSV}")
        sys.exit(1)

    df = pd.read_csv(SAMPLES_CSV)
    print(f"Loaded {len(df)} test samples from {SAMPLES_CSV}")

    if args.species:
        df = df[df["ebird_species_code"] == args.species]
        if df.empty:
            print(f"ERROR: No samples found for species '{args.species}'.")
            sys.exit(1)
        print(f"Filtered to {len(df)} samples for species '{args.species}'")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} files")

    print(f"Thresholds: ON={args.on_threshold}, OFF={args.off_threshold}")
    print(f"Merge gap={args.merge_gap}s, Duration range=[{args.min_duration}, {args.max_duration}]s")
    print(f"Output directory: {RESULTS_DIR}")
    print()

    # Initialize PANNs detector
    print("Loading PANNs Cnn14_DecisionLevelMax model...")
    detector = PANNsDetector(device=args.device)
    print()

    # Process all files
    all_results = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        print(f"[{i + 1}/{total}] {species} / {rec_id}")

        result = process_file(
            row, detector,
            on_threshold=args.on_threshold,
            off_threshold=args.off_threshold,
            merge_gap=args.merge_gap,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        if result is not None:
            all_results.append(result)

    # Save results CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved {len(results_df)} results to {RESULTS_CSV}")

        # Print summary
        print("\n--- Summary ---")
        print(f"  Files processed: {len(results_df)}")
        print(f"  Total segments:  {results_df['n_segments'].sum()}")
        print(f"  Mean segs/file:  {results_df['n_segments'].mean():.1f}")
        print(f"  Mean seg dur:    {results_df['mean_segment_dur'].mean():.3f}s")
        print(f"  Mean total dur:  {results_df['total_duration_sec'].mean():.1f}s")

        # Per-species summary
        print("\n--- Per-species ---")
        for species, sdf in results_df.groupby("ebird_species_code"):
            print(
                f"  {species:12s}: {len(sdf)} files, "
                f"{sdf['n_segments'].sum():4d} segs, "
                f"{sdf['n_segments'].mean():.1f} segs/file, "
                f"{sdf['mean_segment_dur'].mean():.3f}s mean dur"
            )
    else:
        print("\nNo results produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
