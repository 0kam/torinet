"""
Bout grouping and BirdNET classification pipeline for TweetyNet segments.

Converts frame-level TweetyNet predictions into bout-level segments,
classifies them with BirdNET, and produces visualizations.

Subcommands:
  group-bouts    - Group TweetyNet frame predictions into bouts
  classify-bouts - Run BirdNET on each bout for species filtering
  visualize      - Generate spectrogram + bout overlay plots

Usage:
  python bout_pipeline.py group-bouts [--species CODE] [--max-gap 0.4]
  python bout_pipeline.py classify-bouts [--species CODE]
  python bout_pipeline.py visualize [--species CODE] [--limit 5]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress TF warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
TWEETYNET_DIR = NAS_BASE / "segments" / "test_samples_results_tweetynet"
BOUTS_DIR = NAS_BASE / "segments" / "test_samples_results_bouts"

# TweetyNet frame parameters (must match prototype_tweetynet.py)
SR = 32000
HOP_LENGTH = 320
FRAME_DUR = HOP_LENGTH / SR  # ~0.01s per frame

# BirdNET parameters
BIRDNET_SR = 48000
BIRDNET_SEGMENT_S = 3.0


# ===========================================================================
# Bout grouping
# ===========================================================================


def extract_notes(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_dur: float = 0.02,
) -> list[tuple[float, float]]:
    """Extract note-level segments from frame probabilities.

    Args:
        probs: 1D array of frame-level probabilities.
        threshold: Binarization threshold.
        min_dur: Minimum note duration in seconds.

    Returns:
        List of (onset, offset) tuples in seconds.
    """
    binary = probs >= threshold
    min_frames = max(1, int(min_dur / FRAME_DUR))

    notes = []
    in_note = False
    start = 0
    for i, val in enumerate(binary):
        if val and not in_note:
            start = i
            in_note = True
        elif not val and in_note:
            if (i - start) >= min_frames:
                notes.append((start * FRAME_DUR, i * FRAME_DUR))
            in_note = False
    # Handle note at end
    if in_note and (len(binary) - start) >= min_frames:
        notes.append((start * FRAME_DUR, len(binary) * FRAME_DUR))

    return notes


def group_notes_into_bouts(
    notes: list[tuple[float, float]],
    max_gap: float = 0.4,
    max_bout_duration: float = 8.0,
    max_silence_ratio: float = 0.6,
) -> list[dict[str, Any]]:
    """Group notes into bouts based on gap and constraint thresholds.

    Args:
        notes: List of (onset, offset) tuples in seconds.
        max_gap: Maximum gap between notes to merge into a bout.
        max_bout_duration: Maximum bout duration before splitting.
        max_silence_ratio: Maximum ratio of silence within a bout.

    Returns:
        List of bout dicts with keys: bout_onset, bout_offset, notes,
        n_notes, total_note_duration, silence_ratio.
    """
    if not notes:
        return []

    # Initial grouping by max_gap
    raw_bouts: list[list[tuple[float, float]]] = [[notes[0]]]
    for note in notes[1:]:
        gap = note[0] - raw_bouts[-1][-1][1]
        if gap <= max_gap:
            raw_bouts[-1].append(note)
        else:
            raw_bouts.append([note])

    # Split bouts that violate constraints
    final_bouts = []
    for bout_notes in raw_bouts:
        final_bouts.extend(
            _split_bout_if_needed(
                bout_notes, max_gap, max_bout_duration, max_silence_ratio
            )
        )

    return final_bouts


def _split_bout_if_needed(
    notes: list[tuple[float, float]],
    max_gap: float,
    max_bout_duration: float,
    max_silence_ratio: float,
) -> list[dict[str, Any]]:
    """Split a bout at the largest gap if it violates constraints."""
    bout = _make_bout_dict(notes)

    # Check constraints
    duration = bout["bout_offset"] - bout["bout_onset"]
    if duration <= max_bout_duration and bout["silence_ratio"] <= max_silence_ratio:
        return [bout]

    if len(notes) <= 1:
        return [bout]

    # Find the largest gap to split at
    gaps = [(notes[i + 1][0] - notes[i][1], i) for i in range(len(notes) - 1)]
    gaps.sort(reverse=True)
    _, split_idx = gaps[0]

    left = notes[: split_idx + 1]
    right = notes[split_idx + 1 :]

    # Recurse
    result = []
    result.extend(
        _split_bout_if_needed(left, max_gap, max_bout_duration, max_silence_ratio)
    )
    result.extend(
        _split_bout_if_needed(right, max_gap, max_bout_duration, max_silence_ratio)
    )
    return result


def _make_bout_dict(notes: list[tuple[float, float]]) -> dict[str, Any]:
    """Create a bout dict from a list of notes."""
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


def process_group_bouts(
    df: pd.DataFrame,
    max_gap: float,
    max_bout_duration: float,
    max_silence_ratio: float,
) -> pd.DataFrame:
    """Run bout grouping on all recordings in df.

    Returns a summary DataFrame with per-recording statistics.
    """
    summary_rows = []

    for _, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")

        pred_path = TWEETYNET_DIR / species / f"{safe_id}_pred.npz"
        if not pred_path.exists():
            print(f"  SKIP {species}/{safe_id}: no prediction file")
            continue

        probs = np.load(pred_path)["probs"]
        notes = extract_notes(probs)
        bouts = group_notes_into_bouts(
            notes,
            max_gap=max_gap,
            max_bout_duration=max_bout_duration,
            max_silence_ratio=max_silence_ratio,
        )

        # Save bout JSON
        out_dir = BOUTS_DIR / species
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_id}_bouts.json"

        bout_data = {
            "recording_id": rec_id,
            "species_code": species,
            "scientific_name": row["scientific_name"],
            "frame_dur": FRAME_DUR,
            "n_frames": len(probs),
            "parameters": {
                "threshold": 0.5,
                "min_note_dur": 0.02,
                "max_gap": max_gap,
                "max_bout_duration": max_bout_duration,
                "max_silence_ratio": max_silence_ratio,
            },
            "n_notes": len(notes),
            "n_bouts": len(bouts),
            "bouts": bouts,
        }

        with open(out_path, "w") as f:
            json.dump(bout_data, f, indent=2)

        # Summary stats
        if bouts:
            durations = [b["bout_offset"] - b["bout_onset"] for b in bouts]
            summary_rows.append(
                {
                    "species_code": species,
                    "recording_id": rec_id,
                    "n_notes": len(notes),
                    "n_bouts": len(bouts),
                    "mean_bout_dur": round(np.mean(durations), 3),
                    "median_bout_dur": round(np.median(durations), 3),
                    "total_bout_dur": round(sum(durations), 3),
                    "audio_dur": round(len(probs) * FRAME_DUR, 3),
                }
            )
        else:
            summary_rows.append(
                {
                    "species_code": species,
                    "recording_id": rec_id,
                    "n_notes": 0,
                    "n_bouts": 0,
                    "mean_bout_dur": 0.0,
                    "median_bout_dur": 0.0,
                    "total_bout_dur": 0.0,
                    "audio_dur": round(len(probs) * FRAME_DUR, 3),
                }
            )

        print(
            f"  {species}/{safe_id}: {len(notes)} notes -> {len(bouts)} bouts"
        )

    return pd.DataFrame(summary_rows)


# ===========================================================================
# BirdNET classification
# ===========================================================================


def _build_species_lookup(species_list: np.ndarray) -> dict[str, int]:
    """Build a dict mapping scientific name -> index in BirdNET species list."""
    lookup = {}
    for idx, name in enumerate(species_list):
        sci = name.split("_")[0]
        lookup[sci] = idx
    return lookup


def _load_bout_audio(
    audio_path: Path,
    bout: dict[str, Any],
) -> np.ndarray:
    """Load and prepare bout audio for BirdNET.

    If bout < 3s, pad with surrounding context (or zero-pad at boundaries).
    Audio is loaded at BirdNET's sample rate (48kHz).
    """
    onset = bout["bout_onset"]
    offset = bout["bout_offset"]
    bout_dur = offset - onset

    if bout_dur >= BIRDNET_SEGMENT_S:
        # Load bout as-is
        audio, _ = librosa.load(
            audio_path, sr=BIRDNET_SR, offset=onset, duration=bout_dur
        )
        return audio

    # Need to pad to 3s
    pad_total = BIRDNET_SEGMENT_S - bout_dur
    pad_before = pad_total / 2
    pad_after = pad_total / 2

    load_onset = max(0.0, onset - pad_before)
    actual_pad_before = onset - load_onset
    load_offset = offset + pad_after + (pad_before - actual_pad_before)
    load_dur = load_offset - load_onset

    audio, _ = librosa.load(
        audio_path, sr=BIRDNET_SR, offset=load_onset, duration=load_dur
    )

    # Zero-pad if we couldn't get enough from the file
    target_samples = int(BIRDNET_SEGMENT_S * BIRDNET_SR)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    return audio[:target_samples]


def _score_bout_result(
    result: Any,
    input_idx: int,
    target_idx: int | None,
) -> tuple[float, float, str]:
    """Extract target score, other-max score, and top species from one input.

    Returns:
        (s_target, s_other_max, birdnet_top_species)
    """
    n_segments = result.species_probs.shape[1]

    if target_idx is not None:
        target_scores = []
        for seg_idx in range(n_segments):
            seg_ids = result.species_ids[input_idx, seg_idx, :]
            seg_probs = result.species_probs[input_idx, seg_idx, :]
            mask = seg_ids == target_idx
            if mask.any():
                target_scores.append(float(seg_probs[mask].max()))
            else:
                target_scores.append(0.0)

        if target_scores:
            target_scores_sorted = sorted(target_scores, reverse=True)
            n_top = max(1, int(len(target_scores_sorted) * 0.3))
            s_target = float(np.mean(target_scores_sorted[:n_top]))
        else:
            s_target = 0.0
    else:
        s_target = -1.0

    # Other species max score
    n_species = len(result.species_list)
    all_species_scores: dict[int, float] = {}
    best_species_name = "unknown"
    best_species_conf = 0.0
    for seg_idx in range(n_segments):
        seg_ids = result.species_ids[input_idx, seg_idx, :]
        seg_probs = result.species_probs[input_idx, seg_idx, :]
        for sid, sprob in zip(seg_ids, seg_probs):
            sid_int = int(sid)
            sprob_f = float(sprob)
            # Skip invalid indices from padded segments
            if sid_int >= n_species:
                continue
            if sprob_f > best_species_conf:
                best_species_conf = sprob_f
                best_species_name = str(result.species_list[sid_int])
            if sid_int != (target_idx if target_idx is not None else -1):
                all_species_scores[sid_int] = max(
                    all_species_scores.get(sid_int, 0.0), sprob_f
                )

    s_other_max = max(all_species_scores.values()) if all_species_scores else 0.0
    return s_target, s_other_max, best_species_name


def _determine_verdict(s_target: float, s_other_max: float) -> str:
    """Determine accept/review/reject verdict."""
    if s_target < 0:
        return "review"  # Species not in BirdNET
    elif s_target >= 0.30 and (s_target - s_other_max) >= 0.10:
        return "accept"
    elif s_target < 0.10 and s_other_max >= 0.25:
        return "reject"
    else:
        return "review"


def classify_bouts_for_recording(
    model: Any,
    species_lookup: dict[str, int],
    row: pd.Series,
) -> dict[str, Any] | None:
    """Classify all bouts in a recording with BirdNET (batch mode).

    Returns updated bout data dict, or None if no bout file found.
    """
    species = row["ebird_species_code"]
    sci_name = row["scientific_name"]
    rec_id = row["recording_id"]
    safe_id = rec_id.replace(":", "_")
    audio_path = Path(row["file_path"])

    bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
    if not bout_path.exists():
        return None

    with open(bout_path) as f:
        bout_data = json.load(f)

    bouts = bout_data["bouts"]
    if not bouts:
        return bout_data

    # Skip if already classified
    if all("birdnet_verdict" in b for b in bouts):
        return bout_data

    # Find unclassified bouts and load their audio
    target_idx = species_lookup.get(sci_name)
    pending_indices = []
    audio_arrays = []

    for i, bout in enumerate(bouts):
        if "birdnet_verdict" not in bout:
            audio = _load_bout_audio(audio_path, bout)
            audio_arrays.append((audio, BIRDNET_SR))
            pending_indices.append(i)

    if not audio_arrays:
        return bout_data

    # Batch BirdNET inference
    result = model.predict_arrays(
        audio_arrays,
        top_k=10,
        default_confidence_threshold=0.01,
        n_workers=1,
        batch_size=8,
        device="CPU",
    )

    # Process results for each bout
    for batch_idx, bout_idx in enumerate(pending_indices):
        bout = bouts[bout_idx]
        s_target, s_other_max, top_species = _score_bout_result(
            result, batch_idx, target_idx
        )
        verdict = _determine_verdict(s_target, s_other_max)

        bout["birdnet_score"] = round(s_target, 4)
        bout["birdnet_other_max"] = round(s_other_max, 4)
        bout["birdnet_top_species"] = top_species
        bout["birdnet_verdict"] = verdict

    # Save updated JSON
    with open(bout_path, "w") as f:
        json.dump(bout_data, f, indent=2)

    return bout_data


def process_classify_bouts(df: pd.DataFrame) -> pd.DataFrame:
    """Run BirdNET classification on all recordings."""
    import birdnet

    print("Loading BirdNET model...")
    model = birdnet.load("acoustic", "2.4", "tf")
    species_lookup = _build_species_lookup(model._species_list)
    print(f"BirdNET loaded: {len(species_lookup)} species")

    summary_rows = []

    for idx, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")

        bout_data = classify_bouts_for_recording(model, species_lookup, row)
        if bout_data is None:
            print(f"  SKIP {species}/{safe_id}: no bout file")
            continue

        bouts = bout_data["bouts"]
        if not bouts:
            print(f"  {species}/{safe_id}: no bouts")
            continue

        verdicts = [b.get("birdnet_verdict", "unknown") for b in bouts]
        scores = [b.get("birdnet_score", 0.0) for b in bouts]

        summary_rows.append(
            {
                "species_code": species,
                "recording_id": rec_id,
                "n_bouts": len(bouts),
                "n_accept": verdicts.count("accept"),
                "n_review": verdicts.count("review"),
                "n_reject": verdicts.count("reject"),
                "mean_target_score": round(np.mean(scores), 4),
                "max_target_score": round(max(scores), 4),
            }
        )

        n_acc = verdicts.count("accept")
        n_rej = verdicts.count("reject")
        n_rev = verdicts.count("review")
        print(
            f"  {species}/{safe_id}: {len(bouts)} bouts "
            f"(accept={n_acc}, review={n_rev}, reject={n_rej})"
        )

    return pd.DataFrame(summary_rows)


# ===========================================================================
# Visualization
# ===========================================================================

VERDICT_COLORS = {
    "accept": "#2ecc71",   # green
    "review": "#f1c40f",   # yellow
    "reject": "#e74c3c",   # red
}


def visualize_recording(row: pd.Series) -> bool:
    """Generate spectrogram + bout overlay for a single recording.

    Returns True if visualization was created, False otherwise.
    """
    species = row["ebird_species_code"]
    rec_id = row["recording_id"]
    safe_id = rec_id.replace(":", "_")
    audio_path = Path(row["file_path"])

    bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
    if not bout_path.exists():
        return False

    with open(bout_path) as f:
        bout_data = json.load(f)

    bouts = bout_data["bouts"]

    # Load audio for spectrogram
    y, _ = librosa.load(audio_path, sr=SR)
    duration = len(y) / SR

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=HOP_LENGTH,
        n_mels=128, fmin=150, fmax=12000,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Create figure
    fig_width = max(12, min(30, duration * 1.5))
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    librosa.display.specshow(
        S_db, sr=SR, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax,
        fmin=150, fmax=12000, cmap="magma",
    )

    # Overlay bouts
    for i, bout in enumerate(bouts):
        onset = bout["bout_onset"]
        offset = bout["bout_offset"]
        verdict = bout.get("birdnet_verdict", "review")
        color = VERDICT_COLORS.get(verdict, "#999999")
        score = bout.get("birdnet_score", None)

        # Draw bout rectangle
        ax.axvspan(onset, offset, alpha=0.25, color=color, zorder=2)
        ax.axvline(onset, color=color, linewidth=0.8, alpha=0.7, zorder=3)
        ax.axvline(offset, color=color, linewidth=0.8, alpha=0.7, zorder=3)

        # Draw note-level masks
        for note_on, note_off in bout["notes"]:
            ax.axvspan(
                note_on, note_off, ymin=0, ymax=0.05,
                color=color, alpha=0.8, zorder=4,
            )

        # Score label
        if score is not None:
            label_text = f"{score:.2f}"
            mid = (onset + offset) / 2
            ax.text(
                mid, ax.get_ylim()[1] * 0.92, label_text,
                ha="center", va="top", fontsize=7,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, alpha=0.8),
                zorder=5,
            )

    ax.set_title(
        f"{species} / {safe_id} — "
        f"{len(bouts)} bouts "
        f"({row['scientific_name']})",
        fontsize=10,
    )
    ax.set_xlim(0, duration)

    # Save
    out_path = BOUTS_DIR / species / f"{safe_id}_bouts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def process_visualize(df: pd.DataFrame, limit: int | None = None) -> None:
    """Generate visualizations for recordings."""
    count = 0
    for _, row in df.iterrows():
        if limit and count >= limit:
            break
        species = row["ebird_species_code"]
        safe_id = row["recording_id"].replace(":", "_")
        ok = visualize_recording(row)
        if ok:
            count += 1
            print(f"  {species}/{safe_id}: visualization saved")
        else:
            print(f"  {species}/{safe_id}: SKIP (no bout file)")

    print(f"\nGenerated {count} visualizations.")


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bout grouping and BirdNET classification pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # group-bouts
    gb = sub.add_parser("group-bouts", help="Group TweetyNet frames into bouts")
    gb.add_argument("--species", type=str, default=None, help="Filter by species code")
    gb.add_argument("--max-gap", type=float, default=0.4, help="Max gap between notes (s)")
    gb.add_argument(
        "--max-bout-duration", type=float, default=8.0, help="Max bout duration (s)"
    )
    gb.add_argument(
        "--max-silence-ratio", type=float, default=0.6, help="Max silence ratio in bout"
    )

    # classify-bouts
    cb = sub.add_parser("classify-bouts", help="BirdNET classification of bouts")
    cb.add_argument("--species", type=str, default=None, help="Filter by species code")

    # visualize
    viz = sub.add_parser("visualize", help="Generate bout visualizations")
    viz.add_argument("--species", type=str, default=None, help="Filter by species code")
    viz.add_argument("--limit", type=int, default=None, help="Max recordings to visualize")

    return parser.parse_args()


def load_metadata(species_filter: str | None = None) -> pd.DataFrame:
    """Load test samples metadata, optionally filtered by species."""
    df = pd.read_csv(SAMPLES_CSV)
    if species_filter:
        df = df[df["ebird_species_code"] == species_filter]
        if df.empty:
            print(f"ERROR: No recordings found for species '{species_filter}'")
            sys.exit(1)
    return df


def main() -> None:
    args = parse_args()

    if args.command == "group-bouts":
        df = load_metadata(args.species)
        print(
            f"Grouping bouts: {len(df)} recordings "
            f"(max_gap={args.max_gap}, max_dur={args.max_bout_duration}, "
            f"max_silence={args.max_silence_ratio})"
        )
        summary = process_group_bouts(
            df, args.max_gap, args.max_bout_duration, args.max_silence_ratio
        )
        if not summary.empty:
            out_csv = STEP_DIR / "bout_results.csv"
            summary.to_csv(out_csv, index=False)
            print(f"\nSummary saved to {out_csv}")
            print(f"Total: {len(summary)} recordings, "
                  f"{summary['n_bouts'].sum()} bouts")
            print(f"Median bout duration: {summary['median_bout_dur'].median():.3f}s")

    elif args.command == "classify-bouts":
        df = load_metadata(args.species)
        print(f"Classifying bouts: {len(df)} recordings")
        summary = process_classify_bouts(df)
        if not summary.empty:
            out_csv = STEP_DIR / "classify_results.csv"
            summary.to_csv(out_csv, index=False)
            print(f"\nSummary saved to {out_csv}")
            total = len(summary)
            n_acc = summary["n_accept"].sum()
            n_rev = summary["n_review"].sum()
            n_rej = summary["n_reject"].sum()
            n_all = n_acc + n_rev + n_rej
            print(
                f"Total: {total} recordings, {n_all} bouts "
                f"(accept={n_acc}, review={n_rev}, reject={n_rej})"
            )

    elif args.command == "visualize":
        df = load_metadata(args.species)
        print(f"Visualizing: {len(df)} recordings" +
              (f" (limit={args.limit})" if args.limit else ""))
        process_visualize(df, args.limit)


if __name__ == "__main__":
    main()
