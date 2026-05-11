"""
Compare three segment extraction methods on focal bird recordings.

Methods:
  1. bambird-style: median clipping ROI + spectral features + DBSCAN
  2. biodenoising: Earth Species Project neural denoising + amplitude selection
  3. bird-mixit: Google source separation + channel selection

Subcommands:
  bambird       - Run bambird-style pipeline
  biodenoising  - Run neural denoising + amplitude-based selection
  bird-mixit    - Run source separation + channel selection
  compare       - Generate comparison summary and visualizations

Usage:
  python method_comparison.py bambird [--species CODE] [--limit N]
  python method_comparison.py biodenoising [--species CODE] [--limit N]
  python method_comparison.py bird-mixit [--species CODE] [--limit N]
  python method_comparison.py compare
"""

import argparse
import json
import os
import sys
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
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
BOUTS_DIR = NAS_BASE / "segments" / "test_samples_results_bouts"
TWEETYNET_DIR = NAS_BASE / "segments" / "test_samples_results_tweetynet"

# Output directories per method
BAMBIRD_DIR = NAS_BASE / "segments" / "method_bambird"
BIODENOISING_DIR = NAS_BASE / "segments" / "method_biodenoising"
BIRDMIXIT_DIR = NAS_BASE / "segments" / "method_birdmixit"

# Audio parameters
SR = 32000
HOP_LENGTH = 320
FRAME_DUR = HOP_LENGTH / SR

# Bird-MixIT model
MIXIT_MODEL_DIR = NAS_BASE / "models" / "bird_mixit" / "output_sources4"
MIXIT_SR = 22050


def load_metadata(species_filter=None):
    """Load test samples metadata."""
    df = pd.read_csv(SAMPLES_CSV)
    if species_filter:
        df = df[df["ebird_species_code"] == species_filter]
        if df.empty:
            print(f"ERROR: No recordings found for species '{species_filter}'")
            sys.exit(1)
    return df


def _get_bout_notes(species, safe_id):
    """Load notes from bout JSON."""
    bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
    if not bout_path.exists():
        return []
    with open(bout_path) as f:
        data = json.load(f)
    notes = []
    for bout in data["bouts"]:
        for onset, offset in bout["notes"]:
            notes.append((onset, offset))
    return notes


def _flush_print(msg):
    """Print with flush for nohup compatibility."""
    print(msg, flush=True)


# ===========================================================================
# Method 1: bambird-style (median clipping ROI + spectral features + DBSCAN)
# ===========================================================================


def _roi_spectral_features(y_roi, sr, n_fft=1024, hop=512):
    """Fast spectral features for a ROI audio segment.

    Returns 20-dim feature vector: MFCC(13 mean) + centroid + bandwidth +
    rolloff + flatness + log_dur + log_power + bandwidth_hz.
    """
    if len(y_roi) < n_fft:
        y_roi = np.pad(y_roi, (0, n_fft - len(y_roi)))

    S = librosa.feature.melspectrogram(
        y=y_roi, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64,
        fmin=150, fmax=min(sr // 2, 12000),
    )
    S_db = librosa.power_to_db(S + 1e-10, ref=np.max)

    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    cent = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    bw = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    dur = len(y_roi) / sr
    power = float(np.sum(y_roi ** 2))

    return np.concatenate([
        mfcc_mean,                   # 13
        [cent / 10000],              # 1
        [bw / 10000],               # 1
        [rolloff / 10000],          # 1
        [flatness],                  # 1
        [np.log(max(dur, 0.01))],   # 1
        [np.log(max(power, 1e-10))], # 1
        [bw],                        # 1 (raw bandwidth)
    ]).astype(np.float32)            # total: 20


def _bambird_process_species(species, df_species, target_n=75, max_per_rec=5):
    """Process one species with bambird-style pipeline."""
    from scipy.ndimage import binary_closing, binary_opening, label as nd_label
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    all_rois_data = []
    all_features = []

    for _, row in df_species.iterrows():
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")
        audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name
        if not audio_path.exists():
            continue

        try:
            y, fs = librosa.load(str(audio_path), sr=None)
        except Exception:
            continue

        # Compute spectrogram for ROI extraction
        n_fft_spec = 1024
        hop_spec = 512
        S = np.abs(librosa.stft(y, n_fft=n_fft_spec, hop_length=hop_spec)) ** 2
        S_db = 10 * np.log10(S + 1e-10)

        # Remove stationary background (median subtraction per freq bin)
        S_db_clean = S_db - np.median(S_db, axis=1, keepdims=True)
        S_db_clean = np.maximum(S_db_clean, 0)

        # Median clipping (Lasseck-style)
        row_med = np.median(S_db_clean, axis=1, keepdims=True)
        col_med = np.median(S_db_clean, axis=0, keepdims=True)
        row_med = np.where(row_med < 1e-6, 1e-6, row_med)
        col_med = np.where(col_med < 1e-6, 1e-6, col_med)
        mask = (S_db_clean > 3 * row_med) & (S_db_clean > 3 * col_med)

        # Morphological cleanup
        struct = np.ones((3, 3))
        mask = binary_closing(mask, structure=struct, iterations=1)
        mask = binary_opening(mask, structure=struct, iterations=1)

        labeled, n_rois = nd_label(mask)
        if n_rois == 0:
            continue

        # Time/freq conversion
        times = librosa.frames_to_time(
            np.arange(S.shape[1]), sr=fs, hop_length=hop_spec
        )
        freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft_spec)

        # Use find_objects for fast bounding box extraction
        from scipy.ndimage import find_objects
        slices = find_objects(labeled)

        # Pre-compute ROI sizes and filter
        roi_candidates = []
        for roi_id, sl in enumerate(slices, 1):
            if sl is None:
                continue
            freq_sl, time_sl = sl
            n_pixels = np.count_nonzero(labeled[sl] == roi_id)
            if n_pixels < 30:
                continue

            t_min = times[time_sl.start] if time_sl.start < len(times) else 0
            t_max = times[min(time_sl.stop - 1, len(times) - 1)]
            f_min = freqs[freq_sl.start] if freq_sl.start < len(freqs) else 0
            f_max = freqs[min(freq_sl.stop - 1, len(freqs) - 1)]

            dur = t_max - t_min
            if dur < 0.03 or dur > 10.0:
                continue
            if f_max - f_min < 200:
                continue

            start_samp = int(t_min * fs)
            end_samp = int(t_max * fs)
            y_roi = y[start_samp:end_samp]
            if len(y_roi) < 100:
                continue

            roi_power = float(np.sum(y_roi ** 2))
            roi_candidates.append((roi_power, t_min, t_max, f_min, f_max, y_roi))

        # Limit to top 300 ROIs by power per recording
        roi_candidates.sort(key=lambda x: x[0], reverse=True)
        for roi_power, t_min, t_max, f_min, f_max, y_roi in roi_candidates[:300]:
            try:
                feat_vec = _roi_spectral_features(y_roi, fs)
                if np.any(np.isnan(feat_vec)):
                    continue
            except Exception:
                continue

            all_features.append(feat_vec)
            all_rois_data.append({
                "species_code": species,
                "recording_id": rec_id,
                "safe_id": safe_id,
                "onset": round(t_min, 4),
                "offset": round(t_max, 4),
                "freq_min": round(f_min, 1),
                "freq_max": round(f_max, 1),
                "power": roi_power,
            })

    if len(all_features) < 10:
        _flush_print(f"  {species}: too few ROIs ({len(all_features)})")
        return pd.DataFrame()

    X = np.stack(all_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # k-NN eps estimation for DBSCAN
    k = min(10, len(X_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, -1])
    eps = max(np.percentile(k_dist, 90), 0.5)

    clusterer = DBSCAN(eps=eps, min_samples=max(3, len(X_scaled) // 50))
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    _flush_print(
        f"  {species}: {len(X_scaled)} ROIs -> {n_clusters} clusters, "
        f"{n_noise} noise ({n_noise / len(X_scaled):.1%})"
    )

    rois_df = pd.DataFrame(all_rois_data)
    rois_df["cluster_id"] = labels

    # Select from largest cluster (= likely target species in focal recording)
    if n_clusters == 0:
        selected = rois_df.nlargest(min(target_n, len(rois_df)), "power")
    else:
        cluster_sizes = rois_df[rois_df["cluster_id"] >= 0]["cluster_id"].value_counts()
        largest = cluster_sizes.index[0]
        candidates = rois_df[rois_df["cluster_id"] == largest].copy()
        if len(candidates) >= target_n:
            selected = candidates.nlargest(target_n, "power")
        else:
            remaining = target_n - len(candidates)
            others = rois_df[
                (rois_df["cluster_id"] >= 0) & (rois_df["cluster_id"] != largest)
            ]
            supplement = others.nlargest(min(remaining, len(others)), "power")
            selected = pd.concat([candidates, supplement])

    # Recording diversity
    final = []
    rec_counts = {}
    for _, r in selected.sort_values("power", ascending=False).iterrows():
        rid = r["recording_id"]
        rec_counts.setdefault(rid, 0)
        if rec_counts[rid] < max_per_rec:
            final.append(r)
            rec_counts[rid] += 1

    selected = pd.DataFrame(final)
    selected["method"] = "bambird"
    return selected


def run_bambird(args):
    """Run bambird-style pipeline on all species."""
    df = load_metadata(args.species)
    species_codes = df["ebird_species_code"].unique()
    if args.limit:
        species_codes = species_codes[: args.limit]

    all_selected = []
    for species in sorted(species_codes):
        df_sp = df[df["ebird_species_code"] == species]
        _flush_print(f"Processing {species} ({len(df_sp)} recordings)...")
        selected = _bambird_process_species(species, df_sp)
        if len(selected) > 0:
            all_selected.append(selected)
            _export_segments(selected, df, BAMBIRD_DIR / species)

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(STEP_DIR / "bambird_results.csv", index=False)
        _flush_print(f"\nbambird: {len(result)} segments -> bambird_results.csv")


# ===========================================================================
# Method 2: Biodenoising + amplitude selection
# ===========================================================================


def _biodenoising_process_species(
    species, df_species, model, model_sr, device, target_n=75, max_per_rec=5
):
    """Process one species with biodenoising."""
    import torch

    all_notes = []

    for _, row in df_species.iterrows():
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")
        audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name
        if not audio_path.exists():
            continue

        notes = _get_bout_notes(species, safe_id)
        if not notes:
            continue

        # Load audio at model sample rate (16kHz)
        try:
            y_orig, _ = librosa.load(str(audio_path), sr=model_sr)
        except Exception:
            continue

        # Denoise
        try:
            with torch.no_grad():
                noisy = torch.tensor(
                    y_orig, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(device)
                estimate = model(noisy)
                y_denoised = estimate.squeeze().cpu().numpy()
        except RuntimeError:
            # CUDA error fallback to CPU
            if device != "cpu":
                with torch.no_grad():
                    noisy = torch.tensor(
                        y_orig, dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)
                    model_cpu = model.cpu()
                    estimate = model_cpu(noisy)
                    y_denoised = estimate.squeeze().numpy()
                    model.to(device)
            else:
                continue

        for ni, (onset, offset) in enumerate(notes):
            dur = offset - onset
            if dur < 0.05:
                continue

            start_dn = int(onset * model_sr)
            end_dn = int(offset * model_sr)
            note_denoised = y_denoised[start_dn:end_dn]
            note_orig = y_orig[start_dn:end_dn]

            if len(note_denoised) == 0 or len(note_orig) == 0:
                continue

            rms_denoised = float(np.sqrt(np.mean(note_denoised ** 2)) + 1e-10)
            rms_orig = float(np.sqrt(np.mean(note_orig ** 2)) + 1e-10)
            snr_improvement = 20 * np.log10(rms_denoised / rms_orig)

            all_notes.append({
                "species_code": species,
                "recording_id": rec_id,
                "safe_id": safe_id,
                "note_idx": ni,
                "onset": round(onset, 4),
                "offset": round(offset, 4),
                "rms_denoised": rms_denoised,
                "rms_original": rms_orig,
                "snr_improvement": round(snr_improvement, 2),
                "duration": round(dur, 4),
            })

    if not all_notes:
        return pd.DataFrame()

    notes_df = pd.DataFrame(all_notes)
    notes_df["rms_rank"] = notes_df.groupby("recording_id")["rms_denoised"].rank(
        pct=True
    )

    selected = notes_df.nlargest(min(target_n * 2, len(notes_df)), "rms_denoised")

    final = []
    rec_counts = {}
    for _, r in selected.sort_values("rms_denoised", ascending=False).iterrows():
        rid = r["recording_id"]
        rec_counts.setdefault(rid, 0)
        if rec_counts[rid] < max_per_rec:
            final.append(r)
            rec_counts[rid] += 1
        if len(final) >= target_n:
            break

    selected = pd.DataFrame(final)
    selected["method"] = "biodenoising"
    return selected


def run_biodenoising(args):
    """Run biodenoising pipeline on all species."""
    import torch
    from biodenoising.denoiser.pretrained import get_model

    class ModelArgs:
        biodenoising16k_dns48 = True
        dns48 = False
        dns64 = False
        master64 = False
        model_path = ""
        method = "biodenoising16k_dns48"

    # Try CUDA, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(ModelArgs()).to(device)
    model.eval()
    model_sr = model.sample_rate

    df = load_metadata(args.species)
    species_codes = df["ebird_species_code"].unique()
    if args.limit:
        species_codes = species_codes[: args.limit]

    all_selected = []
    for species in sorted(species_codes):
        df_sp = df[df["ebird_species_code"] == species]
        _flush_print(f"Processing {species} ({len(df_sp)} recordings)...")
        try:
            selected = _biodenoising_process_species(
                species, df_sp, model, model_sr, device
            )
        except Exception as e:
            _flush_print(f"  {species}: ERROR {e}")
            # Try CPU fallback for remaining species
            if device != "cpu":
                _flush_print("  Falling back to CPU...")
                device = "cpu"
                model = model.cpu()
                try:
                    selected = _biodenoising_process_species(
                        species, df_sp, model, model_sr, device
                    )
                except Exception as e2:
                    _flush_print(f"  {species}: CPU also failed: {e2}")
                    continue
            else:
                continue

        if len(selected) > 0:
            all_selected.append(selected)
            _export_segments(selected, df, BIODENOISING_DIR / species)
            _flush_print(
                f"  {species}: selected {len(selected)} notes "
                f"from {selected['recording_id'].nunique()} recordings"
            )

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(STEP_DIR / "biodenoising_results.csv", index=False)
        _flush_print(
            f"\nbiodenoising: {len(result)} segments -> biodenoising_results.csv"
        )


# ===========================================================================
# Method 3: Bird-MixIT source separation
# ===========================================================================


def run_birdmixit(args):
    """Run Bird-MixIT pipeline on all species.

    Loads model once and processes all species sequentially.
    """
    import tensorflow as tf

    ckpt_path = MIXIT_MODEL_DIR / "model.ckpt-3223090"
    meta_path = str(MIXIT_MODEL_DIR / "inference.meta")
    if not Path(meta_path).exists():
        _flush_print(f"ERROR: Bird-MixIT model not found at {MIXIT_MODEL_DIR}")
        return

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.compat.v1.disable_v2_behavior()

    # Load model once
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(meta_path)

    config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    sess = tf.compat.v1.Session(graph=graph, config=config)
    saver.restore(sess, str(ckpt_path))

    input_tensor = graph.get_tensor_by_name("input_audio/receiver_audio:0")
    output_tensor = graph.get_tensor_by_name("denoised_waveforms:0")

    df = load_metadata(args.species)
    species_codes = df["ebird_species_code"].unique()
    if args.limit:
        species_codes = species_codes[: args.limit]

    all_selected = []
    for species in sorted(species_codes):
        df_sp = df[df["ebird_species_code"] == species]
        _flush_print(f"Processing {species} ({len(df_sp)} recordings)...")

        all_notes = []
        for _, row in df_sp.iterrows():
            rec_id = row["recording_id"]
            safe_id = rec_id.replace(":", "_")
            audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name
            if not audio_path.exists():
                continue

            notes = _get_bout_notes(species, safe_id)
            if not notes:
                continue

            try:
                y_mixit, _ = librosa.load(str(audio_path), sr=MIXIT_SR, mono=True)
            except Exception:
                continue

            try:
                feed = {
                    input_tensor: y_mixit[np.newaxis, np.newaxis, :].astype(
                        np.float32
                    )
                }
                raw_output = sess.run(output_tensor, feed_dict=feed)
                sources = raw_output[0]  # (n_sources, samples)
            except Exception as e:
                _flush_print(f"    WARN: {safe_id}: {e}")
                continue

            n_sources = sources.shape[0]

            for ni, (onset, offset) in enumerate(notes):
                dur = offset - onset
                if dur < 0.05:
                    continue

                start_mx = int(onset * MIXIT_SR)
                end_mx = int(offset * MIXIT_SR)

                energies = []
                for src_idx in range(n_sources):
                    src_note = sources[src_idx, start_mx:end_mx]
                    if len(src_note) == 0:
                        energies.append(0.0)
                    else:
                        energies.append(float(np.sum(src_note ** 2)))

                total_energy = sum(energies) + 1e-10
                best_src = int(np.argmax(energies))
                purity = energies[best_src] / total_energy

                src_note = sources[best_src, start_mx:end_mx]
                rms_best = float(np.sqrt(np.mean(src_note ** 2)) + 1e-10)

                all_notes.append({
                    "species_code": species,
                    "recording_id": rec_id,
                    "safe_id": safe_id,
                    "note_idx": ni,
                    "onset": round(onset, 4),
                    "offset": round(offset, 4),
                    "best_source": best_src,
                    "purity": round(purity, 4),
                    "rms_best_source": rms_best,
                    "duration": round(dur, 4),
                })

        if not all_notes:
            continue

        notes_df = pd.DataFrame(all_notes)
        notes_df["rms_rank"] = notes_df.groupby("recording_id")[
            "rms_best_source"
        ].rank(pct=True)
        notes_df["score"] = notes_df["purity"] * notes_df["rms_rank"]

        target_n = 75
        max_per_rec = 5
        selected = notes_df.nlargest(min(target_n * 2, len(notes_df)), "score")

        final = []
        rec_counts = {}
        for _, r in selected.sort_values("score", ascending=False).iterrows():
            rid = r["recording_id"]
            rec_counts.setdefault(rid, 0)
            if rec_counts[rid] < max_per_rec:
                final.append(r)
                rec_counts[rid] += 1
            if len(final) >= target_n:
                break

        selected = pd.DataFrame(final)
        selected["method"] = "bird-mixit"

        if len(selected) > 0:
            all_selected.append(selected)
            _export_segments(selected, df, BIRDMIXIT_DIR / species)
            _flush_print(
                f"  {species}: selected {len(selected)} notes "
                f"from {selected['recording_id'].nunique()} recordings"
            )

    sess.close()

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(STEP_DIR / "birdmixit_results.csv", index=False)
        _flush_print(
            f"\nbird-mixit: {len(result)} segments -> birdmixit_results.csv"
        )


# ===========================================================================
# Audio export (shared)
# ===========================================================================


def _export_segments(selected_df, samples_df, out_dir):
    """Export selected segments as WAV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    for (species, rec_id), group in selected_df.groupby(
        ["species_code", "recording_id"]
    ):
        safe_id = rec_id.replace(":", "_")
        rec_row = samples_df[samples_df["recording_id"] == rec_id]
        if rec_row.empty:
            continue

        audio_path = (
            TEST_SAMPLES_DIR / species / Path(rec_row.iloc[0]["file_path"]).name
        )
        if not audio_path.exists():
            continue

        try:
            y, sr = librosa.load(str(audio_path), sr=None)
        except Exception:
            continue

        for _, seg in group.iterrows():
            onset = seg["onset"]
            offset = seg["offset"]
            start = int(onset * sr)
            end = int(offset * sr)
            y_seg = y[start:end]

            if len(y_seg) == 0:
                continue

            idx = seg.get("note_idx", exported)
            out_path = out_dir / f"{safe_id}_n{int(idx):03d}.wav"
            try:
                sf.write(str(out_path), y_seg, sr)
                exported += 1
            except Exception:
                pass

    return exported


# ===========================================================================
# Comparison summary
# ===========================================================================


def run_compare(args):
    """Generate comparison summary across all three methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {
        "bambird": STEP_DIR / "bambird_results.csv",
        "biodenoising": STEP_DIR / "biodenoising_results.csv",
        "bird-mixit": STEP_DIR / "birdmixit_results.csv",
    }

    summaries = {}
    for name, path in methods.items():
        if path.exists():
            df_m = pd.read_csv(path)
            summaries[name] = {
                "total_segments": len(df_m),
                "n_species": df_m["species_code"].nunique(),
                "n_recordings": df_m["recording_id"].nunique(),
                "avg_per_species": len(df_m)
                / max(df_m["species_code"].nunique(), 1),
                "avg_duration": (df_m["offset"] - df_m["onset"]).mean()
                if "offset" in df_m.columns
                else 0,
                "df": df_m,
            }
            _flush_print(f"\n{name}:")
            _flush_print(f"  Total segments: {len(df_m)}")
            _flush_print(f"  Species: {df_m['species_code'].nunique()}")
            _flush_print(f"  Recordings used: {df_m['recording_id'].nunique()}")
            _flush_print(
                f"  Avg per species: "
                f"{len(df_m) / max(df_m['species_code'].nunique(), 1):.1f}"
            )
        else:
            _flush_print(f"\n{name}: no results (run '{name}' subcommand first)")

    if len(summaries) < 2:
        _flush_print("\nNeed at least 2 methods to compare.")
        return

    # Species overlap
    species_sets = {
        name: set(info["df"]["species_code"].unique())
        for name, info in summaries.items()
    }
    _flush_print("\n--- Species Coverage ---")
    for name, sset in species_sets.items():
        _flush_print(f"  {name}: {len(sset)} species")

    if len(species_sets) >= 2:
        names = list(species_sets.keys())
        common = species_sets[names[0]]
        for n in names[1:]:
            common = common & species_sets[n]
        _flush_print(f"  Common to all: {len(common)} species")

    # Per-species comparison chart
    fig, ax = plt.subplots(figsize=(14, 6))
    all_species = sorted(
        set().union(
            *[info["df"]["species_code"].unique() for info in summaries.values()]
        )
    )
    x = np.arange(len(all_species))
    width = 0.8 / len(summaries)

    for i, (name, info) in enumerate(summaries.items()):
        counts = info["df"].groupby("species_code").size()
        values = [counts.get(sp, 0) for sp in all_species]
        ax.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax.set_xlabel("Species")
    ax.set_ylabel("Selected segments")
    ax.set_title("Method Comparison: Selected Segments per Species")
    ax.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax.set_xticklabels(all_species, rotation=90, fontsize=6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STEP_DIR / "method_comparison.png", dpi=150)
    plt.close(fig)
    _flush_print(f"\nComparison chart -> {STEP_DIR / 'method_comparison.png'}")


# ===========================================================================
# CLI
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare segment extraction methods."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bam = sub.add_parser("bambird", help="ROI + spectral features + DBSCAN")
    p_bam.add_argument("--species", type=str, default=None)
    p_bam.add_argument("--limit", type=int, default=None)

    p_bio = sub.add_parser(
        "biodenoising", help="Neural denoising + amplitude selection"
    )
    p_bio.add_argument("--species", type=str, default=None)
    p_bio.add_argument("--limit", type=int, default=None)

    p_mix = sub.add_parser(
        "bird-mixit", help="Source separation + channel selection"
    )
    p_mix.add_argument("--species", type=str, default=None)
    p_mix.add_argument("--limit", type=int, default=None)

    sub.add_parser("compare", help="Generate comparison summary")

    args = parser.parse_args()

    if args.command == "bambird":
        run_bambird(args)
    elif args.command == "biodenoising":
        run_biodenoising(args)
    elif args.command == "bird-mixit":
        run_birdmixit(args)
    elif args.command == "compare":
        run_compare(args)


if __name__ == "__main__":
    main()
