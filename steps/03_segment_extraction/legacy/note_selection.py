"""
Note-level feature extraction, clustering, and selection pipeline.

Uses lightweight spectral features (MFCC, spectral stats) instead of
heavy neural embeddings (Perch v2) for cleaner, noise-aware selection
of individual bird vocalization notes.

Subcommands:
  extract-features - Extract spectral features + quality indicators per note
  cluster          - HDBSCAN clustering per species on note features
  select-notes     - Select clean representative notes with quality scoring
  visualize        - t-SNE visualization of note clusters

Usage:
  python note_selection.py extract-features [--species CODE]
  python note_selection.py cluster [--species CODE]
  python note_selection.py select-notes [--target-n 75] [--max-per-recording 5]
  python note_selection.py visualize [--species CODE]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"
BOUTS_DIR = NAS_BASE / "segments" / "test_samples_results_bouts"
TWEETYNET_DIR = NAS_BASE / "segments" / "test_samples_results_tweetynet"
NOTE_FEATURES_DIR = NAS_BASE / "segments" / "note_features"
SELECTED_NOTES_DIR = NAS_BASE / "segments" / "test_samples_selected_notes"

# Audio / spectrogram parameters (must match prototype_tweetynet.py)
SR = 32000
HOP_LENGTH = 320
N_MELS = 128
N_FFT = 1024
FMIN = 150
FMAX = 12000
FRAME_DUR = HOP_LENGTH / SR  # ~0.01s

# Note filtering
MIN_NOTE_DUR = 0.06  # minimum note duration for clustering


# ===========================================================================
# Feature extraction
# ===========================================================================


def _spectral_entropy(S: np.ndarray) -> float:
    """Compute mean spectral entropy across time frames.

    Args:
        S: Power spectrogram (n_freq, n_frames).

    Returns:
        Mean entropy value (higher = more noise-like).
    """
    # Normalize each frame to a probability distribution
    S_sum = S.sum(axis=0, keepdims=True)
    S_sum = np.where(S_sum < 1e-10, 1e-10, S_sum)
    P = S / S_sum
    P = np.where(P < 1e-10, 1e-10, P)
    entropy = -np.sum(P * np.log2(P), axis=0)
    return float(np.mean(entropy))


def extract_note_features(
    y_full: np.ndarray,
    onset: float,
    offset: float,
    probs: np.ndarray | None = None,
) -> dict:
    """Extract spectral features and quality indicators for a single note.

    Args:
        y_full: Full recording audio at SR.
        onset: Note onset in seconds.
        offset: Note offset in seconds.
        probs: TweetyNet frame-level P(bird) for the full recording.

    Returns:
        Dict with 'features' (31-dim array for clustering) and
        'quality' (4-dim array: snr, pbird, flatness, entropy).
    """
    dur = offset - onset
    start = int(onset * SR)
    end = int(offset * SR)
    y_note = y_full[start:end]

    if len(y_note) < N_FFT:
        # Pad very short notes to minimum FFT size
        y_note = np.pad(y_note, (0, N_FFT - len(y_note)))

    # --- Clustering features (31 dims) ---

    # Mel spectrogram for the note
    S = librosa.feature.melspectrogram(
        y=y_note, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    S_db = librosa.power_to_db(S + 1e-10, ref=np.max)

    # MFCC: 13 coefficients, mean + std = 26 dims
    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # (13,)
    mfcc_std = np.std(mfcc, axis=1)    # (13,)

    # Spectral centroid, bandwidth, rolloff: mean each = 3 dims
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SR)

    # Spectral flatness: mean = 1 dim
    flatness = librosa.feature.spectral_flatness(S=S)
    flatness_mean = float(np.mean(flatness))

    # log(duration) = 1 dim
    log_dur = float(np.log(max(dur, 0.01)))

    features = np.concatenate([
        mfcc_mean,                           # 13
        mfcc_std,                            # 13
        [np.mean(cent)],                     # 1
        [np.mean(bw)],                       # 1
        [np.mean(rolloff)],                  # 1
        [flatness_mean],                     # 1
        [log_dur],                           # 1
    ]).astype(np.float32)                    # total: 31

    # --- Quality indicators (4 dims) ---

    # SNR estimate: note RMS vs surrounding silence RMS
    note_rms = float(np.sqrt(np.mean(y_note ** 2)) + 1e-10)
    ctx_ms = 0.1  # 100ms context
    ctx_samples = int(ctx_ms * SR)
    before_start = max(0, start - ctx_samples)
    after_end = min(len(y_full), end + ctx_samples)
    y_before = y_full[before_start:start]
    y_after = y_full[end:after_end]
    y_context = np.concatenate([y_before, y_after]) if len(y_before) + len(y_after) > 0 else np.zeros(1)
    ctx_rms = float(np.sqrt(np.mean(y_context ** 2)) + 1e-10)
    snr_db = float(20.0 * np.log10(note_rms / ctx_rms))

    # P(bird) mean from TweetyNet predictions
    if probs is not None:
        frame_start = int(onset / FRAME_DUR)
        frame_end = int(offset / FRAME_DUR)
        frame_start = max(0, min(frame_start, len(probs) - 1))
        frame_end = max(frame_start + 1, min(frame_end, len(probs)))
        pbird_mean = float(np.mean(probs[frame_start:frame_end]))
    else:
        pbird_mean = 0.5

    # Spectral entropy
    S_power = librosa.feature.melspectrogram(
        y=y_note, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=64, fmin=FMIN, fmax=FMAX,
    )
    entropy = _spectral_entropy(S_power)

    quality = np.array([snr_db, pbird_mean, flatness_mean, entropy], dtype=np.float32)

    return {"features": features, "quality": quality}


def extract_features_command(args) -> None:
    """Extract spectral features for all notes in bout JSONs."""
    df = load_metadata(args.species)
    print(f"Extracting note features: {len(df)} recordings")

    total_notes = 0
    total_skipped = 0

    for _, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")

        bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
        if not bout_path.exists():
            continue

        out_path = NOTE_FEATURES_DIR / species / f"{safe_id}_note_features.npz"
        if out_path.exists():
            continue

        with open(bout_path) as f:
            bout_data = json.load(f)

        bouts = bout_data["bouts"]
        if not bouts:
            continue

        # Load audio once
        audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name
        if not audio_path.exists():
            continue
        y_full, _ = librosa.load(str(audio_path), sr=SR)

        # Load TweetyNet predictions
        pred_path = TWEETYNET_DIR / species / f"{safe_id}_pred.npz"
        probs = np.load(str(pred_path))["probs"] if pred_path.exists() else None

        features_list = []
        quality_list = []
        bout_idx_list = []
        note_idx_list = []
        onset_list = []
        offset_list = []

        for bi, bout in enumerate(bouts):
            for ni, (note_on, note_off) in enumerate(bout["notes"]):
                dur = note_off - note_on
                if dur < 0.03:  # absolute minimum
                    total_skipped += 1
                    continue

                try:
                    result = extract_note_features(y_full, note_on, note_off, probs)
                    features_list.append(result["features"])
                    quality_list.append(result["quality"])
                    bout_idx_list.append(bi)
                    note_idx_list.append(ni)
                    onset_list.append(note_on)
                    offset_list.append(note_off)
                except Exception as e:
                    print(f"    WARN: {species}/{safe_id} bout{bi} note{ni}: {e}")
                    total_skipped += 1

        if not features_list:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            features=np.stack(features_list),
            quality=np.stack(quality_list),
            bout_indices=np.array(bout_idx_list, dtype=np.int32),
            note_indices=np.array(note_idx_list, dtype=np.int32),
            onsets=np.array(onset_list, dtype=np.float32),
            offsets=np.array(offset_list, dtype=np.float32),
        )

        n = len(features_list)
        total_notes += n
        print(f"  {species}/{safe_id}: {n} notes extracted")

    print(f"\nTotal: {total_notes} notes extracted, {total_skipped} skipped")


# ===========================================================================
# Quality scoring
# ===========================================================================


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    """Robust normalization using median and IQR, clipped to [0, 1]."""
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-8:
        return np.full_like(values, 0.5)
    normalized = (values - med) / (iqr * 1.5)  # ~[-1, 1] for middle 50%
    return np.clip(normalized * 0.5 + 0.5, 0.0, 1.0)


def compute_quality_scores(quality_array: np.ndarray) -> np.ndarray:
    """Compute quality score Q for each note.

    Args:
        quality_array: (n_notes, 4) array of [snr, pbird, flatness, entropy].

    Returns:
        (n_notes,) quality scores in [0, 1].
    """
    snr = _robust_normalize(quality_array[:, 0])
    pbird = _robust_normalize(quality_array[:, 1])
    flatness = _robust_normalize(quality_array[:, 2])
    entropy = _robust_normalize(quality_array[:, 3])

    # Q = 0.4*SNR + 0.2*(1-flatness) + 0.2*(1-entropy) + 0.2*Pbird
    Q = 0.4 * snr + 0.2 * (1.0 - flatness) + 0.2 * (1.0 - entropy) + 0.2 * pbird
    return Q


# ===========================================================================
# Clustering
# ===========================================================================


def cluster_notes(args) -> None:
    """Per-species HDBSCAN clustering on note spectral features."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(SAMPLES_CSV)
    species_codes = df["ebird_species_code"].unique()
    if args.species:
        species_codes = [args.species]

    cluster_summary = []

    for species in sorted(species_codes):
        feat_dir = NOTE_FEATURES_DIR / species
        if not feat_dir.exists():
            continue

        # Collect all note features for this species
        all_features = []
        all_quality = []
        all_meta = []  # (safe_id, bout_idx, note_idx, onset, offset)

        for npz_path in sorted(feat_dir.glob("*_note_features.npz")):
            safe_id = npz_path.stem.replace("_note_features", "")
            data = np.load(str(npz_path))
            feats = data["features"]
            quals = data["quality"]
            bout_idxs = data["bout_indices"]
            note_idxs = data["note_indices"]
            onsets = data["onsets"]
            offsets = data["offsets"]
            data.close()

            for i in range(len(feats)):
                dur = offsets[i] - onsets[i]
                if dur < MIN_NOTE_DUR:
                    continue
                all_features.append(feats[i])
                all_quality.append(quals[i])
                all_meta.append((
                    safe_id,
                    int(bout_idxs[i]),
                    int(note_idxs[i]),
                    float(onsets[i]),
                    float(offsets[i]),
                ))

        if len(all_features) < 10:
            print(f"  {species}: too few notes ({len(all_features)}), skipping")
            continue

        X = np.stack(all_features)
        Q_all = np.stack(all_quality)

        # Compute quality scores for this species
        quality_scores = compute_quality_scores(Q_all)

        # Filter out low-quality notes (bottom 20%)
        q_threshold = np.percentile(quality_scores, 20)
        keep_mask = quality_scores >= q_threshold
        X_filtered = X[keep_mask]
        quality_filtered = quality_scores[keep_mask]
        meta_filtered = [m for m, k in zip(all_meta, keep_mask) if k]

        if len(X_filtered) < 10:
            print(f"  {species}: too few notes after filtering ({len(X_filtered)})")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)

        # Replace any NaN/inf with 0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # PCA dimensionality reduction
        n = len(X_scaled)
        n_pca = min(20, n - 1, X_scaled.shape[1])
        if n_pca < X_scaled.shape[1]:
            pca = PCA(n_components=n_pca, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            explained = pca.explained_variance_ratio_.sum()
            print(f"    PCA: {X_scaled.shape[1]} -> {n_pca} dims ({explained:.1%} variance)")
        else:
            X_pca = X_scaled

        # HDBSCAN
        min_cluster_size = max(8, min(40, round(0.02 * n)))
        min_samples = max(3, min(20, round(0.5 * min_cluster_size)))

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
        )
        cluster_labels = clusterer.fit_predict(X_pca)
        probabilities = clusterer.probabilities_

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = int((cluster_labels == -1).sum())
        noise_ratio = n_noise / n if n > 0 else 0.0

        print(
            f"  {species}: {n} notes -> {n_clusters} clusters, "
            f"{n_noise} noise ({noise_ratio:.1%})"
        )

        cluster_summary.append({
            "species_code": species,
            "n_notes_total": len(all_features),
            "n_notes_filtered": n,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": round(noise_ratio, 3),
        })

        # Write cluster info back to bout JSONs
        updates: dict[str, list[tuple[int, int, int, float, float]]] = {}
        for i, (safe_id, bout_idx, note_idx, onset, offset) in enumerate(meta_filtered):
            if safe_id not in updates:
                updates[safe_id] = []
            updates[safe_id].append((
                bout_idx, note_idx,
                int(cluster_labels[i]),
                round(float(probabilities[i]), 4),
                round(float(quality_filtered[i]), 4),
            ))

        for safe_id, note_updates in updates.items():
            bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
            if not bout_path.exists():
                continue
            with open(bout_path) as f:
                bout_data = json.load(f)

            # Initialize note_clusters dict for each bout if not present
            for bout in bout_data["bouts"]:
                if "note_clusters" not in bout:
                    bout["note_clusters"] = {}

            for bout_idx, note_idx, cl, prob, qscore in note_updates:
                if bout_idx < len(bout_data["bouts"]):
                    bout_data["bouts"][bout_idx]["note_clusters"][str(note_idx)] = {
                        "cluster_id": cl,
                        "cluster_prob": prob,
                        "quality_score": qscore,
                    }

            with open(bout_path, "w") as f:
                json.dump(bout_data, f, indent=2)

    if cluster_summary:
        summary_df = pd.DataFrame(cluster_summary)
        summary_df.to_csv(STEP_DIR / "note_cluster_results.csv", index=False)
        print(f"\nCluster summary -> {STEP_DIR / 'note_cluster_results.csv'}")


# ===========================================================================
# Note selection
# ===========================================================================


def select_notes(args) -> None:
    """Select high-quality notes for the training set.

    Strategy:
    - Remove noise points (cluster_id == -1), supplement if needed
    - Proportional allocation across clusters
    - Within cluster: rank by quality_score
    - Recording diversity: max N notes per recording_id
    """
    df = pd.read_csv(SAMPLES_CSV)
    species_codes = df["ebird_species_code"].unique()
    if args.species:
        species_codes = [args.species]

    target_n = args.target_n
    max_per_recording = args.max_per_recording
    all_selected = []

    for species in sorted(species_codes):
        bout_dir = BOUTS_DIR / species
        if not bout_dir.exists():
            continue

        # Collect all notes with cluster info
        species_notes = []

        for json_path in sorted(bout_dir.glob("*_bouts.json")):
            with open(json_path) as f:
                bout_data = json.load(f)

            rec_id = bout_data["recording_id"]
            safe_id = rec_id.replace(":", "_")

            for bi, bout in enumerate(bout_data["bouts"]):
                note_clusters = bout.get("note_clusters", {})
                for ni, (note_on, note_off) in enumerate(bout["notes"]):
                    nc = note_clusters.get(str(ni))
                    if nc is None:
                        continue  # not clustered (filtered out)
                    species_notes.append({
                        "species_code": species,
                        "recording_id": rec_id,
                        "safe_id": safe_id,
                        "bout_idx": bi,
                        "note_idx": ni,
                        "note_onset": note_on,
                        "note_offset": note_off,
                        "cluster_id": nc["cluster_id"],
                        "cluster_prob": nc["cluster_prob"],
                        "quality_score": nc["quality_score"],
                    })

        if not species_notes:
            continue

        notes_df = pd.DataFrame(species_notes)

        # Separate noise and non-noise
        non_noise = notes_df[notes_df["cluster_id"] >= 0].copy()
        noise = notes_df[notes_df["cluster_id"] == -1].copy()

        if len(non_noise) == 0:
            # All noise - select by quality score
            selected = noise.nlargest(min(target_n, len(noise)), "quality_score")
        elif len(non_noise) >= target_n:
            # Proportional allocation across clusters
            cluster_counts = non_noise["cluster_id"].value_counts()
            total = cluster_counts.sum()

            selected_parts = []
            remaining = target_n

            for cluster_id in cluster_counts.index:
                cluster_notes_df = non_noise[non_noise["cluster_id"] == cluster_id]
                n_alloc = max(1, round(target_n * len(cluster_notes_df) / total))
                n_alloc = min(n_alloc, remaining, len(cluster_notes_df))

                top = cluster_notes_df.nlargest(n_alloc, "quality_score")
                selected_parts.append(top)
                remaining -= len(top)
                if remaining <= 0:
                    break

            selected = pd.concat(selected_parts)
        else:
            # non-noise < target_n: take all non-noise + supplement from noise
            supplement_n = target_n - len(non_noise)
            noise_supplement = noise.nlargest(
                min(supplement_n, len(noise)), "quality_score"
            )
            selected = pd.concat([non_noise, noise_supplement])

        # Apply recording diversity constraint
        final_selected = []
        rec_counts: dict[str, int] = {}
        for _, note in selected.sort_values(
            "quality_score", ascending=False
        ).iterrows():
            rid = note["recording_id"]
            rec_counts.setdefault(rid, 0)
            if rec_counts[rid] < max_per_recording:
                final_selected.append(note)
                rec_counts[rid] += 1

        selected = pd.DataFrame(final_selected)
        all_selected.append(selected)

        print(
            f"  {species}: selected {len(selected)} notes "
            f"from {selected['recording_id'].nunique()} recordings"
        )

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(STEP_DIR / "selected_notes.csv", index=False)
        print(f"\nSelected {len(result)} total notes -> {STEP_DIR / 'selected_notes.csv'}")

        if not args.skip_export:
            _export_selected_audio(result, df)


def _export_selected_audio(
    selected_df: pd.DataFrame,
    samples_df: pd.DataFrame,
) -> None:
    """Export selected note audio segments to individual WAV files."""
    import soundfile as sf

    SELECTED_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    exported = 0

    # Group by recording for efficient audio loading
    for (species, rec_id), group in selected_df.groupby(
        ["species_code", "recording_id"]
    ):
        safe_id = rec_id.replace(":", "_")
        rec_row = samples_df[samples_df["recording_id"] == rec_id]
        if rec_row.empty:
            continue

        audio_path = TEST_SAMPLES_DIR / species / Path(rec_row.iloc[0]["file_path"]).name
        if not audio_path.exists():
            continue

        y, sr = librosa.load(str(audio_path), sr=None)

        out_dir = SELECTED_NOTES_DIR / species
        out_dir.mkdir(parents=True, exist_ok=True)

        for _, note in group.iterrows():
            onset = note["note_onset"]
            offset = note["note_offset"]
            start = int(onset * sr)
            end = int(offset * sr)
            y_note = y[start:end]

            if len(y_note) == 0:
                continue

            bi = int(note["bout_idx"])
            ni = int(note["note_idx"])
            out_path = out_dir / f"{safe_id}_b{bi:03d}_n{ni:03d}.wav"

            try:
                sf.write(str(out_path), y_note, sr)
                exported += 1
            except Exception as e:
                print(f"    Export error: {e}")

    print(f"  Exported {exported} note audio files to {SELECTED_NOTES_DIR}")


# ===========================================================================
# Visualization
# ===========================================================================


def visualize_notes(args) -> None:
    """t-SNE scatter plot per species, colored by cluster, sized by quality."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(SAMPLES_CSV)
    species_codes = df["ebird_species_code"].unique()
    if args.species:
        species_codes = [args.species]

    vis_dir = STEP_DIR / "note_cluster_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for species in sorted(species_codes):
        feat_dir = NOTE_FEATURES_DIR / species
        if not feat_dir.exists():
            continue

        all_feats = []
        all_cluster_ids = []
        all_quality = []

        # Read cluster info from bout JSONs
        bout_dir = BOUTS_DIR / species
        if not bout_dir.exists():
            continue

        # Collect features and cluster info
        for npz_path in sorted(feat_dir.glob("*_note_features.npz")):
            safe_id = npz_path.stem.replace("_note_features", "")
            data = np.load(str(npz_path))
            feats = data["features"]
            bout_idxs = data["bout_indices"]
            note_idxs = data["note_indices"]
            onsets = data["onsets"]
            offsets = data["offsets"]
            data.close()

            bout_path = bout_dir / f"{safe_id}_bouts.json"
            if not bout_path.exists():
                continue
            with open(bout_path) as f:
                bout_data = json.load(f)

            for i in range(len(feats)):
                dur = offsets[i] - onsets[i]
                if dur < MIN_NOTE_DUR:
                    continue

                bi = int(bout_idxs[i])
                ni = int(note_idxs[i])
                if bi >= len(bout_data["bouts"]):
                    continue

                nc = bout_data["bouts"][bi].get("note_clusters", {}).get(str(ni))
                if nc is None:
                    continue

                all_feats.append(feats[i])
                all_cluster_ids.append(nc["cluster_id"])
                all_quality.append(nc["quality_score"])

        if len(all_feats) < 10:
            continue

        X = np.stack(all_feats)
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        perplexity = min(30, len(X) - 1)
        if perplexity < 5:
            continue
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, max_iter=1000
        )
        X_2d = tsne.fit_transform(X)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        cluster_ids = np.array(all_cluster_ids)
        quality_arr = np.array(all_quality)
        unique_clusters = sorted(set(cluster_ids))
        sizes = 15 + 35 * quality_arr

        cmap = plt.cm.tab10
        for cl in unique_clusters:
            mask = cluster_ids == cl
            if cl == -1:
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c="gray", alpha=0.3, s=sizes[mask],
                    label="noise", marker="x",
                )
            else:
                color = cmap(cl % 10)
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[color], alpha=0.6, s=sizes[mask],
                    label=f"C{cl}",
                )

        n_real = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = int((cluster_ids == -1).sum())
        ax.set_title(
            f"{species} — Note Clusters "
            f"({n_real} clusters, {n_noise} noise / {len(X)} total)"
        )
        if len(unique_clusters) <= 20:
            ax.legend(fontsize=7, loc="best", ncol=2)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        fig.tight_layout()
        fig.savefig(vis_dir / f"{species}_note_clusters.png", dpi=150)
        plt.close(fig)

        print(f"  {species}: visualization saved")

    print(f"\nVisualizations saved to {vis_dir}")


# ===========================================================================
# CLI
# ===========================================================================


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
    parser = argparse.ArgumentParser(
        description="Note-level feature extraction, clustering, and selection."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract-features
    ef = sub.add_parser(
        "extract-features", help="Extract spectral features for each note"
    )
    ef.add_argument("--species", type=str, default=None)

    # cluster
    cl = sub.add_parser(
        "cluster", help="HDBSCAN clustering of note features"
    )
    cl.add_argument("--species", type=str, default=None)

    # select-notes
    sn = sub.add_parser(
        "select-notes", help="Select clean representative notes"
    )
    sn.add_argument("--species", type=str, default=None)
    sn.add_argument("--target-n", type=int, default=75)
    sn.add_argument("--max-per-recording", type=int, default=5)
    sn.add_argument("--skip-export", action="store_true")

    # visualize
    viz = sub.add_parser("visualize", help="t-SNE visualization of note clusters")
    viz.add_argument("--species", type=str, default=None)

    args = parser.parse_args()

    if args.command == "extract-features":
        extract_features_command(args)
    elif args.command == "cluster":
        cluster_notes(args)
    elif args.command == "select-notes":
        select_notes(args)
    elif args.command == "visualize":
        visualize_notes(args)


if __name__ == "__main__":
    main()
