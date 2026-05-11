"""
Bout embedding extraction, HDBSCAN clustering, and selection pipeline.

Uses Google Perch v2 embeddings for bout-level similarity analysis,
clusters bouts per species with HDBSCAN, and selects representative bouts
for training data.

Subcommands:
  extract-embeddings - Extract Perch v2 embeddings for each bout
  cluster           - HDBSCAN clustering per species
  select-bouts      - Select representative bouts for training
  visualize         - t-SNE visualization of clusters

Usage:
  python bout_embedding.py extract-embeddings [--species CODE]
  python bout_embedding.py cluster [--species CODE]
  python bout_embedding.py select-bouts [--target-n 50] [--max-per-recording 5]
  python bout_embedding.py visualize [--species CODE]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
EMBEDDINGS_DIR = NAS_BASE / "segments" / "test_samples_embeddings"
SELECTED_BOUTS_DIR = NAS_BASE / "segments" / "test_samples_selected_bouts"

# Perch v2 parameters
PERCH_SR = 32000
PERCH_WINDOW_S = 5.0  # Perch v2 expects 5 seconds at 32kHz
PERCH_WINDOW_SAMPLES = int(PERCH_SR * PERCH_WINDOW_S)  # 160000
PERCH_MODEL_URL = "https://tfhub.dev/google/bird-vocalization-classifier/2"


# ===========================================================================
# Perch v2 embedding extraction
# ===========================================================================


def _load_perch_model():
    """Load Google Perch v2 model from TensorFlow Hub.

    Returns:
        TF Hub model signature function for inference.
    """
    import tensorflow_hub as hub

    print("Loading Perch v2 model from TF Hub...")
    model = hub.load(PERCH_MODEL_URL)
    infer = model.signatures["serving_default"]
    print("Perch v2 loaded (embedding dim=1280)")
    return infer


def _extract_perch_embedding(
    infer, audio: np.ndarray,
) -> np.ndarray:
    """Extract Perch v2 embedding from audio.

    Args:
        infer: Perch model signature function.
        audio: Audio array at PERCH_SR. Will be padded/trimmed to 5s.

    Returns:
        L2-normalized embedding vector of shape (1280,).
    """
    import tensorflow as tf

    # Ensure exactly PERCH_WINDOW_SAMPLES
    if len(audio) < PERCH_WINDOW_SAMPLES:
        audio = np.pad(audio, (0, PERCH_WINDOW_SAMPLES - len(audio)))
    audio = audio[:PERCH_WINDOW_SAMPLES]

    # Perch expects (1, 160000) float32
    inp = tf.constant(audio[np.newaxis].astype(np.float32))
    result = infer(inputs=inp)
    # output_1 is the embedding (1, 1280)
    emb = result["output_1"].numpy()[0]

    # L2-normalize
    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm
    return emb


def extract_embeddings(args) -> None:
    """Extract Perch v2 embeddings for each bout."""
    infer = _load_perch_model()

    df = load_metadata(args.species)
    print(f"Processing {len(df)} recordings")

    for _, row in df.iterrows():
        species = row["ebird_species_code"]
        rec_id = row["recording_id"]
        safe_id = rec_id.replace(":", "_")

        bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
        if not bout_path.exists():
            continue

        with open(bout_path) as f:
            bout_data = json.load(f)

        bouts = bout_data["bouts"]
        if not bouts:
            continue

        out_path = EMBEDDINGS_DIR / species / f"{safe_id}_embeddings.npz"
        if out_path.exists():
            print(f"  EXISTS: {species}/{safe_id}")
            continue

        audio_path = TEST_SAMPLES_DIR / species / Path(row["file_path"]).name
        if not audio_path.exists():
            continue

        # Load full audio once at Perch's sample rate (32kHz)
        y_full, _ = librosa.load(str(audio_path), sr=PERCH_SR)

        bout_embeddings = []
        bout_indices = []

        for bout_idx, bout in enumerate(bouts):
            onset = bout["bout_onset"]
            offset = bout["bout_offset"]
            dur = offset - onset

            if dur <= PERCH_WINDOW_S:
                # Center bout in 5s window with audio context
                pad_total = PERCH_WINDOW_S - dur
                load_onset = max(0.0, onset - pad_total / 2)
                start_sample = int(load_onset * PERCH_SR)
                end_sample = start_sample + PERCH_WINDOW_SAMPLES
                audio = y_full[start_sample:end_sample]

                emb = _extract_perch_embedding(infer, audio)
            else:
                # Split into 5s chunks, average L2-normalized embeddings
                chunk_embs = []
                for chunk_start in np.arange(onset, offset, PERCH_WINDOW_S):
                    chunk_dur = min(PERCH_WINDOW_S, offset - chunk_start)
                    if chunk_dur < 1.0:  # skip very short tail chunks
                        continue
                    start_sample = int(chunk_start * PERCH_SR)
                    end_sample = start_sample + PERCH_WINDOW_SAMPLES
                    audio = y_full[start_sample:end_sample]

                    e = _extract_perch_embedding(infer, audio)
                    chunk_embs.append(e)

                if not chunk_embs:
                    continue

                emb = np.mean(chunk_embs, axis=0)
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm

            bout_embeddings.append(emb)
            bout_indices.append(bout_idx)

        if not bout_embeddings:
            continue

        emb_array = np.stack(bout_embeddings)
        idx_array = np.array(bout_indices)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path), embeddings=emb_array, bout_indices=idx_array
        )
        print(
            f"  {species}/{safe_id}: "
            f"{len(bout_embeddings)} bout embeddings (dim={emb_array.shape[1]})"
        )


# ===========================================================================
# HDBSCAN clustering
# ===========================================================================


def cluster_bouts(args) -> None:
    """Per-species HDBSCAN clustering on L2-normalized Perch embeddings."""
    from sklearn.cluster import HDBSCAN

    df = pd.read_csv(SAMPLES_CSV)
    species_codes = df["ebird_species_code"].unique()
    if args.species:
        species_codes = [args.species]

    cluster_summary = []

    for species in sorted(species_codes):
        emb_dir = EMBEDDINGS_DIR / species
        if not emb_dir.exists():
            continue

        # Collect all embeddings for this species
        all_embeddings = []
        all_meta = []  # (safe_id, bout_idx)

        for npz_path in sorted(emb_dir.glob("*_embeddings.npz")):
            safe_id = npz_path.stem.replace("_embeddings", "")
            data = np.load(str(npz_path))
            embs = data["embeddings"]
            indices = data["bout_indices"]
            data.close()

            for i in range(len(embs)):
                all_embeddings.append(embs[i])
                all_meta.append((safe_id, int(indices[i])))

        if len(all_embeddings) < 5:
            print(
                f"  {species}: too few bouts ({len(all_embeddings)}), "
                f"skipping clustering"
            )
            continue

        X = np.stack(all_embeddings)
        # L2-normalize (should already be normalized, but ensure)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-8)

        # HDBSCAN parameters (relaxed to avoid over-strict clustering for large n)
        n = len(X)
        min_cluster_size = max(8, min(40, round(0.02 * n)))
        min_samples = max(3, min(20, round(0.5 * min_cluster_size)))

        # PCA dimensionality reduction (1280 -> 50 dims)
        from sklearn.decomposition import PCA

        n_pca = min(50, n - 1, X.shape[1])
        if n_pca < X.shape[1]:
            pca = PCA(n_components=n_pca, random_state=42)
            X_pca = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum()
            print(f"    PCA: {X.shape[1]} -> {n_pca} dims ({explained:.1%} variance)")
        else:
            X_pca = X

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # euclidean on unit sphere ≈ cosine
            cluster_selection_method="leaf",
        )
        cluster_labels = clusterer.fit_predict(X_pca)
        probabilities = clusterer.probabilities_

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = int((cluster_labels == -1).sum())
        noise_ratio = n_noise / n

        print(
            f"  {species}: {n} bouts -> {n_clusters} clusters, "
            f"{n_noise} noise ({noise_ratio:.1%})"
        )

        cluster_summary.append({
            "species_code": species,
            "n_bouts": n,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": round(noise_ratio, 3),
        })

        # Batch updates per file to avoid repeated read/write
        updates: dict[str, dict[int, tuple[int, float]]] = {}
        for i, (safe_id, bout_idx) in enumerate(all_meta):
            if safe_id not in updates:
                updates[safe_id] = {}
            updates[safe_id][bout_idx] = (
                int(cluster_labels[i]),
                round(float(probabilities[i]), 4),
            )

        for safe_id, bout_updates in updates.items():
            bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
            if not bout_path.exists():
                continue
            with open(bout_path) as f:
                bout_data = json.load(f)
            for bout_idx, (cl, prob) in bout_updates.items():
                if bout_idx < len(bout_data["bouts"]):
                    bout_data["bouts"][bout_idx]["cluster_id"] = cl
                    bout_data["bouts"][bout_idx]["cluster_prob"] = prob
            with open(bout_path, "w") as f:
                json.dump(bout_data, f, indent=2)

    if cluster_summary:
        summary_df = pd.DataFrame(cluster_summary)
        summary_df.to_csv(STEP_DIR / "cluster_results.csv", index=False)
        print(f"\nCluster summary saved to {STEP_DIR / 'cluster_results.csv'}")


# ===========================================================================
# Bout selection
# ===========================================================================


def _export_selected_audio(
    selected_df: pd.DataFrame,
    samples_df: pd.DataFrame,
) -> None:
    """Export selected bout audio segments to individual files."""
    SELECTED_BOUTS_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0
    for _, row in selected_df.iterrows():
        species = row["species_code"]
        safe_id = row["safe_id"]
        bout_idx = row["bout_idx"]
        onset = row["bout_onset"]
        offset = row["bout_offset"]

        # Find audio file path from samples CSV
        rec_row = samples_df[samples_df["recording_id"] == row["recording_id"]]
        if rec_row.empty:
            continue

        audio_path = TEST_SAMPLES_DIR / species / Path(rec_row.iloc[0]["file_path"]).name
        if not audio_path.exists():
            continue

        try:
            import soundfile as sf

            y, sr = librosa.load(
                str(audio_path), sr=None, offset=onset, duration=offset - onset
            )
            out_dir = SELECTED_BOUTS_DIR / species
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{safe_id}_bout{bout_idx:03d}.wav"
            sf.write(str(out_path), y, sr)
            exported += 1
        except Exception as e:
            print(f"    Export error: {e}")

    print(f"  Exported {exported} bout audio files to {SELECTED_BOUTS_DIR}")


def select_bouts(args) -> None:
    """Select high-quality bouts for the training set.

    Selection strategy:
    - Remove noise points (HDBSCAN cluster_id == -1)
    - Allocate target_n proportionally across clusters
    - Within each cluster: rank by cluster_prob (closeness to core)
    - Recording diversity: max N bouts per recording_id
    - If non-noise < target_n: supplement from noise by cluster_prob
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

        # Collect all bouts with cluster info
        species_bouts = []

        for json_path in sorted(bout_dir.glob("*_bouts.json")):
            with open(json_path) as f:
                bout_data = json.load(f)

            rec_id = bout_data["recording_id"]
            safe_id = rec_id.replace(":", "_")

            for i, bout in enumerate(bout_data["bouts"]):
                species_bouts.append({
                    "species_code": species,
                    "recording_id": rec_id,
                    "safe_id": safe_id,
                    "bout_idx": i,
                    "bout_onset": bout["bout_onset"],
                    "bout_offset": bout["bout_offset"],
                    "n_notes": bout["n_notes"],
                    "cluster_id": bout.get("cluster_id", -1),
                    "cluster_prob": bout.get("cluster_prob", 0.0),
                })

        if not species_bouts:
            continue

        bouts_df = pd.DataFrame(species_bouts)

        # Separate noise and non-noise points
        non_noise = bouts_df[bouts_df["cluster_id"] >= 0].copy()
        noise = bouts_df[bouts_df["cluster_id"] == -1].copy()

        if len(non_noise) == 0:
            # All noise - select by n_notes (longer bouts more likely valid)
            selected = noise.nlargest(min(target_n, len(noise)), "n_notes")
        elif len(non_noise) >= target_n:
            # Proportional allocation across clusters
            cluster_counts = non_noise["cluster_id"].value_counts()
            total = cluster_counts.sum()

            selected_parts = []
            remaining = target_n

            for cluster_id in cluster_counts.index:
                cluster_bouts_df = non_noise[non_noise["cluster_id"] == cluster_id]
                n_alloc = max(1, round(target_n * len(cluster_bouts_df) / total))
                n_alloc = min(n_alloc, remaining, len(cluster_bouts_df))

                top = cluster_bouts_df.nlargest(n_alloc, "cluster_prob")
                selected_parts.append(top)
                remaining -= len(top)

                if remaining <= 0:
                    break

            selected = pd.concat(selected_parts)
        else:
            # non-noise < target_n: take all non-noise + supplement from noise
            supplement_n = target_n - len(non_noise)
            # Noise points have cluster_prob=0, so sort by n_notes instead
            noise_supplement = noise.nlargest(
                min(supplement_n, len(noise)), "n_notes"
            )
            selected = pd.concat([non_noise, noise_supplement])

        # Apply recording diversity constraint: max N bouts per recording
        final_selected = []
        rec_counts: dict[str, int] = {}
        for _, bout in selected.sort_values(
            "cluster_prob", ascending=False
        ).iterrows():
            rid = bout["recording_id"]
            rec_counts.setdefault(rid, 0)
            if rec_counts[rid] < max_per_recording:
                final_selected.append(bout)
                rec_counts[rid] += 1

        selected = pd.DataFrame(final_selected)
        all_selected.append(selected)

        print(
            f"  {species}: selected {len(selected)} bouts "
            f"from {selected['recording_id'].nunique()} recordings"
        )

    if all_selected:
        result = pd.concat(all_selected, ignore_index=True)
        result.to_csv(STEP_DIR / "selected_bouts.csv", index=False)
        print(f"\nSelected {len(result)} total bouts -> {STEP_DIR / 'selected_bouts.csv'}")

        if not args.skip_export:
            _export_selected_audio(result, df)


# ===========================================================================
# Visualization
# ===========================================================================


def visualize_embeddings(args) -> None:
    """t-SNE scatter plot per species, colored by cluster."""
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(SAMPLES_CSV)
    species_codes = df["ebird_species_code"].unique()
    if args.species:
        species_codes = [args.species]

    vis_dir = STEP_DIR / "cluster_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for species in sorted(species_codes):
        emb_dir = EMBEDDINGS_DIR / species
        if not emb_dir.exists():
            continue

        # Collect embeddings and metadata
        all_embs = []
        all_cluster_ids = []
        all_n_notes = []

        for npz_path in sorted(emb_dir.glob("*_embeddings.npz")):
            safe_id = npz_path.stem.replace("_embeddings", "")
            data = np.load(str(npz_path))
            embs = data["embeddings"]
            indices = data["bout_indices"]
            data.close()

            bout_path = BOUTS_DIR / species / f"{safe_id}_bouts.json"
            if not bout_path.exists():
                continue
            with open(bout_path) as f:
                bout_data = json.load(f)

            for i in range(len(embs)):
                bout_idx = int(indices[i])
                if bout_idx < len(bout_data["bouts"]):
                    bout = bout_data["bouts"][bout_idx]
                    all_embs.append(embs[i])
                    all_cluster_ids.append(bout.get("cluster_id", -1))
                    all_n_notes.append(bout.get("n_notes", 1))

        if len(all_embs) < 5:
            continue

        X = np.stack(all_embs)

        # t-SNE dimensionality reduction
        perplexity = min(30, len(X) - 1)
        if perplexity < 5:
            continue  # too few points for meaningful t-SNE
        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, max_iter=1000
        )
        X_2d = tsne.fit_transform(X)

        # Create figure: colored by cluster, sized by n_notes
        fig, ax = plt.subplots(figsize=(10, 8))

        cluster_ids = np.array(all_cluster_ids)
        unique_clusters = sorted(set(cluster_ids))
        n_notes_arr = np.array(all_n_notes, dtype=float)
        sizes = 20 + 30 * np.clip(n_notes_arr / n_notes_arr.max(), 0, 1)

        cmap = plt.cm.tab10
        for cl in unique_clusters:
            mask = cluster_ids == cl
            if cl == -1:
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c="gray", alpha=0.3, s=sizes[mask], label="noise", marker="x",
                )
            else:
                color = cmap(cl % 10)
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=[color], alpha=0.7, s=sizes[mask], label=f"cluster {cl}",
                )

        n_real_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = int((cluster_ids == -1).sum())
        ax.set_title(
            f"{species} — Perch v2 Embeddings "
            f"({n_real_clusters} clusters, {n_noise} noise / {len(X)} total)"
        )
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        fig.tight_layout()
        fig.savefig(vis_dir / f"{species}_clusters.png", dpi=150)
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
        description="Bout embedding (Perch v2), clustering, and selection pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract-embeddings
    ee = sub.add_parser(
        "extract-embeddings", help="Extract Perch v2 embeddings for bouts"
    )
    ee.add_argument("--species", type=str, default=None, help="Filter by species code")

    # cluster
    cl = sub.add_parser(
        "cluster", help="HDBSCAN clustering of bout embeddings"
    )
    cl.add_argument("--species", type=str, default=None, help="Filter by species code")

    # select-bouts
    sb = sub.add_parser(
        "select-bouts", help="Select high-quality bouts for training"
    )
    sb.add_argument("--species", type=str, default=None, help="Filter by species code")
    sb.add_argument(
        "--target-n", type=int, default=50, help="Target number of bouts per species"
    )
    sb.add_argument(
        "--max-per-recording", type=int, default=5, help="Max bouts per recording"
    )
    sb.add_argument(
        "--skip-export", action="store_true", help="Skip audio export"
    )

    # visualize
    viz = sub.add_parser(
        "visualize", help="Visualize embeddings with t-SNE"
    )
    viz.add_argument(
        "--species", type=str, default=None, help="Filter by species code"
    )

    args = parser.parse_args()

    if args.command == "extract-embeddings":
        extract_embeddings(args)
    elif args.command == "cluster":
        cluster_bouts(args)
    elif args.command == "select-bouts":
        select_bouts(args)
    elif args.command == "visualize":
        visualize_embeddings(args)


if __name__ == "__main__":
    main()
