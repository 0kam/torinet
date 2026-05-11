"""
Cluster selection UI for species prototype building.

For BirdNET-unregistered species, HDBSCAN clustering on Perch v2 embeddings
finds candidate clusters. This web UI lets the user listen to representative
samples from each cluster and select which ones are the target species.

Usage:
    python cluster_selector.py [--port 8053]
"""

import argparse
import io
import sys
from functools import lru_cache
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from jinja2 import Template

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent.parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SEGMENTS_BASE = NAS_BASE / "segments"
SOURCES_DIR = SEGMENTS_BASE / "birdmixit_sources"
ACOUSTIC_FEATURES_DIR = SEGMENTS_BASE / "acoustic_features"
PERCH_EMBEDDINGS_DIR = SEGMENTS_BASE / "perch_embeddings"
PROTOTYPES_DIR = STEP_DIR / "species_prototypes"
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
SPECIES_LIST_CSV = REPO_ROOT / "steps" / "01_species_list" / "species_list.csv"

# Perch v2 parameters
PERCH_SR = 32000
PERCH_WINDOW_S = 5.0

# Source audio sample rate (Bird-MixIT output)
MIXIT_SR = 22050

# Filtering thresholds (same as build-prototypes)
SNR_THRESHOLD = 5.0
NDSI_THRESHOLD = -0.5

# ---------------------------------------------------------------------------
# Species name mapping
# ---------------------------------------------------------------------------


def load_species_names() -> dict:
    """Load ebird_species_code -> {ja, sci} mapping."""
    if not SPECIES_LIST_CSV.exists():
        return {}
    df = pd.read_csv(SPECIES_LIST_CSV, encoding="utf-8-sig")
    mapping = {}
    for _, row in df.iterrows():
        code = row.get("ebird_species_code")
        ja = row.get("japanese_name", "")
        sci = row.get("scientific_name", "")
        if pd.notna(code) and code:
            mapping[code] = {
                "ja": ja if pd.notna(ja) else "",
                "sci": sci if pd.notna(sci) else "",
            }
    return mapping


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def list_species_with_embeddings() -> list[str]:
    """List species codes that have precomputed Perch embeddings."""
    if not PERCH_EMBEDDINGS_DIR.exists():
        return []
    return sorted(
        d.name for d in PERCH_EMBEDDINGS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.npz"))
    )


def load_embeddings_for_species(species: str) -> dict:
    """Load all precomputed embeddings for a species, joined with acoustic features.

    Reads Perch embeddings from perch_embeddings/{species}/ and the matching
    SNR/NDSI/bird_ratio from acoustic_features/{species}/. Windows are paired
    by window_starts — if the two files disagree (file count or window layout),
    the mismatched source is silently skipped.

    Returns dict with keys:
      embeddings: np.ndarray (N, 1280)
      snr: np.ndarray (N,)
      ndsi: np.ndarray (N,)
      bird_ratio: np.ndarray (N,)
      sources: list of (safe_id, ch, window_start_sec) for each embedding
    """
    sp_emb_dir = PERCH_EMBEDDINGS_DIR / species
    sp_acoustic_dir = ACOUSTIC_FEATURES_DIR / species
    empty = {
        "embeddings": np.empty((0, 1280)),
        "snr": np.empty(0),
        "ndsi": np.empty(0),
        "bird_ratio": np.empty(0),
        "sources": [],
    }
    if not sp_emb_dir.exists():
        return empty

    all_embs = []
    all_snr = []
    all_ndsi = []
    all_bird_ratio = []
    all_sources = []

    for emb_path in sorted(sp_emb_dir.glob("*.npz")):
        # Parse filename: {safe_id}_src{ch}.npz
        stem = emb_path.stem
        parts = stem.rsplit("_src", 1)
        if len(parts) != 2:
            continue
        safe_id = parts[0]
        try:
            ch = int(parts[1])
        except ValueError:
            continue

        emb_data = np.load(emb_path)
        embs = emb_data["embeddings"]
        emb_starts = emb_data["window_starts"]

        acoustic_path = sp_acoustic_dir / emb_path.name
        if not acoustic_path.exists():
            continue
        acoustic_data = np.load(acoustic_path)
        snr = acoustic_data["snr"]
        ndsi = acoustic_data["ndsi"]
        bird_ratio = acoustic_data["bird_ratio"]
        ac_starts = acoustic_data["window_starts"]

        # Require matching window layout
        if len(emb_starts) != len(ac_starts) or not np.array_equal(
            emb_starts, ac_starts
        ):
            continue

        for i in range(len(embs)):
            all_embs.append(embs[i])
            all_snr.append(snr[i])
            all_ndsi.append(ndsi[i])
            all_bird_ratio.append(bird_ratio[i])
            win_start_sec = float(emb_starts[i]) / MIXIT_SR
            all_sources.append((safe_id, ch, win_start_sec))

    if not all_embs:
        return empty

    return {
        "embeddings": np.stack(all_embs),
        "snr": np.array(all_snr),
        "ndsi": np.array(all_ndsi),
        "bird_ratio": np.array(all_bird_ratio),
        "sources": all_sources,
    }


def run_clustering(species: str) -> dict | None:
    """Run HDBSCAN clustering for a species.

    Returns dict with:
      clusters: list of cluster dicts (sorted by size, noise last)
      n_total: total embeddings after filtering
      n_raw: total embeddings before filtering
    """
    from sklearn.cluster import HDBSCAN

    data = load_embeddings_for_species(species)
    if len(data["embeddings"]) == 0:
        return None

    n_raw = len(data["embeddings"])

    # Filter: TweetyNet detected bird activity, falling back to SNR+NDSI
    if "bird_ratio" in data and len(data["bird_ratio"]) == len(data["embeddings"]):
        mask = data["bird_ratio"] > 0.0
    else:
        mask = (data["snr"] > SNR_THRESHOLD) & (data["ndsi"] > NDSI_THRESHOLD)
    if mask.sum() < 2:
        return None

    embs = data["embeddings"][mask]
    snr = data["snr"][mask]
    ndsi = data["ndsi"][mask]
    sources = [data["sources"][i] for i in range(n_raw) if mask[i]]

    n = len(embs)

    # L2-normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    X = embs / norms

    # HDBSCAN with adaptive parameters (same as build-prototypes)
    min_cluster_size = max(5, min(30, round(0.03 * n)))
    min_samples = max(3, min(15, round(0.5 * min_cluster_size)))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(X)

    # Build cluster info
    unique_labels = sorted(set(labels))
    clusters = []

    for cl in unique_labels:
        cl_mask = labels == cl
        cl_embs = X[cl_mask]
        cl_snr = snr[cl_mask]
        cl_ndsi = ndsi[cl_mask]
        cl_sources = [sources[i] for i in range(n) if cl_mask[i]]
        cl_indices = np.where(cl_mask)[0]

        # Centroid
        centroid = np.mean(cl_embs, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8

        # Find 5 closest to centroid
        dists = np.linalg.norm(cl_embs - centroid, axis=1)
        n_repr = min(5, len(cl_embs))
        repr_idx = np.argsort(dists)[:n_repr]

        representatives = []
        for ri in repr_idx:
            safe_id, ch, win_start = cl_sources[ri]
            representatives.append({
                "safe_id": safe_id,
                "ch": ch,
                "window_start": win_start,
                "snr": float(cl_snr[ri]),
                "ndsi": float(cl_ndsi[ri]),
                "dist_to_centroid": float(dists[ri]),
            })

        clusters.append({
            "label": int(cl),
            "size": int(cl_mask.sum()),
            "mean_snr": float(np.mean(cl_snr)),
            "mean_ndsi": float(np.mean(cl_ndsi)),
            "std_snr": float(np.std(cl_snr)),
            "centroid": centroid,
            "representatives": representatives,
            "is_noise": cl == -1,
        })

    # Sort: numbered clusters by size (desc), noise cluster at end
    noise_clusters = [c for c in clusters if c["is_noise"]]
    numbered_clusters = sorted(
        [c for c in clusters if not c["is_noise"]],
        key=lambda c: -c["size"]
    )
    clusters = numbered_clusters + noise_clusters

    return {
        "clusters": clusters,
        "n_total": n,
        "n_raw": n_raw,
        "n_filtered_out": n_raw - n,
    }


# ---------------------------------------------------------------------------
# Audio clip extraction
# ---------------------------------------------------------------------------


def extract_audio_clip(species: str, safe_id: str, ch: int,
                       window_start: float) -> bytes | None:
    """Extract a 5-second audio clip from a source WAV file.

    Returns WAV bytes or None if file not found.
    """
    wav_path = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
    if not wav_path.exists():
        return None

    try:
        y, sr = librosa.load(
            str(wav_path), sr=MIXIT_SR, mono=True,
            offset=window_start,
            duration=PERCH_WINDOW_S,
        )
        if len(y) == 0:
            return None

        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2048)
def generate_clip_spectrogram(
    species: str, safe_id: str, ch: int, window_start: float,
    width: float = 3.0, height: float = 1.0,
) -> bytes | None:
    """Generate a mel spectrogram PNG for a 5s audio clip."""
    wav_path = SOURCES_DIR / species / f"{safe_id}_src{ch}.wav"
    if not wav_path.exists():
        return None

    try:
        y, sr = librosa.load(
            str(wav_path), sr=MIXIT_SR, mono=True,
            offset=window_start,
            duration=PERCH_WINDOW_S,
        )
        if len(y) == 0:
            return None

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=11000, hop_length=256,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=100)
        fig.patch.set_facecolor("#0a0a1a")
        ax.set_facecolor("#0a0a1a")
        ax.imshow(
            S_db, aspect="auto", origin="lower",
            cmap="magma", interpolation="antialiased",
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Cluster Selector — ToriNet</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { background: #16213e; color: white; padding: 16px 24px;
            border-bottom: 1px solid #0f3460; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header p { font-size: 13px; color: #8899aa; margin-top: 4px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
  .summary-card { background: #16213e; border-radius: 8px; padding: 16px;
                  border: 1px solid #0f3460; }
  .summary-card h3 { font-size: 13px; color: #8899aa; margin-bottom: 8px; }
  .summary-card .value { font-size: 28px; font-weight: 700; color: #e94560; }
  .summary-card .sub { font-size: 12px; color: #667788; margin-top: 4px; }
  .filter-bar { margin-bottom: 16px; }
  .filter-bar input { padding: 8px 12px; border: 1px solid #0f3460; border-radius: 6px;
                      width: 300px; font-size: 14px; background: #16213e; color: #e0e0e0; }
  .filter-bar input::placeholder { color: #556677; }
  table.species-table { width: 100%; border-collapse: collapse; background: #16213e;
                        border-radius: 8px; overflow: hidden; }
  table.species-table th { text-align: left; padding: 10px 14px; background: #0f3460;
                           font-size: 12px; color: #8899aa; text-transform: uppercase; }
  table.species-table td { padding: 10px 14px; border-bottom: 1px solid #1a1a3e; font-size: 14px; }
  table.species-table tr:hover { background: #1a1a3e; }
  table.species-table a { color: #e94560; text-decoration: none; }
  table.species-table a:hover { text-decoration: underline; }
  .code { font-size: 11px; color: #667788; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
           font-weight: 600; }
  .badge-done { background: #1b4332; color: #52b788; }
  .badge-auto { background: #0f3460; color: #8899aa; }
  .badge-none { background: #2a1a1a; color: #aa5555; }
</style>
</head>
<body>
<div class="header">
  <h1>Cluster Selector</h1>
  <p>Step 03 — Select target species clusters from HDBSCAN results</p>
</div>
<div class="container">
  <div class="summary">
    <div class="summary-card">
      <h3>Species with Embeddings</h3>
      <div class="value">{{ n_species }}</div>
    </div>
    <div class="summary-card">
      <h3>User-Selected Prototypes</h3>
      <div class="value">{{ n_user_selected }}</div>
    </div>
    <div class="summary-card">
      <h3>Remaining</h3>
      <div class="value">{{ n_species - n_user_selected }}</div>
      <div class="sub">species without user-selected prototypes</div>
    </div>
  </div>

  <div class="filter-bar">
    <input type="text" id="filter" placeholder="Filter species..." oninput="filterSpecies()">
  </div>

  <table class="species-table" id="species-table">
    <thead>
      <tr>
        <th>Species</th>
        <th>Code</th>
        <th>Prototype Status</th>
      </tr>
    </thead>
    <tbody>
      {% for sp in species_list %}
      <tr data-code="{{ sp }}" data-ja="{{ species_names.get(sp, {}).get('ja', '') }}"
          data-sci="{{ species_names.get(sp, {}).get('sci', '') }}">
        <td>
          <a href="/cluster/{{ sp }}">{{ species_names.get(sp, {}).get('ja', '') or sp }}</a><br>
          <span class="code"><em>{{ species_names.get(sp, {}).get('sci', '') }}</em></span>
        </td>
        <td class="code">{{ sp }}</td>
        <td>
          {% if proto_status.get(sp) == 'user-selected' %}
          <span class="badge badge-done">User Selected</span>
          {% elif proto_status.get(sp) %}
          <span class="badge badge-auto">{{ proto_status.get(sp) }}</span>
          {% else %}
          <span class="badge badge-none">No Prototype</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
<script>
function filterSpecies() {
  const q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('#species-table tbody tr').forEach(r => {
    const text = (r.dataset.code + ' ' + r.dataset.ja + ' ' + r.dataset.sci).toLowerCase();
    r.style.display = text.includes(q) ? '' : 'none';
  });
}
</script>
</body>
</html>
""")


CLUSTER_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{{ ja_name }} — Cluster Selector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { background: #16213e; color: white; padding: 16px 24px;
            display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #0f3460; }
  .header a { color: #e94560; text-decoration: none; font-size: 14px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header .sub { font-size: 13px; color: #8899aa; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

  .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 14px; border: 1px solid #0f3460; }
  .stat-card h3 { font-size: 12px; color: #8899aa; margin-bottom: 4px; text-transform: uppercase; }
  .stat-card .value { font-size: 22px; font-weight: 700; color: #e0e0e0; }
  .stat-card .sub { font-size: 11px; color: #667788; margin-top: 2px; }

  .loading { text-align: center; padding: 60px 20px; color: #8899aa; font-size: 16px; }
  .spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid #334;
             border-top-color: #e94560; border-radius: 50%; animation: spin 0.8s linear infinite;
             margin-right: 10px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .cluster-card { background: #16213e; border-radius: 8px; margin-bottom: 16px;
                  border: 2px solid #0f3460; overflow: hidden; transition: border-color 0.2s; }
  .cluster-card.selected { border-color: #52b788; }
  .cluster-card.noise { border-color: #553333; opacity: 0.7; }

  .cluster-header { padding: 12px 16px; display: flex; align-items: center; gap: 16px;
                    flex-wrap: wrap; }
  .cluster-header label { display: flex; align-items: center; gap: 8px; cursor: pointer;
                          font-weight: 600; font-size: 15px; }
  .cluster-header input[type="checkbox"] { width: 18px; height: 18px; accent-color: #52b788;
                                           cursor: pointer; }
  .cluster-meta { display: flex; gap: 16px; font-size: 12px; color: #8899aa; }
  .cluster-meta span { white-space: nowrap; }
  .cluster-meta .val { color: #e0e0e0; font-weight: 600; }

  .repr-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;
               padding: 12px 16px 16px 16px; }
  .repr-card { background: #0f1a2e; border-radius: 6px; padding: 8px;
               border: 1px solid #1a2a4e; }
  .repr-card img { width: 100%; height: 70px; border-radius: 4px; display: block;
                   background: #0a0a1a; object-fit: cover; }
  .repr-card audio { width: 100%; height: 28px; margin-top: 4px; }
  .repr-card .repr-info { font-size: 10px; color: #667788; margin-top: 3px; line-height: 1.4; }

  .save-bar { position: sticky; bottom: 0; background: #16213e; border-top: 2px solid #0f3460;
              padding: 14px 24px; display: flex; align-items: center; gap: 16px; z-index: 100; }
  .save-btn { background: #e94560; color: white; border: none; padding: 10px 28px;
              border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; }
  .save-btn:hover { background: #d63851; }
  .save-btn:disabled { background: #444; cursor: not-allowed; }
  .save-status { font-size: 13px; color: #8899aa; }
  .save-status.ok { color: #52b788; font-weight: 600; }
  .save-status.err { color: #e94560; font-weight: 600; }

  @media (max-width: 1000px) {
    .repr-grid { grid-template-columns: repeat(3, 1fr); }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<div class="header">
  <a href="/">&larr; Back</a>
  <div>
    <h1>{{ ja_name }}</h1>
    <div class="sub">{{ species_code }} — <em>{{ sci_name }}</em></div>
  </div>
</div>
<div class="container" id="main-content">
  <div class="loading" id="loading-msg">
    <span class="spinner"></span>Loading embeddings and running HDBSCAN clustering...
  </div>
</div>

<div class="save-bar" id="save-bar" style="display:none;">
  <button class="save-btn" id="save-btn" onclick="saveSelection()">Save Selection as Prototype</button>
  <span class="save-status" id="save-status"></span>
  <span style="margin-left:auto; font-size:12px; color:#667788;" id="selection-summary"></span>
</div>

<script>
const speciesCode = "{{ species_code }}";

// Load cluster data asynchronously
fetch(`/api/clusters/${speciesCode}`)
  .then(r => r.json())
  .then(data => {
    if (data.error) {
      document.getElementById('main-content').innerHTML =
        `<div class="loading" style="color:#e94560;">${data.error}</div>`;
      return;
    }
    renderClusters(data);
  })
  .catch(err => {
    document.getElementById('main-content').innerHTML =
      `<div class="loading" style="color:#e94560;">Error: ${err.message}</div>`;
  });

function renderClusters(data) {
  const container = document.getElementById('main-content');
  let html = '';

  // Stats row
  html += `<div class="stats-row">
    <div class="stat-card">
      <h3>Total Embeddings</h3>
      <div class="value">${data.n_raw}</div>
      <div class="sub">before filtering</div>
    </div>
    <div class="stat-card">
      <h3>After Filtering</h3>
      <div class="value">${data.n_total}</div>
      <div class="sub">SNR &gt; ${data.snr_threshold}, NDSI &gt; ${data.ndsi_threshold}</div>
    </div>
    <div class="stat-card">
      <h3>Clusters Found</h3>
      <div class="value">${data.n_clusters}</div>
      <div class="sub">excluding noise</div>
    </div>
    <div class="stat-card">
      <h3>Noise Points</h3>
      <div class="value">${data.n_noise}</div>
      <div class="sub">HDBSCAN label = -1</div>
    </div>
  </div>`;

  // Previously selected info
  if (data.previous_method) {
    html += `<div style="margin-bottom:16px; padding:10px 14px; background:#1b4332; border-radius:6px; border:1px solid #2d6a4f; font-size:13px;">
      Existing prototype: <strong>${data.previous_method}</strong>
      (${data.previous_n_prototypes} prototype${data.previous_n_prototypes !== 1 ? 's' : ''})
    </div>`;
  }

  // Cluster cards
  data.clusters.forEach((cl, idx) => {
    const isNoise = cl.is_noise;
    const cardClass = isNoise ? 'cluster-card noise' : 'cluster-card';
    const label = isNoise ? 'Noise (label=-1)' : `Cluster ${cl.label}`;
    const pctOfTotal = ((cl.size / data.n_total) * 100).toFixed(1);

    html += `<div class="${cardClass}" id="cluster-${cl.label}">
      <div class="cluster-header">
        <label>
          <input type="checkbox" name="cluster" value="${cl.label}"
                 onchange="updateSelection()" ${isNoise ? '' : ''}>
          ${label}
        </label>
        <div class="cluster-meta">
          <span>Size: <span class="val">${cl.size}</span> (${pctOfTotal}%)</span>
          <span>Mean SNR: <span class="val">${cl.mean_snr.toFixed(1)} dB</span></span>
          <span>Mean NDSI: <span class="val">${cl.mean_ndsi.toFixed(2)}</span></span>
        </div>
      </div>
      <div class="repr-grid">`;

    cl.representatives.forEach((r, ri) => {
      const audioUrl = `/api/clip/${speciesCode}/${r.safe_id}/${r.ch}/${r.window_start}`;
      const spectUrl = `/api/spectrogram/${speciesCode}/${r.safe_id}/${r.ch}/${r.window_start}`;
      html += `<div class="repr-card">
        <img src="${spectUrl}" loading="lazy" alt="spectrogram">
        <audio controls preload="none">
          <source src="${audioUrl}" type="audio/wav">
        </audio>
        <div class="repr-info">
          ${r.safe_id} src${r.ch}<br>
          offset ${r.window_start.toFixed(1)}s, SNR ${r.snr.toFixed(1)} dB
        </div>
      </div>`;
    });

    html += `</div></div>`;
  });

  container.innerHTML = html;
  document.getElementById('save-bar').style.display = 'flex';
  updateSelection();
}

function updateSelection() {
  const checks = document.querySelectorAll('input[name="cluster"]:checked');
  const selected = Array.from(checks).map(c => parseInt(c.value));
  const summary = document.getElementById('selection-summary');
  summary.textContent = selected.length > 0
    ? `${selected.length} cluster(s) selected: [${selected.join(', ')}]`
    : 'No clusters selected';

  // Visual highlight
  document.querySelectorAll('.cluster-card').forEach(card => {
    card.classList.remove('selected');
  });
  selected.forEach(label => {
    const card = document.getElementById(`cluster-${label}`);
    if (card) card.classList.add('selected');
  });
}

function saveSelection() {
  const checks = document.querySelectorAll('input[name="cluster"]:checked');
  const selected = Array.from(checks).map(c => parseInt(c.value));

  if (selected.length === 0) {
    document.getElementById('save-status').className = 'save-status err';
    document.getElementById('save-status').textContent = 'Select at least one cluster.';
    return;
  }

  const btn = document.getElementById('save-btn');
  const status = document.getElementById('save-status');
  btn.disabled = true;
  status.className = 'save-status';
  status.textContent = 'Saving...';

  fetch(`/api/save/${speciesCode}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({selected_clusters: selected}),
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    if (data.ok) {
      status.className = 'save-status ok';
      status.textContent = `Saved: ${data.n_prototypes} prototype(s), ${data.n_embeddings} embeddings`;
    } else {
      status.className = 'save-status err';
      status.textContent = `Error: ${data.error}`;
    }
  })
  .catch(err => {
    btn.disabled = false;
    status.className = 'save-status err';
    status.textContent = `Error: ${err.message}`;
  });
}
</script>
</body>
</html>
""")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Cluster Selector — ToriNet")

_species_names: dict = {}
_species_list: list[str] = []

# Cache clustering results to avoid recomputation
_cluster_cache: dict[str, dict] = {}


@app.on_event("startup")
def startup():
    global _species_names, _species_list
    print("Loading species data...", flush=True)
    _species_names = load_species_names()
    _species_list = list_species_with_embeddings()
    print(f"  {len(_species_list)} species with embeddings", flush=True)
    print(f"  {len(_species_names)} species names loaded", flush=True)
    print("Ready.", flush=True)


def _get_proto_status() -> dict[str, str]:
    """Check prototype status for all species."""
    status = {}
    if not PROTOTYPES_DIR.exists():
        return status
    for p in PROTOTYPES_DIR.glob("*.npz"):
        sp = p.stem
        try:
            data = np.load(p)
            method = str(data.get("method", "unknown"))
            status[sp] = method
        except Exception:
            status[sp] = "unknown"
    return status


@app.get("/", response_class=HTMLResponse)
def index():
    proto_status = _get_proto_status()
    n_user_selected = sum(1 for v in proto_status.values() if v == "user-selected")

    return INDEX_TEMPLATE.render(
        n_species=len(_species_list),
        n_user_selected=n_user_selected,
        species_list=_species_list,
        species_names=_species_names,
        proto_status=proto_status,
    )


@app.get("/cluster/{species_code}", response_class=HTMLResponse)
def cluster_page(species_code: str):
    names = _species_names.get(species_code, {})
    ja_name = names.get("ja", "") or species_code
    sci_name = names.get("sci", "")

    return CLUSTER_TEMPLATE.render(
        species_code=species_code,
        ja_name=ja_name,
        sci_name=sci_name,
    )


@app.get("/api/clusters/{species_code}")
def api_clusters(species_code: str):
    """Run clustering and return cluster data as JSON."""
    if species_code in _cluster_cache:
        result = _cluster_cache[species_code]
    else:
        result = run_clustering(species_code)
        if result is not None:
            _cluster_cache[species_code] = result

    if result is None:
        return JSONResponse({"error": f"No embeddings found for {species_code} "
                             f"(or too few after filtering)."})

    # Check existing prototype
    prev_method = None
    prev_n_protos = 0
    proto_path = PROTOTYPES_DIR / f"{species_code}.npz"
    if proto_path.exists():
        try:
            pdata = np.load(proto_path)
            prev_method = str(pdata.get("method", "unknown"))
            prev_n_protos = int(pdata.get("n_prototypes", 0))
        except Exception:
            pass

    # Serialize clusters (strip numpy arrays for JSON)
    clusters_json = []
    n_noise = 0
    for cl in result["clusters"]:
        if cl["is_noise"]:
            n_noise = cl["size"]
        clusters_json.append({
            "label": int(cl["label"]),
            "size": int(cl["size"]),
            "mean_snr": float(cl["mean_snr"]),
            "mean_ndsi": float(cl["mean_ndsi"]),
            "std_snr": float(cl["std_snr"]),
            "is_noise": bool(cl["is_noise"]),
            "representatives": [
                {k: (float(v) if isinstance(v, (np.floating,)) else
                     int(v) if isinstance(v, (np.integer,)) else v)
                 for k, v in rep.items()}
                for rep in cl["representatives"]
            ],
        })

    return JSONResponse({
        "clusters": clusters_json,
        "n_total": int(result["n_total"]),
        "n_raw": int(result["n_raw"]),
        "n_filtered_out": int(result["n_filtered_out"]),
        "n_clusters": len([c for c in result["clusters"] if not c["is_noise"]]),
        "n_noise": int(n_noise),
        "snr_threshold": float(SNR_THRESHOLD),
        "ndsi_threshold": float(NDSI_THRESHOLD),
        "previous_method": str(prev_method) if prev_method else None,
        "previous_n_prototypes": int(prev_n_protos) if prev_n_protos else 0,
    })


@app.get("/api/clip/{species}/{safe_id}/{ch}/{window_start}")
def api_clip(species: str, safe_id: str, ch: int, window_start: float):
    """Serve a 5-second audio clip from a source WAV."""
    wav_bytes = extract_audio_clip(species, safe_id, ch, window_start)
    if wav_bytes is None:
        return Response(status_code=404)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/spectrogram/{species}/{safe_id}/{ch}/{window_start}")
def api_spectrogram(species: str, safe_id: str, ch: int, window_start: float):
    """Generate and serve a spectrogram PNG for a 5-second clip."""
    png_data = generate_clip_spectrogram(species, safe_id, ch, window_start)
    if png_data is None:
        return Response(status_code=404)
    return Response(content=png_data, media_type="image/png")


@app.post("/api/save/{species_code}")
async def api_save(species_code: str, request: Request):
    """Save user-selected clusters as the species prototype."""
    try:
        body = await request.json()
        selected_labels = body.get("selected_clusters", [])
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid request: {e}"})

    if not selected_labels:
        return JSONResponse({"ok": False, "error": "No clusters selected."})

    # Get cached clustering result
    result = _cluster_cache.get(species_code)
    if result is None:
        # Re-run clustering
        result = run_clustering(species_code)
        if result is None:
            return JSONResponse({"ok": False, "error": "No clustering data available."})
        _cluster_cache[species_code] = result

    # Compute centroids for selected clusters
    selected_centroids = []
    total_points = 0
    for cl in result["clusters"]:
        if cl["label"] in selected_labels:
            selected_centroids.append(cl["centroid"])
            total_points += cl["size"]

    if not selected_centroids:
        return JSONResponse({"ok": False, "error": "Selected clusters not found in data."})

    prototypes = np.stack(selected_centroids)

    # Save
    PROTOTYPES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROTOTYPES_DIR / f"{species_code}.npz"
    np.savez(
        out_path,
        prototypes=prototypes,
        n_embeddings=result["n_total"],
        n_prototypes=len(selected_centroids),
        method="user-selected",
    )

    print(f"Saved prototype: {species_code} — {len(selected_centroids)} clusters, "
          f"{total_points} points, {result['n_total']} total embeddings", flush=True)

    return JSONResponse({
        "ok": True,
        "n_prototypes": len(selected_centroids),
        "n_embeddings": result["n_total"],
        "total_points": total_points,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Cluster selection UI for species prototypes")
    parser.add_argument("--port", type=int, default=8053)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting cluster selector at http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
