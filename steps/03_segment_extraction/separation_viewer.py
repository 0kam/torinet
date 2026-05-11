"""
Bird-MixIT source separation viewer — lightweight web app.

Browse Bird-MixIT separation results: original recordings, 4 separated
sources, focal channel selection, and extracted segments per species.

Usage:
    python separation_viewer.py [--port 8052]
"""

import argparse
import ast
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from jinja2 import Template

matplotlib.use("Agg")

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent.parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
SEGMENTS_BASE = NAS_BASE / "segments"
TEST_SAMPLES_DIR = SEGMENTS_BASE / "test_samples"
SOURCES_DIR = SEGMENTS_BASE / "birdmixit_sources"
SELECTED_DIR = SEGMENTS_BASE / "birdmixit_selected"
RESULTS_CSV = STEP_DIR / "birdmixit_pipeline_results.csv"
SAMPLES_CSV = STEP_DIR / "test_samples.csv"
SPECIES_LIST_CSV = REPO_ROOT / "steps" / "01_species_list" / "species_list.csv"

# Map relative path prefixes to NAS directories
PATH_ROOTS = {
    "test_samples": TEST_SAMPLES_DIR,
    "birdmixit_sources": SOURCES_DIR,
    "birdmixit_selected": SELECTED_DIR,
}


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


def load_results() -> pd.DataFrame:
    """Load the pipeline results CSV."""
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_CSV)
    # Parse channel_scores from string to list
    if "channel_scores" in df.columns:
        df["channel_scores_list"] = df["channel_scores"].apply(_parse_scores)
    return df


def _parse_scores(val):
    """Parse channel_scores string like '[0.811, 0.535, 0.001, 0.523]' to list."""
    if pd.isna(val):
        return [0, 0, 0, 0]
    try:
        return list(ast.literal_eval(str(val)))
    except Exception:
        return [0, 0, 0, 0]


def build_species_index(results_df: pd.DataFrame) -> dict:
    """Build species index from source directories and results.

    Returns {species_code: {
        n_recordings: int,  # recordings with sources
        n_selected: int,    # selected segments
        recordings: [...]   # list of recording info dicts
    }}
    """
    index = {}

    # Scan sources directory for species with separated sources
    if SOURCES_DIR.exists():
        for sp_dir in sorted(SOURCES_DIR.iterdir()):
            if not sp_dir.is_dir():
                continue
            code = sp_dir.name
            # Find unique safe_ids from source files
            src_files = list(sp_dir.glob("*_src0.wav"))
            safe_ids = sorted({f.stem.rsplit("_src", 1)[0] for f in src_files})

            # Count selected segments
            sel_dir = SELECTED_DIR / code
            n_selected = len(list(sel_dir.glob("*.wav"))) if sel_dir.exists() else 0

            index[code] = {
                "n_recordings": len(safe_ids),
                "n_selected": n_selected,
                "safe_ids": safe_ids,
            }

    return index


def get_recording_data(
    species_code: str, safe_id: str, results_df: pd.DataFrame
) -> dict:
    """Build recording data dict for a single recording."""
    rec = {"safe_id": safe_id, "focal_ch": None, "scores": [0, 0, 0, 0]}

    # Find original file
    orig_dir = TEST_SAMPLES_DIR / species_code
    orig_file = None
    if orig_dir.exists():
        for ext in (".mp3", ".wav", ".ogg", ".flac"):
            candidate = orig_dir / f"{safe_id}{ext}"
            if candidate.exists():
                orig_file = f"test_samples/{species_code}/{candidate.name}"
                break
    rec["original_path"] = orig_file

    # Check which sources exist
    src_dir = SOURCES_DIR / species_code
    rec["sources"] = []
    for i in range(4):
        src_file = src_dir / f"{safe_id}_src{i}.wav"
        if src_file.exists():
            rec["sources"].append(
                f"birdmixit_sources/{species_code}/{safe_id}_src{i}.wav"
            )
        else:
            rec["sources"].append(None)

    # Get channel scores and focal channel from results
    if not results_df.empty and "safe_id" in results_df.columns:
        rec_rows = results_df[results_df["safe_id"] == safe_id]
        if not rec_rows.empty:
            first = rec_rows.iloc[0]
            rec["focal_ch"] = (
                int(first["focal_channel"])
                if pd.notna(first.get("focal_channel"))
                else None
            )
            scores = first.get("channel_scores_list", [0, 0, 0, 0])
            if not scores or len(scores) < 4:
                scores = [0, 0, 0, 0]
            rec["scores"] = scores

    # Get selected segments (bouts) for this recording
    sel_dir = SELECTED_DIR / species_code
    rec["selected_segments"] = []
    if sel_dir.exists():
        seg_files = sorted(sel_dir.glob(f"{safe_id}_b*.wav"))
        for sf_path in seg_files:
            seg_info = {
                "path": f"birdmixit_selected/{species_code}/{sf_path.name}",
                "filename": sf_path.name,
            }
            # Try to find bout info from results
            bout_idx_str = sf_path.stem.rsplit("_b", 1)[-1]
            defaults = {
                "bout_idx": "-",
                "bout_onset": "-",
                "bout_offset": "-",
                "bout_duration": "-",
                "mean_rms": "-",
                "n_notes": "-",
                "silence_ratio": "-",
                "notes_json": "[]",
            }
            try:
                bout_idx = int(bout_idx_str)
                if not results_df.empty:
                    match = results_df[
                        (results_df["safe_id"] == safe_id)
                        & (results_df["bout_idx"] == bout_idx)
                    ]
                    if not match.empty:
                        row = match.iloc[0]
                        seg_info["bout_idx"] = bout_idx
                        seg_info["bout_onset"] = (
                            f"{row['bout_onset']:.2f}"
                            if pd.notna(row.get("bout_onset"))
                            else "-"
                        )
                        seg_info["bout_offset"] = (
                            f"{row['bout_offset']:.2f}"
                            if pd.notna(row.get("bout_offset"))
                            else "-"
                        )
                        seg_info["bout_duration"] = (
                            f"{row['bout_duration']:.2f}"
                            if pd.notna(row.get("bout_duration"))
                            else "-"
                        )
                        seg_info["mean_rms"] = (
                            f"{row['mean_rms']:.4f}"
                            if pd.notna(row.get("mean_rms"))
                            else "-"
                        )
                        seg_info["n_notes"] = (
                            int(row["n_notes"])
                            if pd.notna(row.get("n_notes"))
                            else "-"
                        )
                        seg_info["silence_ratio"] = (
                            f"{row['silence_ratio']:.2f}"
                            if pd.notna(row.get("silence_ratio"))
                            else "-"
                        )
                        seg_info["notes_json"] = (
                            str(row["notes_json"])
                            if pd.notna(row.get("notes_json"))
                            else "[]"
                        )
                    else:
                        seg_info.update(defaults)
                else:
                    seg_info.update(defaults)
            except (ValueError, IndexError):
                seg_info.update(defaults)
            rec["selected_segments"].append(seg_info)

    return rec


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def generate_spectrogram_png(
    filepath: str, width: float = 3.0, height: float = 0.8
) -> bytes | None:
    """Generate a mel spectrogram PNG for an audio file."""
    p = Path(filepath)
    if not p.exists():
        return None
    try:
        y, sr = librosa.load(str(p), sr=22050, mono=True, duration=30.0)
        if len(y) == 0:
            return None

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=11000, hop_length=512
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=100)
        ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="antialiased",
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


MIXIT_SR = 22050


@lru_cache(maxsize=256)
def generate_focal_overlay_png(
    focal_wav_path: str,
    bout_onset: float,
    bout_offset: float,
    notes_json: str,
    width: float = 8.0,
    height: float = 1.8,
) -> bytes | None:
    """Generate a mel spectrogram PNG of the focal channel with bout/note overlays."""
    p = Path(focal_wav_path)
    if not p.exists():
        return None
    try:
        y, sr = librosa.load(str(p), sr=MIXIT_SR, mono=True)
        if len(y) == 0:
            return None

        audio_duration = len(y) / sr
        t_start = max(0.0, bout_onset - 2.0)
        t_end = min(audio_duration, bout_offset + 2.0)

        # Slice audio to the display window
        s_start = int(t_start * sr)
        s_end = int(t_end * sr)
        y_win = y[s_start:s_end]
        if len(y_win) == 0:
            return None

        S = librosa.feature.melspectrogram(
            y=y_win, sr=sr, n_mels=64, fmax=11000, hop_length=512
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        freq_max = 11000

        fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[t_start, t_end, 0, freq_max],
        )

        # Dim regions outside the bout to create contrast
        if bout_onset > t_start:
            ax.axvspan(t_start, bout_onset, alpha=0.55, color="black", zorder=2)
        if bout_offset < t_end:
            ax.axvspan(bout_offset, t_end, alpha=0.55, color="black", zorder=2)

        # Bout boundary lines (solid, bright)
        ax.axvline(
            bout_onset, color="#58a6ff", linewidth=2, linestyle="-", alpha=0.9, zorder=5
        )
        ax.axvline(
            bout_offset, color="#58a6ff", linewidth=2, linestyle="-", alpha=0.9, zorder=5
        )

        # Horizontal bout markers at top and bottom
        ax.plot(
            [bout_onset, bout_offset], [freq_max * 0.97] * 2,
            color="#58a6ff", linewidth=3, solid_capstyle="butt", alpha=0.9, zorder=5,
        )
        ax.plot(
            [bout_onset, bout_offset], [freq_max * 0.03] * 2,
            color="#58a6ff", linewidth=3, solid_capstyle="butt", alpha=0.9, zorder=5,
        )

        # Note overlays (bright green filled spans)
        try:
            notes = json.loads(notes_json) if notes_json else []
        except (json.JSONDecodeError, TypeError):
            notes = []
        for note in notes:
            if isinstance(note, (list, tuple)) and len(note) >= 2:
                note_on, note_off = float(note[0]), float(note[1])
                ax.axvspan(note_on, note_off, alpha=0.3, color="#2ecc71", zorder=3)
                # Bright bar at bottom to mark note extent
                ax.axvspan(
                    note_on, note_off, ymin=0, ymax=0.04,
                    alpha=0.9, color="#2ecc71", zorder=6,
                )

        ax.set_xlim(t_start, t_end)
        ax.set_ylim(0, freq_max)
        ax.tick_params(axis="x", colors="#8899aa", labelsize=8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout(pad=0.3)

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=100,
            facecolor=fig.get_facecolor(),
        )
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def _resolve_nas_path(rel_path: str) -> Path | None:
    """Resolve a relative path like 'test_samples/species/file' to NAS absolute path."""
    parts = rel_path.split("/", 1)
    if len(parts) < 2:
        return None
    root_key = parts[0]
    remainder = parts[1]
    root = PATH_ROOTS.get(root_key)
    if root is None:
        return None
    resolved = root / remainder
    # Prevent directory traversal
    try:
        resolved.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Bird-MixIT Separation Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { background: #16213e; color: white; padding: 16px 24px; border-bottom: 1px solid #0f3460; }
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
  .count { font-weight: 600; }
  .count-zero { color: #445566; }
</style>
</head>
<body>
<div class="header">
  <h1>Bird-MixIT Separation Viewer</h1>
  <p>Step 03 — {{ n_species }} species with separated sources</p>
</div>
<div class="container">
  <div class="summary">
    <div class="summary-card">
      <h3>Species</h3>
      <div class="value">{{ n_species }}</div>
      <div class="sub">with Bird-MixIT sources</div>
    </div>
    <div class="summary-card">
      <h3>Recordings</h3>
      <div class="value">{{ total_recordings }}</div>
      <div class="sub">separated into 4 sources each</div>
    </div>
    <div class="summary-card">
      <h3>Selected Segments</h3>
      <div class="value">{{ total_selected }}</div>
      <div class="sub">extracted from focal channels</div>
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
        <th>Recordings</th>
        <th>Selected Segments</th>
      </tr>
    </thead>
    <tbody>
      {% for sp, info in species_index.items() %}
      <tr data-code="{{ sp }}" data-ja="{{ species_names.get(sp, {}).get('ja', '') }}"
          data-sci="{{ species_names.get(sp, {}).get('sci', '') }}">
        <td>
          <a href="/species/{{ sp }}">{{ species_names.get(sp, {}).get('ja', '') or sp }}</a><br>
          <span class="code"><em>{{ species_names.get(sp, {}).get('sci', '') }}</em></span>
        </td>
        <td class="code">{{ sp }}</td>
        <td class="count">{{ info.n_recordings }}</td>
        <td class="count {% if info.n_selected == 0 %}count-zero{% endif %}">{{ info.n_selected }}</td>
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

SPECIES_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{{ ja_name }} — Bird-MixIT Separation</title>
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

  .stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 14px; border: 1px solid #0f3460; }
  .stat-card h3 { font-size: 14px; color: #e94560; margin-bottom: 8px; }
  .stat-card .detail { font-size: 13px; color: #8899aa; line-height: 1.6; }
  .stat-card .detail strong { color: #e0e0e0; }

  .recording-card { background: #16213e; border-radius: 8px; margin-bottom: 12px;
                    border: 1px solid #0f3460; overflow: hidden; }
  .recording-header { padding: 12px 16px; cursor: pointer; display: flex;
                      align-items: center; gap: 12px; user-select: none; }
  .recording-header:hover { background: #1a1a3e; }
  .recording-header .expand-icon { color: #667788; font-size: 12px; transition: transform 0.2s; }
  .recording-header.expanded .expand-icon { transform: rotate(90deg); }
  .rec-id { font-weight: 600; font-size: 14px; color: #e0e0e0; }
  .focal-badge { background: #1b4332; color: #52b788; padding: 2px 8px; border-radius: 4px;
                 font-size: 12px; font-weight: 600; }
  .seg-count-badge { background: #0f3460; color: #8899aa; padding: 2px 8px; border-radius: 4px;
                     font-size: 12px; }

  .recording-body { padding: 0 16px 16px 16px; display: none; }
  .recording-body.visible { display: block; }

  .original-section { margin-bottom: 16px; }
  .original-section h4 { font-size: 13px; color: #8899aa; margin-bottom: 8px; }
  .original-section img { width: 100%; max-width: 600px; height: 100px; border-radius: 4px;
                          display: block; background: #0a0a1a; object-fit: cover; }
  .original-section audio { width: 100%; max-width: 600px; height: 32px; margin-top: 6px; }

  .sources-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
  .source-card { background: #0f1a2e; border-radius: 6px; padding: 10px;
                 border: 1px solid #1a2a4e; }
  .source-card.focal { border-color: #52b788; background: #0f2a1e; }
  .source-card h5 { font-size: 12px; color: #8899aa; margin-bottom: 6px; }
  .source-card.focal h5 { color: #52b788; }
  .source-card img { width: 100%; height: 80px; border-radius: 4px; display: block;
                     background: #0a0a1a; object-fit: cover; }
  .source-card audio { width: 100%; height: 28px; margin-top: 4px; }
  .source-card .score { font-size: 11px; color: #667788; margin-top: 4px; }
  .source-card.focal .score { color: #52b788; }

  .selected-section { margin-top: 12px; }
  .selected-section h4 { font-size: 13px; color: #8899aa; margin-bottom: 8px; }
  .segments-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .segment-card { background: #0f1a2e; border-radius: 6px; padding: 8px;
                  border: 1px solid #1a2a4e; width: 320px; }
  .segment-card img { width: 100%; border-radius: 4px; display: block;
                      background: #0a0a1a; object-fit: cover; }
  .segment-card audio { width: 100%; height: 28px; margin-top: 4px; }
  .segment-card .seg-detail { font-size: 11px; color: #667788; margin-top: 4px; }

  .no-data { color: #445566; font-style: italic; padding: 12px 0; }
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
<div class="container">
  <div class="stats-row">
    <div class="stat-card">
      <h3>Recordings</h3>
      <div class="detail">
        <strong>{{ recordings | length }}</strong> recordings with sources
      </div>
    </div>
    <div class="stat-card">
      <h3>Selected Segments</h3>
      <div class="detail">
        <strong>{{ total_selected }}</strong> segments extracted
      </div>
    </div>
    <div class="stat-card">
      <h3>Focal Channel Distribution</h3>
      <div class="detail">
        {% for ch, cnt in focal_dist.items() %}
        Ch {{ ch }}: <strong>{{ cnt }}</strong>{% if not loop.last %}, {% endif %}
        {% endfor %}
        {% if not focal_dist %}<span class="no-data">No data</span>{% endif %}
      </div>
    </div>
  </div>

  {% for rec in recordings %}
  <div class="recording-card">
    <div class="recording-header" onclick="toggleExpand(this)">
      <span class="expand-icon">&#9654;</span>
      <span class="rec-id">{{ rec.safe_id }}</span>
      {% if rec.focal_ch is not none %}
      <span class="focal-badge">Focal: Ch {{ rec.focal_ch }} ({{ "%.3f"|format(rec.scores[rec.focal_ch]) }})</span>
      {% endif %}
      {% if rec.selected_segments %}
      <span class="seg-count-badge">{{ rec.selected_segments | length }} bouts</span>
      {% endif %}
    </div>
    <div class="recording-body" id="body-{{ rec.safe_id }}">
      {% if rec.original_path %}
      <div class="original-section">
        <h4>Original</h4>
        <img src="/spectrogram/{{ rec.original_path }}?w=6&h=1" loading="lazy" alt="original spectrogram">
        <audio controls preload="none">
          <source src="/audio/{{ rec.original_path }}" type="audio/{{ 'mpeg' if rec.original_path.endswith('.mp3') else 'wav' }}">
        </audio>
      </div>
      {% else %}
      <div class="no-data">Original recording not found</div>
      {% endif %}

      <div class="sources-grid">
        {% for i in range(4) %}
        <div class="source-card {{ 'focal' if i == rec.focal_ch else '' }}">
          <h5>Source {{ i }} {{ '&#9733; FOCAL' if i == rec.focal_ch else '' }}</h5>
          {% if rec.sources[i] %}
          <img src="/spectrogram/{{ rec.sources[i] }}" loading="lazy" alt="source {{ i }}">
          <audio controls preload="none">
            <source src="/audio/{{ rec.sources[i] }}" type="audio/wav">
          </audio>
          <div class="score">Score: {{ "%.3f"|format(rec.scores[i]) }}</div>
          {% else %}
          <div class="no-data">Source file not found</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>

      {% if rec.selected_segments %}
      <div class="selected-section">
        <h4>Selected Bouts ({{ rec.selected_segments | length }})</h4>
        <div class="segments-row">
          {% for seg in rec.selected_segments %}
          <div class="segment-card">
            <img src="/focal-overlay/{{ species_code }}/{{ rec.safe_id }}/{{ seg.bout_idx }}" loading="lazy" alt="focal overlay" style="width: 100%; border-radius: 4px;">
            <img src="/spectrogram/{{ seg.path }}" loading="lazy" alt="{{ seg.filename }}">
            <audio controls preload="none">
              <source src="/audio/{{ seg.path }}" type="audio/wav">
            </audio>
            <div class="seg-detail">
              {{ seg.bout_onset }}s – {{ seg.bout_offset }}s ({{ seg.bout_duration }}s)<br>
              {{ seg.n_notes }} notes, silence {{ seg.silence_ratio }}
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
      {% endif %}
    </div>
  </div>
  {% endfor %}

  {% if not recordings %}
  <div class="no-data">No recordings found for this species.</div>
  {% endif %}
</div>
<script>
function toggleExpand(header) {
  const body = header.nextElementSibling;
  const isVisible = body.classList.contains('visible');
  body.classList.toggle('visible');
  header.classList.toggle('expanded');
  if (!isVisible) {
    body.style.display = 'block';
  } else {
    body.style.display = 'none';
  }
}
</script>
</body>
</html>
""")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Bird-MixIT Separation Viewer")

_results_df: pd.DataFrame = pd.DataFrame()
_species_index: dict = {}
_species_names: dict = {}


@app.on_event("startup")
def startup():
    global _results_df, _species_index, _species_names
    print("Loading Bird-MixIT pipeline results...", flush=True)
    _results_df = load_results()
    print(f"  Results: {len(_results_df)} rows", flush=True)
    _species_index = build_species_index(_results_df)
    _species_names = load_species_names()
    print(f"  {len(_species_index)} species with sources", flush=True)
    print(f"  {len(_species_names)} species names loaded", flush=True)
    print("Ready.", flush=True)


@app.get("/", response_class=HTMLResponse)
def index():
    total_recordings = sum(v["n_recordings"] for v in _species_index.values())
    total_selected = sum(v["n_selected"] for v in _species_index.values())

    return INDEX_TEMPLATE.render(
        n_species=len(_species_index),
        total_recordings=total_recordings,
        total_selected=total_selected,
        species_index=_species_index,
        species_names=_species_names,
    )


@app.get("/species/{species_code}", response_class=HTMLResponse)
def species_page(species_code: str):
    names = _species_names.get(species_code, {})
    ja_name = names.get("ja", "") or species_code
    sci_name = names.get("sci", "")

    sp_info = _species_index.get(species_code, {"safe_ids": [], "n_selected": 0})
    safe_ids = sp_info.get("safe_ids", [])

    recordings = []
    focal_dist: dict[int, int] = {}
    total_selected = 0

    for sid in safe_ids:
        rec = get_recording_data(species_code, sid, _results_df)
        recordings.append(rec)
        if rec["focal_ch"] is not None:
            focal_dist[rec["focal_ch"]] = focal_dist.get(rec["focal_ch"], 0) + 1
        total_selected += len(rec["selected_segments"])

    # Sort focal_dist by channel number
    focal_dist = dict(sorted(focal_dist.items()))

    return SPECIES_TEMPLATE.render(
        species_code=species_code,
        ja_name=ja_name,
        sci_name=sci_name,
        recordings=recordings,
        total_selected=total_selected,
        focal_dist=focal_dist,
    )


@app.get("/audio/{path:path}")
def serve_audio(path: str):
    """Serve audio files (WAV/MP3) from NAS."""
    resolved = _resolve_nas_path(path)
    if resolved is None or not resolved.exists():
        return Response(status_code=404)

    suffix = resolved.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return Response(content=resolved.read_bytes(), media_type=media_type)


@app.get("/spectrogram/{path:path}")
def serve_spectrogram(request: Request, path: str):
    """Generate and serve a mel spectrogram PNG for an audio file."""
    resolved = _resolve_nas_path(path)
    if resolved is None:
        return Response(status_code=404)

    # Optional width/height from query params
    w = float(request.query_params.get("w", "3.0"))
    h = float(request.query_params.get("h", "0.8"))

    png_data = generate_spectrogram_png(str(resolved), width=w, height=h)
    if png_data is None:
        return Response(status_code=404)
    return Response(content=png_data, media_type="image/png")


@app.get("/focal-overlay/{species_code}/{safe_id}/{bout_idx}")
def serve_focal_overlay(species_code: str, safe_id: str, bout_idx: int):
    """Generate and serve a focal channel spectrogram with bout/note overlays."""
    if _results_df.empty:
        return Response(status_code=404)

    match = _results_df[
        (_results_df["safe_id"] == safe_id) & (_results_df["bout_idx"] == bout_idx)
    ]
    if match.empty:
        return Response(status_code=404)

    row = match.iloc[0]
    focal_ch = int(row["focal_channel"]) if pd.notna(row.get("focal_channel")) else 0
    bout_onset = float(row["bout_onset"]) if pd.notna(row.get("bout_onset")) else 0.0
    bout_offset = float(row["bout_offset"]) if pd.notna(row.get("bout_offset")) else 0.0
    notes_json_val = str(row["notes_json"]) if pd.notna(row.get("notes_json")) else "[]"

    focal_wav = SOURCES_DIR / species_code / f"{safe_id}_src{focal_ch}.wav"
    png_data = generate_focal_overlay_png(
        str(focal_wav), bout_onset, bout_offset, notes_json_val
    )
    if png_data is None:
        return Response(status_code=404)
    return Response(content=png_data, media_type="image/png")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Bird-MixIT separation viewer")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Bird-MixIT separation viewer at http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
