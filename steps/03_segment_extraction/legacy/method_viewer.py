"""
Method comparison viewer — lightweight web app.

Browse and compare segment extraction results across three methods
(bambird, biodenoising, bird-mixit) for each species. Includes
Japanese species names, audio playback, and on-the-fly spectrograms.

Usage:
    python method_viewer.py [--port 8051]
"""

import argparse
import base64
import io
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

METHODS = {
    "bambird": {
        "label": "bambird",
        "dir": SEGMENTS_BASE / "method_bambird",
        "csv": "bambird_results.csv",
        "quality_col": "power",
        "quality_ascending": False,
    },
    "biodenoising": {
        "label": "biodenoising",
        "dir": SEGMENTS_BASE / "method_biodenoising",
        "csv": "biodenoising_results.csv",
        "quality_col": "rms_denoised",
        "quality_ascending": False,
    },
    "birdmixit": {
        "label": "bird-mixit",
        "dir": SEGMENTS_BASE / "method_birdmixit",
        "csv": "birdmixit_results.csv",
        "quality_col": "score",
        "quality_ascending": False,
    },
}

MAX_SAMPLES = 20

# ---------------------------------------------------------------------------
# Species name mapping
# ---------------------------------------------------------------------------


def load_species_names() -> dict:
    """Load ebird_species_code -> {ja, sci} mapping."""
    csv_path = REPO_ROOT / "steps" / "01_species_list" / "species_list.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
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
# Load data
# ---------------------------------------------------------------------------


def load_method_results() -> dict[str, pd.DataFrame]:
    """Load result CSVs for all three methods."""
    results = {}
    for method_key, info in METHODS.items():
        csv_path = STEP_DIR / info["csv"]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[method_key] = df
        else:
            results[method_key] = pd.DataFrame()
    return results


def build_species_index(results: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Build {species_code: {method: segment_count}} from results."""
    all_species = set()
    for df in results.values():
        if not df.empty and "species_code" in df.columns:
            all_species.update(df["species_code"].unique())

    index = {}
    for sp in sorted(all_species):
        counts = {}
        for method_key, df in results.items():
            if not df.empty and "species_code" in df.columns:
                counts[method_key] = int(len(df[df["species_code"] == sp]))
            else:
                counts[method_key] = 0
        index[sp] = counts
    return index


def get_species_segments(
    species_code: str, method_key: str, results: dict[str, pd.DataFrame]
) -> list[dict]:
    """Get segment info for a species+method, sorted by quality, limited to MAX_SAMPLES."""
    df = results.get(method_key, pd.DataFrame())
    if df.empty:
        return []

    sp_df = df[df["species_code"] == species_code].copy()
    if sp_df.empty:
        return []

    info = METHODS[method_key]
    quality_col = info["quality_col"]
    if quality_col in sp_df.columns:
        # Drop NaN quality values to the end
        sp_df = sp_df.sort_values(
            quality_col, ascending=info["quality_ascending"], na_position="last"
        )

    sp_df = sp_df.head(MAX_SAMPLES)

    segments = []
    seg_dir = info["dir"] / species_code
    for _, row in sp_df.iterrows():
        safe_id = row.get("safe_id", "")
        note_idx = row.get("note_idx", row.get("cluster_id", 0))
        filename = f"{safe_id}_n{int(note_idx):03d}.wav"
        filepath = seg_dir / filename
        if not filepath.exists():
            continue

        seg = {
            "filename": filename,
            "safe_id": safe_id,
            "onset": f"{row.get('onset', 0):.2f}" if pd.notna(row.get("onset")) else "-",
            "offset": f"{row.get('offset', 0):.2f}" if pd.notna(row.get("offset")) else "-",
            "duration": f"{(row.get('offset', 0) - row.get('onset', 0)):.2f}"
            if pd.notna(row.get("onset")) and pd.notna(row.get("offset"))
            else "-",
        }

        # Add quality metric
        if quality_col in sp_df.columns and pd.notna(row.get(quality_col)):
            seg["quality"] = f"{row[quality_col]:.4f}"
            seg["quality_label"] = quality_col
        else:
            seg["quality"] = "-"
            seg["quality_label"] = quality_col

        segments.append(seg)

    return segments


def get_species_stats(
    species_code: str, method_key: str, results: dict[str, pd.DataFrame]
) -> dict:
    """Compute summary stats for a species+method."""
    df = results.get(method_key, pd.DataFrame())
    if df.empty:
        return {"count": 0, "avg_duration": "-", "n_recordings": 0}

    sp_df = df[df["species_code"] == species_code]
    if sp_df.empty:
        return {"count": 0, "avg_duration": "-", "n_recordings": 0}

    count = len(sp_df)
    n_recordings = sp_df["recording_id"].nunique() if "recording_id" in sp_df.columns else 0

    if "onset" in sp_df.columns and "offset" in sp_df.columns:
        durations = sp_df["offset"] - sp_df["onset"]
        avg_dur = durations.mean()
        avg_duration = f"{avg_dur:.2f}s" if pd.notna(avg_dur) else "-"
    else:
        avg_duration = "-"

    return {"count": count, "avg_duration": avg_duration, "n_recordings": n_recordings}


# ---------------------------------------------------------------------------
# Spectrogram generation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=512)
def generate_spectrogram_png(filepath: str) -> bytes | None:
    """Generate a small mel spectrogram PNG for a WAV file."""
    p = Path(filepath)
    if not p.exists():
        return None
    try:
        y, sr = librosa.load(str(p), sr=22050, mono=True, duration=10.0)
        if len(y) == 0:
            return None

        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=64, fmax=11000, hop_length=512
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 1, figsize=(3.0, 0.8), dpi=100)
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


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Method Comparison Viewer</title>
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
  <h1>Method Comparison Viewer</h1>
  <p>Step 03 — {{ n_species }} species, 3 methods (bambird, biodenoising, bird-mixit)</p>
</div>
<div class="container">
  <div class="summary">
    {% for method_key, info in methods.items() %}
    <div class="summary-card">
      <h3>{{ info.label }}</h3>
      <div class="value">{{ totals[method_key] }}</div>
      <div class="sub">segments across {{ n_species_with[method_key] }} species</div>
    </div>
    {% endfor %}
  </div>

  <div class="filter-bar">
    <input type="text" id="filter" placeholder="Filter species..." oninput="filterSpecies()">
  </div>

  <table class="species-table" id="species-table">
    <thead>
      <tr>
        <th>Species</th>
        <th>Code</th>
        <th>bambird</th>
        <th>biodenoising</th>
        <th>bird-mixit</th>
        <th>Total</th>
      </tr>
    </thead>
    <tbody>
      {% for sp, counts in species_index.items() %}
      <tr data-code="{{ sp }}" data-ja="{{ species_names.get(sp, {}).get('ja', '') }}"
          data-sci="{{ species_names.get(sp, {}).get('sci', '') }}">
        <td>
          <a href="/species/{{ sp }}">{{ species_names.get(sp, {}).get('ja', '') or sp }}</a><br>
          <span class="code"><em>{{ species_names.get(sp, {}).get('sci', '') }}</em></span>
        </td>
        <td class="code">{{ sp }}</td>
        {% for mk in ['bambird', 'biodenoising', 'birdmixit'] %}
        <td class="count {% if counts[mk] == 0 %}count-zero{% endif %}">{{ counts[mk] }}</td>
        {% endfor %}
        <td class="count">{{ counts.values() | sum }}</td>
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
<title>{{ ja_name }} — Method Comparison</title>
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
  .nav { margin-bottom: 16px; display: flex; gap: 4px; flex-wrap: wrap; }
  .nav a { padding: 3px 7px; background: #0f3460; color: #8899aa; border-radius: 4px;
           text-decoration: none; font-size: 10px; }
  .nav a:hover { background: #16213e; color: #e0e0e0; }
  .nav a.current { background: #e94560; color: white; }
  .stats-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #16213e; border-radius: 8px; padding: 14px; border: 1px solid #0f3460; }
  .stat-card h3 { font-size: 14px; color: #e94560; margin-bottom: 8px; }
  .stat-card .detail { font-size: 13px; color: #8899aa; line-height: 1.6; }
  .stat-card .detail strong { color: #e0e0e0; }
  .method-section { margin-bottom: 32px; }
  .method-section h2 { font-size: 16px; color: #e94560; margin-bottom: 12px;
                       padding-bottom: 6px; border-bottom: 1px solid #0f3460; }
  .segment-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
  .segment-card { background: #16213e; border-radius: 8px; padding: 10px;
                  border: 1px solid #0f3460; }
  .segment-card .seg-info { font-size: 11px; color: #667788; margin-bottom: 6px;
                            display: flex; justify-content: space-between; }
  .segment-card .seg-info .quality { color: #e94560; font-weight: 600; }
  .segment-card img { width: 300px; height: 80px; border-radius: 4px; display: block;
                      background: #0a0a1a; object-fit: cover; }
  .segment-card audio { width: 100%; height: 32px; margin-top: 6px; }
  .no-segments { color: #445566; font-style: italic; padding: 20px; }
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
  <div class="nav">
    {% for sp in all_species %}
    <a href="/species/{{ sp }}" {% if sp == species_code %}class="current"{% endif %}
       title="{{ species_names.get(sp, {}).get('ja', sp) }}">{{ species_names.get(sp, {}).get('ja', '') or sp }}</a>
    {% endfor %}
  </div>

  <div class="stats-row">
    {% for method_key, info in methods.items() %}
    <div class="stat-card">
      <h3>{{ info.label }}</h3>
      <div class="detail">
        <strong>{{ stats[method_key].count }}</strong> segments<br>
        Avg duration: <strong>{{ stats[method_key].avg_duration }}</strong><br>
        Recordings: <strong>{{ stats[method_key].n_recordings }}</strong>
      </div>
    </div>
    {% endfor %}
  </div>

  {% for method_key, info in methods.items() %}
  <div class="method-section">
    <h2>{{ info.label }} ({{ segments[method_key] | length }} samples{% if segments[method_key] | length >= max_samples %}, top {{ max_samples }} by {{ info.quality_col }}{% endif %})</h2>
    {% if segments[method_key] %}
    <div class="segment-grid">
      {% for seg in segments[method_key] %}
      <div class="segment-card">
        <div class="seg-info">
          <span>{{ seg.safe_id }} | {{ seg.onset }}–{{ seg.offset }}s ({{ seg.duration }}s)</span>
          <span class="quality">{{ seg.quality_label }}: {{ seg.quality }}</span>
        </div>
        <img src="/spectrogram/{{ method_key }}/{{ species_code }}/{{ seg.filename }}" loading="lazy"
             alt="spectrogram" width="300" height="80">
        <audio controls preload="none">
          <source src="/audio/{{ method_key }}/{{ species_code }}/{{ seg.filename }}" type="audio/wav">
        </audio>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div class="no-segments">No segments available for this method.</div>
    {% endif %}
  </div>
  {% endfor %}
</div>
</body>
</html>
""")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Method Comparison Viewer")

_results: dict[str, pd.DataFrame] = {}
_species_index: dict[str, dict] = {}
_species_names: dict = {}


@app.on_event("startup")
def startup():
    global _results, _species_index, _species_names
    print("Loading method results...", flush=True)
    _results = load_method_results()
    _species_index = build_species_index(_results)
    _species_names = load_species_names()
    for mk, df in _results.items():
        print(f"  {mk}: {len(df)} rows", flush=True)
    print(f"  {len(_species_index)} species found", flush=True)
    print("Ready.", flush=True)


@app.get("/", response_class=HTMLResponse)
def index():
    totals = {}
    n_species_with = {}
    for mk in METHODS:
        df = _results.get(mk, pd.DataFrame())
        totals[mk] = len(df) if not df.empty else 0
        n_species_with[mk] = (
            df["species_code"].nunique()
            if not df.empty and "species_code" in df.columns
            else 0
        )

    return INDEX_TEMPLATE.render(
        n_species=len(_species_index),
        species_index=_species_index,
        species_names=_species_names,
        methods=METHODS,
        totals=totals,
        n_species_with=n_species_with,
    )


@app.get("/species/{species_code}", response_class=HTMLResponse)
def species_page(species_code: str):
    names = _species_names.get(species_code, {})
    ja_name = names.get("ja", "") or species_code
    sci_name = names.get("sci", "")

    stats = {}
    segments = {}
    for mk in METHODS:
        stats[mk] = get_species_stats(species_code, mk, _results)
        segments[mk] = get_species_segments(species_code, mk, _results)

    return SPECIES_TEMPLATE.render(
        species_code=species_code,
        ja_name=ja_name,
        sci_name=sci_name,
        methods=METHODS,
        stats=stats,
        segments=segments,
        all_species=sorted(_species_index.keys()),
        species_names=_species_names,
        max_samples=MAX_SAMPLES,
    )


@app.get("/audio/{method}/{species}/{filename}")
def serve_audio(method: str, species: str, filename: str):
    """Serve a WAV segment file."""
    info = METHODS.get(method)
    if info is None:
        return Response(status_code=404)
    filepath = info["dir"] / species / filename
    if not filepath.exists():
        return Response(status_code=404)
    return Response(content=filepath.read_bytes(), media_type="audio/wav")


@app.get("/spectrogram/{method}/{species}/{filename}")
def serve_spectrogram(method: str, species: str, filename: str):
    """Generate and serve a mel spectrogram PNG for a WAV segment."""
    info = METHODS.get(method)
    if info is None:
        return Response(status_code=404)
    filepath = info["dir"] / species / filename
    png_data = generate_spectrogram_png(str(filepath))
    if png_data is None:
        return Response(status_code=404)
    return Response(content=png_data, media_type="image/png")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Method comparison viewer")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting method viewer at http://localhost:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
