"""
Segmentation results viewer — lightweight web app.

Browse and compare segmentation results across methods (signal-processing
baseline, PANNs, TweetyNet, Bout pipeline) for each species and recording.
Includes Japanese species names and audio playback.

Usage:
    python results_viewer.py [--port 8050]
"""

import argparse
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from jinja2 import Template

STEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEP_DIR.parent.parent
NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"

# Result directories
VIS_DIRS = {
    "signal_processing": NAS_BASE / "segments" / "test_samples_results_v2",
    "panns": NAS_BASE / "segments" / "test_samples_results_panns",
    "tweetynet": NAS_BASE / "segments" / "test_samples_results_tweetynet",
    "bouts": NAS_BASE / "segments" / "test_samples_results_bouts",
}

METHOD_LABELS = {
    "signal_processing": "Signal Processing (7 methods)",
    "panns": "PANNs (zero-shot SED)",
    "tweetynet": "TweetyNet (pseudo-label)",
    "bouts": "Bout Pipeline (TweetyNet + BirdNET)",
}

# ---------------------------------------------------------------------------
# Species name mapping
# ---------------------------------------------------------------------------

def load_species_names() -> dict:
    """Load ebird_species_code -> japanese_name mapping."""
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
            mapping[code] = {"ja": ja if pd.notna(ja) else "", "sci": sci if pd.notna(sci) else ""}
    return mapping


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_results() -> dict:
    """Load all result CSVs and merge into a unified structure."""
    data = {}

    sp_csv = STEP_DIR / "prototype_results_v2.csv"
    if sp_csv.exists():
        sp = pd.read_csv(sp_csv)
        sp_agg = sp.groupby(["recording_id", "ebird_species_code"]).agg(
            sp_methods=("method", "nunique"),
            sp_total_segments=("n_segments", "sum"),
            sp_mean_segments_per_method=("n_segments", "mean"),
            sp_mean_duration=("mean_segment_sec", "mean"),
        ).reset_index()
        data["signal_processing"] = sp_agg

    panns_csv = STEP_DIR / "prototype_results_panns.csv"
    if panns_csv.exists():
        data["panns"] = pd.read_csv(panns_csv)

    tw_csv = STEP_DIR / "tweetynet_results.csv"
    if tw_csv.exists():
        data["tweetynet"] = pd.read_csv(tw_csv)

    bout_csv = STEP_DIR / "classify_results.csv"
    if bout_csv.exists():
        data["bouts"] = pd.read_csv(bout_csv)

    return data


def build_species_index(data: dict) -> dict:
    """Build {species_code: [recording_ids]} from available data."""
    all_recs = set()
    for key, df in data.items():
        sp_col = "ebird_species_code" if "ebird_species_code" in df.columns else "species_code"
        for _, row in df.iterrows():
            sp = row[sp_col]
            rid = row["recording_id"]
            all_recs.add((sp, rid))

    species_recs = {}
    for sp, rid in sorted(all_recs):
        species_recs.setdefault(sp, []).append(rid)
    return species_recs


def load_test_samples() -> pd.DataFrame:
    """Load test_samples.csv for audio file path resolution."""
    csv_path = STEP_DIR / "test_samples.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def find_audio_path(species_code: str, recording_id: str, samples_df: pd.DataFrame) -> str | None:
    """Find audio file path for a recording."""
    if samples_df.empty:
        return None
    row = samples_df[
        (samples_df["ebird_species_code"] == species_code) &
        (samples_df["recording_id"] == recording_id)
    ]
    if len(row) == 0:
        return None
    original_path = Path(row.iloc[0]["file_path"])
    audio_path = TEST_SAMPLES_DIR / species_code / original_path.name
    if audio_path.exists():
        return str(audio_path)
    return None


def find_image(method_key: str, species_code: str, recording_id: str) -> str | None:
    """Find visualization image path for a given method/species/recording."""
    vis_dir = VIS_DIRS.get(method_key)
    if vis_dir is None:
        return None
    species_dir = vis_dir / species_code
    if not species_dir.exists():
        return None

    safe_id = recording_id.replace(":", "_").replace("/", "_")
    patterns = [
        f"{safe_id}_comparison.png",
        f"{safe_id}_panns.png",
        f"{safe_id}_tweetynet.png",
        f"{safe_id}_bouts.png",
    ]
    for pat in patterns:
        p = species_dir / pat
        if p.exists():
            return str(p)

    matches = list(species_dir.glob(f"{safe_id}*.png"))
    if matches:
        return str(matches[0])
    return None


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

INDEX_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Segmentation Results Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; }
  .header { background: #2c3e50; color: white; padding: 16px 24px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header p { font-size: 13px; color: #bdc3c7; margin-top: 4px; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .summary-card { background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .summary-card h3 { font-size: 13px; color: #7f8c8d; margin-bottom: 8px; }
  .summary-card .value { font-size: 28px; font-weight: 700; color: #2c3e50; }
  .summary-card .sub { font-size: 12px; color: #95a5a6; margin-top: 4px; }
  .species-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px; }
  .species-card { background: white; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                  text-decoration: none; color: inherit; transition: transform 0.1s; }
  .species-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  .species-card .ja-name { font-size: 17px; font-weight: 700; color: #2c3e50; }
  .species-card .code { font-size: 12px; color: #7f8c8d; margin-top: 2px; }
  .species-card .info { font-size: 12px; color: #95a5a6; margin-top: 4px; }
  .filter-bar { margin-bottom: 16px; }
  .filter-bar input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; width: 300px; font-size: 14px; }
</style>
</head>
<body>
<div class="header">
  <h1>Segmentation Results Viewer</h1>
  <p>Step 03 — {{ n_species }} species, {{ n_recordings }} recordings, 4 methods</p>
</div>
<div class="container">
  <div class="summary">
    <div class="summary-card">
      <h3>Signal Processing</h3>
      <div class="value">{{ sp_stats.total_segments }}</div>
      <div class="sub">segments (7 methods combined), {{ sp_stats.mean_per_file }} mean/file</div>
    </div>
    <div class="summary-card">
      <h3>PANNs (zero-shot)</h3>
      <div class="value">{{ panns_stats.total_segments }}</div>
      <div class="sub">segments, {{ panns_stats.mean_per_file }} mean/file, {{ panns_stats.mean_dur }}s mean dur</div>
    </div>
    <div class="summary-card">
      <h3>TweetyNet (pseudo-label)</h3>
      <div class="value">{{ tw_stats.total_segments }}</div>
      <div class="sub">segments, {{ tw_stats.mean_per_file }} mean/file</div>
    </div>
    <div class="summary-card">
      <h3>Bout Pipeline</h3>
      <div class="value">{{ bout_stats.total_bouts }}</div>
      <div class="sub">bouts: {{ bout_stats.accept }} accept, {{ bout_stats.review }} review, {{ bout_stats.reject }} reject</div>
    </div>
  </div>

  <div class="filter-bar">
    <input type="text" id="filter" placeholder="Filter species (code or Japanese name)..." oninput="filterSpecies()">
  </div>

  <div class="species-grid" id="grid">
    {% for sp, recs in species_recs.items() %}
    <a href="/species/{{ sp }}" class="species-card" data-code="{{ sp }}" data-ja="{{ species_names.get(sp, {}).get('ja', '') }}" data-sci="{{ species_names.get(sp, {}).get('sci', '') }}">
      <div class="ja-name">{{ species_names.get(sp, {}).get('ja', '') or sp }}</div>
      <div class="code">{{ sp }} — <em>{{ species_names.get(sp, {}).get('sci', '') }}</em></div>
      <div class="info">{{ recs | length }} recordings</div>
    </a>
    {% endfor %}
  </div>
</div>
<script>
function filterSpecies() {
  const q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('.species-card').forEach(c => {
    const text = (c.dataset.code + ' ' + c.dataset.ja + ' ' + c.dataset.sci).toLowerCase();
    c.style.display = text.includes(q) ? '' : 'none';
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
<title>{{ ja_name }} ({{ species_code }}) — Segmentation Results</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; }
  .header { background: #2c3e50; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
  .header a { color: #3498db; text-decoration: none; font-size: 14px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header .sub { font-size: 13px; color: #bdc3c7; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  .recording { background: white; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }
  .recording-header { padding: 12px 16px; background: #ecf0f1; font-weight: 600; font-size: 14px;
                       display: flex; justify-content: space-between; align-items: center; cursor: pointer; }
  .recording-header:hover { background: #d5dbdb; }
  .recording-body { padding: 16px; }
  .method-row { margin-bottom: 16px; }
  .method-row h3 { font-size: 13px; color: #7f8c8d; margin-bottom: 8px; }
  .method-row img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
  .no-image { color: #bdc3c7; font-style: italic; font-size: 13px; }
  .stats-table { width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; }
  .stats-table th { text-align: left; padding: 6px 12px; background: #f8f9fa; border-bottom: 2px solid #dee2e6; }
  .stats-table td { padding: 6px 12px; border-bottom: 1px solid #eee; }
  .nav { margin-bottom: 16px; display: flex; gap: 6px; flex-wrap: wrap; }
  .nav a { padding: 4px 8px; background: #3498db; color: white; border-radius: 4px; text-decoration: none; font-size: 11px; }
  .nav a:hover { background: #2980b9; }
  .nav a.current { background: #e74c3c; }
  .collapsed .recording-body { display: none; }
  .audio-player { margin: 8px 0 12px 0; }
  .audio-player audio { width: 100%; height: 36px; }
  .other-methods { display: none; margin-top: 12px; }
  .other-methods.open { display: block; }
  .toggle-others { background: none; border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px 10px;
                   font-size: 12px; color: #7f8c8d; cursor: pointer; margin-top: 8px; }
  .toggle-others:hover { background: #ecf0f1; }
  .spec-container { position: relative; display: inline-block; width: 100%; cursor: crosshair; }
  .spec-container img { display: block; }
  .seek-line { position: absolute; top: 0; bottom: 0; width: 2px; background: rgba(231, 76, 60, 0.85);
               pointer-events: none; display: none; z-index: 10; box-shadow: 0 0 4px rgba(231, 76, 60, 0.5); }
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

  <table class="stats-table">
    <tr>
      <th>Recording</th>
      <th>SP Segments</th>
      <th>PANNs</th>
      <th>TweetyNet</th>
      <th>Bouts (A/R/J)</th>
    </tr>
    {% for rec in recordings %}
    <tr>
      <td><strong>{{ rec.recording_id }}</strong></td>
      <td>{{ rec.sp_segments }}</td>
      <td>{{ rec.panns_segments }}</td>
      <td>{{ rec.tw_segments }}</td>
      <td>{{ rec.bout_count }} ({{ rec.bout_verdict }})</td>
    </tr>
    {% endfor %}
  </table>

  {% for rec in recordings %}
  {% set rec_idx = loop.index %}
  <div class="recording" id="rec-{{ rec_idx }}">
    <div class="recording-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span>{{ rec.recording_id }}</span>
      <span style="font-weight:normal;color:#7f8c8d;">click to toggle</span>
    </div>
    <div class="recording-body">
      {% if rec.audio_url %}
      <div class="audio-player">
        <audio controls preload="none">
          <source src="{{ rec.audio_url }}" type="audio/mpeg">
        </audio>
      </div>
      {% endif %}
      <div class="method-row">
        <h3>{{ methods.bouts }}</h3>
        {% if rec.images.bouts %}
        <div class="spec-container" data-rec="{{ rec_idx }}">
          <div class="seek-line"></div>
          <img src="/image?path={{ rec.images.bouts }}" loading="lazy" alt="{{ methods.bouts }}">
        </div>
        {% else %}
        <div class="no-image">No visualization available</div>
        {% endif %}
      </div>
      <button class="toggle-others" onclick="this.nextElementSibling.classList.toggle('open'); this.textContent = this.nextElementSibling.classList.contains('open') ? 'Hide other methods' : 'Show other methods (SP / PANNs / TweetyNet)';">Show other methods (SP / PANNs / TweetyNet)</button>
      <div class="other-methods">
        {% for method_key in ['signal_processing', 'panns', 'tweetynet'] %}
        <div class="method-row">
          <h3>{{ methods[method_key] }}</h3>
          {% if rec.images[method_key] %}
          <div class="spec-container" data-rec="{{ rec_idx }}">
            <div class="seek-line"></div>
            <img src="/image?path={{ rec.images[method_key] }}" loading="lazy" alt="{{ methods[method_key] }}">
          </div>
          {% else %}
          <div class="no-image">No visualization available</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </div>
  </div>
  {% endfor %}
</div>
<script>
// Seek-line sync: map audio currentTime to spectrogram x-position.
// Plot area margins as fraction of image width (approximate for tight_layout matplotlib).
const PLOT_LEFT = 0.10;
const PLOT_RIGHT = 0.98;

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.recording').forEach((card, idx) => {
    const recIdx = idx + 1;
    const audio = card.querySelector('audio');
    if (!audio) return;

    const containers = card.querySelectorAll('.spec-container');

    // Click on spectrogram to seek
    containers.forEach(c => {
      c.addEventListener('click', e => {
        if (!audio.duration) return;
        const rect = c.getBoundingClientRect();
        const xFrac = (e.clientX - rect.left) / rect.width;
        const timeFrac = Math.max(0, Math.min(1, (xFrac - PLOT_LEFT) / (PLOT_RIGHT - PLOT_LEFT)));
        audio.currentTime = timeFrac * audio.duration;
        if (audio.paused) audio.play();
      });
    });

    // Update seek line position
    function updateSeekLines() {
      if (!audio.duration) return;
      const frac = audio.currentTime / audio.duration;
      const xPercent = (PLOT_LEFT + frac * (PLOT_RIGHT - PLOT_LEFT)) * 100;
      containers.forEach(c => {
        const line = c.querySelector('.seek-line');
        line.style.left = xPercent + '%';
        line.style.display = audio.paused && audio.currentTime === 0 ? 'none' : 'block';
      });
    }

    audio.addEventListener('timeupdate', updateSeekLines);
    audio.addEventListener('seeked', updateSeekLines);
    audio.addEventListener('play', () => {
      containers.forEach(c => c.querySelector('.seek-line').style.display = 'block');
    });
    audio.addEventListener('ended', () => {
      containers.forEach(c => c.querySelector('.seek-line').style.display = 'none');
    });
  });
});
</script>
</body>
</html>
""")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Segmentation Results Viewer")

_data: dict = {}
_species_recs: dict = {}
_species_names: dict = {}
_samples_df: pd.DataFrame = pd.DataFrame()


@app.on_event("startup")
def startup():
    global _data, _species_recs, _species_names, _samples_df
    _data = load_results()
    _species_recs = build_species_index(_data)
    _species_names = load_species_names()
    _samples_df = load_test_samples()


@app.get("/", response_class=HTMLResponse)
def index():
    sp_df = _data.get("signal_processing")
    panns_df = _data.get("panns")
    tw_df = _data.get("tweetynet")
    bout_df = _data.get("bouts")

    sp_stats = {
        "total_segments": int(sp_df["sp_total_segments"].sum()) if sp_df is not None else 0,
        "mean_per_file": f"{sp_df['sp_total_segments'].mean():.1f}" if sp_df is not None else "N/A",
    }
    panns_stats = {
        "total_segments": int(panns_df["n_segments"].sum()) if panns_df is not None else 0,
        "mean_per_file": f"{panns_df['n_segments'].mean():.1f}" if panns_df is not None else "N/A",
        "mean_dur": f"{panns_df['mean_segment_dur'].mean():.2f}" if panns_df is not None and "mean_segment_dur" in panns_df.columns else "N/A",
    }
    tw_stats = {
        "total_segments": int(tw_df["n_segments"].sum()) if tw_df is not None else 0,
        "mean_per_file": f"{tw_df['n_segments'].mean():.1f}" if tw_df is not None else "N/A",
    }
    bout_stats = {
        "total_bouts": int(bout_df["n_bouts"].sum()) if bout_df is not None else 0,
        "accept": int(bout_df["n_accept"].sum()) if bout_df is not None else 0,
        "review": int(bout_df["n_review"].sum()) if bout_df is not None else 0,
        "reject": int(bout_df["n_reject"].sum()) if bout_df is not None else 0,
    }

    return INDEX_TEMPLATE.render(
        n_species=len(_species_recs),
        n_recordings=sum(len(v) for v in _species_recs.values()),
        species_recs=_species_recs,
        species_names=_species_names,
        sp_stats=sp_stats,
        panns_stats=panns_stats,
        tw_stats=tw_stats,
        bout_stats=bout_stats,
    )


@app.get("/species/{species_code}", response_class=HTMLResponse)
def species_page(species_code: str):
    recs = _species_recs.get(species_code, [])
    names = _species_names.get(species_code, {})
    ja_name = names.get("ja", "") or species_code
    sci_name = names.get("sci", "")

    sp_df = _data.get("signal_processing")
    panns_df = _data.get("panns")
    tw_df = _data.get("tweetynet")
    bout_df = _data.get("bouts")

    recordings = []
    for rid in recs:
        rec_info = {"recording_id": rid, "images": {}}

        # Signal processing
        if sp_df is not None:
            sp_row = sp_df[(sp_df["recording_id"] == rid) & (sp_df["ebird_species_code"] == species_code)]
            rec_info["sp_segments"] = int(sp_row["sp_total_segments"].iloc[0]) if len(sp_row) > 0 else "-"
        else:
            rec_info["sp_segments"] = "-"

        # PANNs
        if panns_df is not None:
            p_row = panns_df[(panns_df["recording_id"] == rid) & (panns_df["ebird_species_code"] == species_code)]
            rec_info["panns_segments"] = int(p_row["n_segments"].iloc[0]) if len(p_row) > 0 else "-"
        else:
            rec_info["panns_segments"] = "-"

        # TweetyNet
        if tw_df is not None:
            t_row = tw_df[(tw_df["recording_id"] == rid) & (tw_df["ebird_species_code"] == species_code)]
            rec_info["tw_segments"] = int(t_row["n_segments"].iloc[0]) if len(t_row) > 0 else "-"
        else:
            rec_info["tw_segments"] = "-"

        # Bouts
        if bout_df is not None:
            b_row = bout_df[(bout_df["recording_id"] == rid) & (bout_df["species_code"] == species_code)]
            if len(b_row) > 0:
                rec_info["bout_count"] = int(b_row["n_bouts"].iloc[0])
                a, rv, rj = int(b_row["n_accept"].iloc[0]), int(b_row["n_review"].iloc[0]), int(b_row["n_reject"].iloc[0])
                rec_info["bout_verdict"] = f"{a}/{rv}/{rj}"
            else:
                rec_info["bout_count"] = "-"
                rec_info["bout_verdict"] = "-"
        else:
            rec_info["bout_count"] = "-"
            rec_info["bout_verdict"] = "-"

        # Audio path
        audio_path = find_audio_path(species_code, rid, _samples_df)
        rec_info["audio_url"] = f"/audio?path={audio_path}" if audio_path else ""

        # Images
        for method_key in VIS_DIRS:
            rec_info["images"][method_key] = find_image(method_key, species_code, rid) or ""

        recordings.append(rec_info)

    return SPECIES_TEMPLATE.render(
        species_code=species_code,
        ja_name=ja_name,
        sci_name=sci_name,
        recordings=recordings,
        methods=METHOD_LABELS,
        all_species=sorted(_species_recs.keys()),
        species_names=_species_names,
    )


@app.get("/image")
def serve_image(path: str):
    """Serve an image file from the NAS."""
    p = Path(path)
    if not str(p).startswith(str(NAS_BASE)):
        return Response(status_code=403)
    if not p.exists():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/png")


@app.get("/audio")
def serve_audio(path: str):
    """Serve an audio file from the NAS."""
    p = Path(path)
    if not str(p).startswith(str(NAS_BASE)):
        return Response(status_code=403)
    if not p.exists():
        return Response(status_code=404)
    suffix = p.suffix.lower()
    media_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac"}
    media_type = media_types.get(suffix, "audio/mpeg")
    return Response(content=p.read_bytes(), media_type=media_type)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Segmentation results viewer")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting viewer at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
