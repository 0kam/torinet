"""データ収集のカバレッジ可視化スクリプト

XC + iNat S3 + iNat API メタデータから種ごとの録音数を集計し、
国内/国外の内訳・分布・不足種を可視化する。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Font setup (Japanese)
# ---------------------------------------------------------------------------
JA_FONTS = ["Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans"]
_font_set = False
for f in JA_FONTS:
    try:
        matplotlib.font_manager.FontProperties(family=f).get_name()
        plt.rcParams["font.family"] = f
        _font_set = True
        break
    except Exception:
        continue
if not _font_set:
    print("Warning: Japanese font not found, falling back to default", file=sys.stderr)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NAS = Path.home() / "NAS" / "nasbi" / "ToriNET"
XC_PARQUET = NAS / "audio" / "xeno-canto" / "metadata" / "xc_metadata.parquet"
INAT_PARQUET = NAS / "audio" / "inat-sounds" / "annotations" / "inat_metadata.parquet"
INAT_API_PARQUET = NAS / "audio" / "inat-api" / "metadata" / "inat_api_metadata.parquet"
SPECIES_CSV = Path(__file__).resolve().parent.parent / "01_species_list" / "species_list.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
xc = pd.read_parquet(XC_PARQUET, columns=["ebird_species_code", "is_japan"])
inat = pd.read_parquet(INAT_PARQUET, columns=["ebird_species_code", "is_japan"])
inat_api = pd.read_parquet(INAT_API_PARQUET, columns=["ebird_species_code", "is_japan"])
species = pd.read_csv(SPECIES_CSV)

# Only eBird-matched species (688)
species_matched = species[species["ebird_matched"] == True].copy()
all_codes = set(species_matched["ebird_species_code"].dropna())

# ---------------------------------------------------------------------------
# Aggregate per-species counts: japan / international, by source
# ---------------------------------------------------------------------------
def count_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with columns [ebird_species_code, japan, international]."""
    grouped = df.groupby(["ebird_species_code", "is_japan"]).size().unstack(fill_value=0)
    grouped.columns = ["international", "japan"] if False in grouped.columns else ["japan"]
    if "international" not in grouped.columns:
        grouped["international"] = 0
    if "japan" not in grouped.columns:
        grouped["japan"] = 0
    return grouped[["japan", "international"]].reset_index()

xc_counts = count_by_location(xc)
inat_counts = count_by_location(inat)
inat_api_counts = count_by_location(inat_api)

# Merge XC + iNat S3 + iNat API (sum japan and international separately)
combined = pd.merge(
    xc_counts, inat_counts,
    on="ebird_species_code", how="outer", suffixes=("_xc", "_inat"),
).fillna(0)
combined = pd.merge(
    combined, inat_api_counts,
    on="ebird_species_code", how="outer", suffixes=("", "_inat_api"),
).fillna(0)
# Rename iNat API columns for clarity
combined = combined.rename(columns={"japan": "japan_inat_api", "international": "international_inat_api"})
combined["japan"] = combined["japan_xc"] + combined["japan_inat"] + combined["japan_inat_api"]
combined["international"] = combined["international_xc"] + combined["international_inat"] + combined["international_inat_api"]
combined["total"] = combined["japan"] + combined["international"]

# Add species with zero recordings
missing_codes = all_codes - set(combined["ebird_species_code"])
if missing_codes:
    missing_df = pd.DataFrame({"ebird_species_code": list(missing_codes)})
    for c in ["japan_xc", "international_xc", "japan_inat", "international_inat",
              "japan", "international", "total"]:
        missing_df[c] = 0
    combined = pd.concat([combined, missing_df], ignore_index=True)

# Merge names from species list
combined = combined.merge(
    species_matched[["ebird_species_code", "japanese_name", "scientific_name"]],
    on="ebird_species_code", how="left",
)
combined["label"] = combined.apply(
    lambda r: f"{r['japanese_name']}  ({r['scientific_name']})" if pd.notna(r["japanese_name"]) else r["scientific_name"],
    axis=1,
)
combined = combined.sort_values("total", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Summary stats (stdout)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Data Collection Coverage Summary")
print("=" * 60)
print(f"  Target species (eBird-matched):  {len(all_codes)}")
print(f"  Species with ≥1 recording:       {(combined['total'] > 0).sum()}")
print(f"  Species with 0 recordings:       {(combined['total'] == 0).sum()}")
print(f"  Coverage:                         {(combined['total'] > 0).sum() / len(all_codes) * 100:.1f}%")
print()
print(f"  Total recordings (XC+iNat+iNat API): {int(combined['total'].sum()):,}")
print(f"    XC:       {int(xc.shape[0]):,}")
print(f"    iNat S3:  {int(inat.shape[0]):,}")
print(f"    iNat API: {int(inat_api.shape[0]):,}")
print()
print(f"  Japan recordings:                {int(combined['japan'].sum()):,}")
print(f"  International recordings:        {int(combined['international'].sum()):,}")
print()
print(f"  Per-species recording count:")
print(f"    Mean:   {combined['total'].mean():.1f}")
print(f"    Median: {combined['total'].median():.0f}")
print(f"    Max:    {int(combined['total'].max()):,}  ({combined.iloc[0]['label']})")
print(f"    Min:    {int(combined['total'].min())}")
q = combined["total"].quantile([0.05, 0.25, 0.75, 0.95])
print(f"    Q05:    {q[0.05]:.0f}")
print(f"    Q25:    {q[0.25]:.0f}")
print(f"    Q75:    {q[0.75]:.0f}")
print(f"    Q95:    {q[0.95]:.0f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Figure 1: Horizontal bar chart — top 50 + bottom 50
# ---------------------------------------------------------------------------
top50 = combined.head(50).copy()
bottom50 = combined.tail(50).copy()

def plot_barh(subset: pd.DataFrame, title: str, ax: plt.Axes):
    subset = subset.sort_values("total", ascending=True)
    y = np.arange(len(subset))
    ax.barh(y, subset["japan"].values, color="#d62728", label="Japan")
    ax.barh(y, subset["international"].values, left=subset["japan"].values,
            color="#1f77b4", label="International")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"].values, fontsize=5)
    ax.set_xlabel("Recording count")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(18, 16))
plot_barh(top50, "Top 50 species by recording count", axes[0])
plot_barh(bottom50, "Bottom 50 species by recording count", axes[1])
fig.suptitle("Per-species recording count (XC + iNat S3 + iNat API, Japan vs International)",
             fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "species_recording_counts.png")
print(f"Saved: {FIG_DIR / 'species_recording_counts.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Histogram — log10(recording count) distribution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
counts_nonzero = combined.loc[combined["total"] > 0, "total"]
log_counts = np.log10(counts_nonzero.values)
bins = np.arange(0, np.ceil(log_counts.max()) + 0.25, 0.25)
ax.hist(log_counts, bins=bins, color="#2ca02c", edgecolor="white", alpha=0.85)
ax.set_xlabel("log10(recording count)")
ax.set_ylabel("Number of species")
ax.set_title("Distribution of per-species recording count (XC + iNat S3 + iNat API)")

# Add annotation for zero-recording species
n_zero = (combined["total"] == 0).sum()
if n_zero > 0:
    ax.annotate(f"{n_zero} species with\n0 recordings",
                xy=(0, 0), xytext=(0.5, ax.get_ylim()[1] * 0.8),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red")

# Add vertical lines for quartiles
for q_val, label in [(np.log10(combined["total"].median()), "median"),
                      (np.log10(max(q[0.25], 1)), "Q25"),
                      (np.log10(max(q[0.75], 1)), "Q75")]:
    ax.axvline(q_val, color="gray", linestyle="--", alpha=0.7, linewidth=0.8)
    ax.text(q_val, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=7, color="gray")

fig.tight_layout()
fig.savefig(FIG_DIR / "recording_count_distribution.png")
print(f"Saved: {FIG_DIR / 'recording_count_distribution.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: Bottom-50 species table
# ---------------------------------------------------------------------------
bottom50_display = combined.tail(50)[["japanese_name", "scientific_name", "ebird_species_code",
                                       "japan", "international", "total"]].copy()
bottom50_display = bottom50_display.sort_values("total", ascending=True).reset_index(drop=True)
bottom50_display.index += 1
bottom50_display.columns = ["和名", "学名", "eBird Code", "Japan", "International", "Total"]

fig, ax = plt.subplots(figsize=(14, 16))
ax.axis("off")
table = ax.table(
    cellText=bottom50_display.values,
    colLabels=bottom50_display.columns,
    rowLabels=bottom50_display.index,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.auto_set_column_width(col=list(range(len(bottom50_display.columns))))
# Header style
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#D9E2F3")

ax.set_title("Bottom 50 species by recording count", fontsize=14, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / "bottom50_species_table.png")
print(f"Saved: {FIG_DIR / 'bottom50_species_table.png'}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Print bottom-50 to stdout too
# ---------------------------------------------------------------------------
print("\n--- Bottom 50 species (fewest recordings) ---")
print(bottom50_display.to_string())
print()
