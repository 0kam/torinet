"""
録音データと日本国内オカレンスの比較分析。

eBird頻度データ（日本国内）と各ソースの録音数（全世界）を突き合わせ、
訓練データが不足している種を特定する。

優先度ティア（downloadable recordings 基準）:
  P1 (< 50 downloadable recordings) — 最優先
  P2 (50-99 downloadable recordings) — 次に優先
  P3 (≥ 100 downloadable recordings) — 十分

使い方:
  python analyze_coverage_gaps.py

出力:
  - coverage_gap_analysis.csv   — 全種の頻度・録音数・優先度ティア
  - figures/coverage_gap_*.png  — 可視化
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import get_target_species, load_config, nas_path

STEP_DIR = Path(__file__).resolve().parent
FIG_DIR = STEP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_all_counts(cfg: dict) -> pd.DataFrame:
    """全ソースの種ごと録音数を集計する。"""

    def _count(df, prefix):
        total = df.groupby("ebird_species_code").size().rename(f"{prefix}_total")
        result = total.to_frame()
        if "is_japan" in df.columns:
            jp = (
                df[df["is_japan"] == True]  # noqa: E712
                .groupby("ebird_species_code")
                .size()
                .rename(f"{prefix}_japan")
            )
            result = pd.concat([result, jp], axis=1)
        return result.fillna(0).astype(int)

    sources = {}
    paths = {
        "xc": nas_path(cfg, "audio/xeno-canto/metadata/xc_metadata.parquet"),
        "inat": nas_path(cfg, "audio/inat-sounds/annotations/inat_metadata.parquet"),
        "inat_api": nas_path(cfg, "audio/inat-api/metadata/inat_api_metadata.parquet"),
        "ml": nas_path(cfg, "audio/macaulay/metadata/ml_metadata.parquet"),
    }

    frames = []
    for name, path in paths.items():
        if path.exists():
            df = pq.read_table(str(path)).to_pandas()
            frames.append(_count(df, name))
            sources[name] = len(df)
            print(f"  {name:10s}: {len(df):>10,} recordings")
        else:
            print(f"  {name:10s}: (not found)")

    counts = pd.concat(frames, axis=1).fillna(0).astype(int)
    return counts


def load_ml_downloaded_counts() -> pd.DataFrame:
    """ML申請バッチCSVからダウンロード済み録音数を種別に集計する。"""
    reqdir = STEP_DIR / "ml_request"
    species_total = {}
    species_japan = {}

    batch_files = sorted(reqdir.glob("ml_request_batch_*.csv"))
    # _ids.csv は除外
    batch_files = [f for f in batch_files if "_ids" not in f.name]

    for fpath in batch_files:
        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sc = row["ebird_species_code"]
                species_total[sc] = species_total.get(sc, 0) + 1
                if row.get("is_japan", "").lower() == "true":
                    species_japan[sc] = species_japan.get(sc, 0) + 1

    df = pd.DataFrame({
        "ebird_species_code": list(species_total.keys()),
        "ml_downloaded": [species_total[k] for k in species_total],
        "ml_downloaded_japan": [species_japan.get(k, 0) for k in species_total],
    })
    return df.set_index("ebird_species_code")


def compute_priority(merged: pd.DataFrame) -> pd.DataFrame:
    """優先度ティアと補助スコアを算出する。

    ティア（downloadable recordings 基準）:
    - P1: < 50  — 最優先
    - P2: 50-99 — 次に優先
    - P3: ≥ 100 — 十分

    補助スコア（同一ティア内の並べ替え用）:
    - priority_score = frequency / log2(downloadable + 2)
    - 頻度が高く、録音が少ない種ほどスコアが高い
    """
    df = merged.copy()

    # 合計録音数（メタデータ上の全録音）
    total_cols = [c for c in df.columns if c.endswith("_total")]
    japan_cols = [c for c in df.columns if c.endswith("_japan")
                  and not c.startswith("ml_downloaded")]
    df["recordings_total"] = df[total_cols].sum(axis=1)
    df["recordings_japan"] = df[japan_cols].sum(axis=1) if japan_cols else 0

    # DL可能な録音数 = XC + iNat S3 + iNat API + ML downloaded
    dl_cols = [c for c in total_cols if c != "ml_total"]
    df["recordings_downloadable"] = df[dl_cols].sum(axis=1)
    if "ml_downloaded" in df.columns:
        df["recordings_downloadable"] += df["ml_downloaded"]
        dl_japan = df[[c for c in japan_cols if c != "ml_japan"]].sum(axis=1)
        df["recordings_japan_downloadable"] = dl_japan + df["ml_downloaded_japan"]
    else:
        df["recordings_japan_downloadable"] = (
            df[japan_cols].sum(axis=1) if japan_cols else 0
        )

    freq = df["frequency_annual_mean"].fillna(0)

    # 優先度ティア（downloadable recordings 基準）
    df["priority_tier"] = pd.cut(
        df["recordings_downloadable"],
        bins=[-1, 49, 99, float("inf")],
        labels=["P1", "P2", "P3"],
    )

    # 補助スコア = 頻度 / log2(downloadable + 2)
    df["priority_score"] = freq / np.log2(df["recordings_downloadable"] + 2)

    # ML metadata 上の総録音数（未DL含む）
    df["ml_metadata_total"] = df.get("ml_total", 0)
    df["ml_metadata_japan"] = df.get("ml_japan", 0)

    return df




def plot_gap_analysis(df: pd.DataFrame):
    """ギャップ分析の可視化。"""
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "sans-serif"]

    regular = df[~df["residence_status"].isin(["undetected", "vagrant", ""])]
    rare = df[df["residence_status"].isin(["undetected", "vagrant"])]

    # --- Figure 1: 頻度 vs DL可能録音数 scatter（P1/P2/P3ティア帯付き）---
    fig, ax = plt.subplots(figsize=(12, 8))

    # ティア帯をハイライト
    ax.axhspan(0.5, 50, alpha=0.06, color="red", label="_P1 zone")
    ax.axhspan(50, 100, alpha=0.04, color="orange", label="_P2 zone")
    ax.axhline(y=50, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(y=100, color="orange", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(0.01, 25, "P1 (< 50)", fontsize=10, color="red", alpha=0.7)
    ax.text(0.01, 70, "P2 (50-99)", fontsize=10, color="orange", alpha=0.7)
    ax.text(0.01, 150, "P3 (≥ 100)", fontsize=10, color="green", alpha=0.5)

    colors = {
        "resident": "#2196F3",
        "summer": "#FF9800",
        "winter": "#00BCD4",
        "passage": "#9C27B0",
    }

    for status, color in colors.items():
        subset = regular[regular["residence_status"] == status]
        ax.scatter(
            subset["frequency_annual_mean"],
            subset["recordings_downloadable"] + 1,
            alpha=0.6, s=30, c=color, label=status,
        )

    ax.scatter(
        rare["frequency_annual_mean"],
        rare["recordings_downloadable"] + 1,
        alpha=0.3, s=15, c="gray", label="vagrant/undetected",
    )

    ax.set_xlabel("eBird Frequency (Japan annual mean)")
    ax.set_ylabel("Downloadable Recordings (log scale)")
    ax.set_yscale("log")
    ax.set_title("Recording Coverage vs eBird Frequency (Downloadable Recordings)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # P1種で高頻度のものにラベル
    p1_regular = regular[regular["priority_tier"] == "P1"]
    top_p1 = p1_regular.nlargest(15, "frequency_annual_mean")
    for _, row in top_p1.iterrows():
        ax.annotate(
            row["japanese_name"],
            (row["frequency_annual_mean"], row["recordings_downloadable"] + 1),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords="offset points",
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_gap_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  → {FIG_DIR / 'coverage_gap_scatter.png'}")

    # --- Figure 2: P1/P2種の録音数（ソース別スタック）---
    p1p2 = regular[regular["priority_tier"].isin(["P1", "P2"])].copy()
    tier_order = {"P1": 0, "P2": 1}
    p1p2["_tier_order"] = p1p2["priority_tier"].map(tier_order)
    p1p2 = p1p2.sort_values(
        ["_tier_order", "priority_score"], ascending=[True, False],
    )

    # 表示上限: 最大50種
    show = p1p2.head(50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 14))

    # 左: 全録音数（ソース別スタック）
    ax = axes[0]
    y_pos = range(len(show))
    labels = [
        f"[{r['priority_tier']}] {r['japanese_name']} ({r['ebird_species_code']})"
        for _, r in show.iterrows()
    ]

    total_cols_plot = []
    colors_stack = []
    names_stack = []
    for col, color, name in [
        ("xc_total", "#4CAF50", "XC"),
        ("inat_total", "#FF9800", "iNat S3"),
        ("inat_api_total", "#2196F3", "iNat API"),
        ("ml_downloaded", "#9C27B0", "ML (downloaded)"),
    ]:
        if col in show.columns:
            total_cols_plot.append(col)
            colors_stack.append(color)
            names_stack.append(name)

    left = np.zeros(len(show))
    for col, color, name in zip(total_cols_plot, colors_stack, names_stack):
        vals = show[col].values.astype(float)
        ax.barh(y_pos, vals, left=left, color=color, label=name, height=0.7)
        left += vals

    ax.axvline(x=50, color="red", linewidth=1, linestyle="--", alpha=0.6)
    ax.axvline(x=100, color="orange", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Downloadable Recordings")
    ax.set_title("P1/P2 Species: Downloadable Recordings by Source")
    ax.legend(loc="lower right", fontsize=8)

    # 右: eBird頻度
    ax = axes[1]
    tier_colors = [
        "#E53935" if r["priority_tier"] == "P1" else "#FF9800"
        for _, r in show.iterrows()
    ]
    ax.barh(y_pos, show["frequency_annual_mean"].values,
            color=tier_colors, alpha=0.7, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("eBird Frequency (Japan)")
    ax.set_title("P1/P2 Species: eBird Frequency")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_gap_top40.png", dpi=150)
    plt.close(fig)
    print(f"  → {FIG_DIR / 'coverage_gap_top40.png'}")

    # --- Figure 3: residence_status 別のティア分布 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    statuses = ["resident", "summer", "winter", "passage"]
    status_colors = ["#2196F3", "#FF9800", "#00BCD4", "#9C27B0"]

    summary_data = []
    for status in statuses:
        subset = df[df["residence_status"] == status]
        n = len(subset)
        p1 = (subset["priority_tier"] == "P1").sum()
        p2 = (subset["priority_tier"] == "P2").sum()
        p3 = (subset["priority_tier"] == "P3").sum()
        summary_data.append({
            "status": status,
            "n_species": n,
            "P1 (< 50)": p1,
            "P2 (50-99)": p2,
            "P3 (≥ 100)": p3,
        })

    summary = pd.DataFrame(summary_data)

    x = np.arange(len(statuses))
    width = 0.22
    bars_p1 = ax.bar(x - width, summary["P1 (< 50)"],
                     width, color="#E53935", label="P1 (< 50)")
    bars_p2 = ax.bar(x, summary["P2 (50-99)"],
                     width, color="#FF9800", label="P2 (50-99)")
    bars_p3 = ax.bar(x + width, summary["P3 (≥ 100)"],
                     width, color="#4CAF50", label="P3 (≥ 100)")

    ax.set_xticks(x)
    ax.set_xticklabels(statuses)
    ax.set_ylabel("Number of Species")
    ax.set_title("Priority Tier Distribution by Residence Status")
    ax.legend()

    for bars in [bars_p1, bars_p2, bars_p3]:
        for bar in bars:
            val = int(bar.get_height())
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(val), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "coverage_gap_by_status.png", dpi=150)
    plt.close(fig)
    print(f"  → {FIG_DIR / 'coverage_gap_by_status.png'}")


def main():
    cfg = load_config()

    print("Loading recording counts...")
    counts = load_all_counts(cfg)

    print("\nLoading ML downloaded counts...")
    ml_dl = load_ml_downloaded_counts()
    if len(ml_dl) > 0:
        print(f"  {len(ml_dl)} species, {ml_dl['ml_downloaded'].sum():,} recordings")
        counts = counts.join(ml_dl, how="left")
        counts[["ml_downloaded", "ml_downloaded_japan"]] = (
            counts[["ml_downloaded", "ml_downloaded_japan"]].fillna(0).astype(int)
        )

    print("\nLoading eBird frequency data...")
    freq = pd.read_csv(STEP_DIR / "ebird_frequency.csv")
    freq = freq.drop_duplicates(subset="ebird_species_code", keep="first")
    print(f"  {len(freq)} species with frequency data")

    print("\nLoading species list...")
    species = get_target_species(cfg)

    # ebird_species_code の重複を除去（亜種が同じコードにマッピングされるケース）
    species = species.drop_duplicates(subset="ebird_species_code", keep="first")
    print(f"  {len(species)} unique species (after dedup)")

    # マージ: species list + frequency + counts
    merged = species.merge(freq, on="ebird_species_code", how="left",
                           suffixes=("", "_freq"))
    # 重複カラム除去
    for col in ["scientific_name_freq", "japanese_name_freq"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    merged = merged.set_index("ebird_species_code")
    merged = merged.join(counts, how="left").fillna(0)
    merged = merged.reset_index()

    # 優先度算出
    print("\nComputing priority scores...")
    result = compute_priority(merged)

    # 全種分析結果を保存
    out_path = STEP_DIR / "coverage_gap_analysis.csv"
    result.sort_values("priority_score", ascending=False).to_csv(
        out_path, index=False, encoding="utf-8-sig",
    )
    print(f"  → {out_path} ({len(result)} species)")

    # ── レポート ──
    print(f"\n{'='*60}")
    print("Coverage Gap Analysis Report")
    print(f"{'='*60}")

    regular = result[~result["residence_status"].isin(["undetected", "vagrant", ""])]
    print(f"\n日本で定期的に観察される種: {len(regular)}")

    # ティア別集計
    print(f"\n--- 優先度ティア（downloadable recordings 基準）---")
    for tier, desc in [("P1", "< 50"), ("P2", "50-99"), ("P3", "≥ 100")]:
        n_all = (result["priority_tier"] == tier).sum()
        n_reg = (regular["priority_tier"] == tier).sum()
        print(f"  {tier} ({desc:>6}): {n_reg} species (全体: {n_all})")

    # P1/P2 の詳細
    p1 = regular[regular["priority_tier"] == "P1"].sort_values(
        "priority_score", ascending=False)
    p2 = regular[regular["priority_tier"] == "P2"].sort_values(
        "priority_score", ascending=False)

    if len(p1) > 0:
        print(f"\n--- P1 最優先種（DL可能 < 50）: {len(p1)} 種 ---")
        header = (f"{'種コード':>12} {'和名':>14} {'頻度':>6} "
                  f"{'DL可能':>6} {'ML_DL':>6} {'ML全体':>6}")
        print(header)
        print("-" * len(header.encode("utf-8")))
        for _, r in p1.head(30).iterrows():
            ml_dl_val = r.get("ml_downloaded", 0)
            ml_meta = r.get("ml_metadata_total", 0)
            print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>14} "
                  f"{r['frequency_annual_mean']:>6.3f} "
                  f"{r['recordings_downloadable']:>6.0f} "
                  f"{ml_dl_val:>6.0f} {ml_meta:>6.0f}")

    if len(p2) > 0:
        print(f"\n--- P2 次優先種（DL可能 50-99）: {len(p2)} 種 ---")
        for _, r in p2.head(20).iterrows():
            ml_dl_val = r.get("ml_downloaded", 0)
            ml_meta = r.get("ml_metadata_total", 0)
            print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>14} "
                  f"{r['frequency_annual_mean']:>6.3f} "
                  f"{r['recordings_downloadable']:>6.0f} "
                  f"{ml_dl_val:>6.0f} {ml_meta:>6.0f}")

    # 可視化
    print(f"\nGenerating figures...")
    plot_gap_analysis(result)

    # ソース別サマリ
    print(f"\n--- ソース別サマリ ---")
    for col, name in [
        ("xc_total", "Xeno-canto"),
        ("inat_total", "iNat Sounds S3"),
        ("inat_api_total", "iNat API"),
        ("ml_downloaded", "ML (downloaded)"),
        ("ml_total", "ML (metadata)"),
    ]:
        if col in result.columns:
            total = result[col].sum()
            n_species = (result[col] > 0).sum()
            print(f"  {name:20s}: {total:>10,.0f} recordings, {n_species:>4} species")
    dl_total = result["recordings_downloadable"].sum()
    dl_species = (result["recordings_downloadable"] > 0).sum()
    print(f"  {'DL可能合計':20s}: {dl_total:>10,.0f} recordings, {dl_species:>4} species")


if __name__ == "__main__":
    main()
