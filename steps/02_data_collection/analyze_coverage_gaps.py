"""
録音データと日本国内オカレンスの比較分析。

eBird頻度データ（日本国内）と各ソースの録音数（全世界）を突き合わせ、
訓練データが不足している種を特定する。

優先度ティア:
  P1 (< 50 total recordings) — 最優先
  P2 (50-99 total recordings) — 次に優先
  P3 (≥ 100 total recordings) — 十分

使い方:
  python analyze_coverage_gaps.py

出力:
  - coverage_gap_analysis.csv   — 全種の頻度・録音数・優先度ティア
  - ml_request_priority.csv     — ML申請用の優先リスト（P1/P2のみ）
  - figures/coverage_gap_*.png  — 可視化
"""

import math
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


def compute_priority(merged: pd.DataFrame) -> pd.DataFrame:
    """優先度ティアと補助スコアを算出する。

    ティア（total recordings 基準）:
    - P1: < 50  — 最優先
    - P2: 50-99 — 次に優先
    - P3: ≥ 100 — 十分

    補助スコア（同一ティア内の並べ替え用）:
    - priority_score = frequency / log2(total_recordings + 2)
    - 頻度が高く、録音が少ない種ほどスコアが高い
    """
    df = merged.copy()

    # 合計録音数
    total_cols = [c for c in df.columns if c.endswith("_total")]
    japan_cols = [c for c in df.columns if c.endswith("_japan")]
    df["recordings_total"] = df[total_cols].sum(axis=1)
    df["recordings_japan"] = df[japan_cols].sum(axis=1) if japan_cols else 0

    # DL可能な録音数（XC + iNat S3 + iNat API、MLは未DLなので除外）
    dl_cols = [c for c in total_cols if c != "ml_total"]
    df["recordings_downloadable"] = df[dl_cols].sum(axis=1)

    freq = df["frequency_annual_mean"].fillna(0)

    # 優先度ティア（total recordings 基準）
    df["priority_tier"] = pd.cut(
        df["recordings_total"],
        bins=[-1, 49, 99, float("inf")],
        labels=["P1", "P2", "P3"],
    )

    # 補助スコア = 頻度 / log2(total録音数 + 2)
    df["priority_score"] = freq / np.log2(df["recordings_total"] + 2)

    # ML で追加取得可能な録音数
    df["ml_available"] = df.get("ml_total", 0)

    # MLの日本録音数
    df["ml_japan_available"] = df.get("ml_japan", 0)

    # ML で追加取得すると何件になるか
    df["recordings_with_ml"] = df["recordings_downloadable"] + df["ml_available"]

    return df


def generate_ml_request_list(df: pd.DataFrame) -> pd.DataFrame:
    """ML申請用の優先リストを生成する。

    条件:
    - undetected/vagrant以外（日本で定期的に観察される種）
    - P1またはP2ティア（total recordings < 100）
    - ML録音がある種のみ
    - ティア→補助スコア順
    """
    # フィルタ: 日本で定期的に観察される種
    regular = df[~df["residence_status"].isin(["undetected", "vagrant", ""])].copy()

    # P1/P2のみ（total recordings < 100）
    need_more = regular[regular["priority_tier"].isin(["P1", "P2"])].copy()

    # MLに録音がある種のみ
    has_ml = need_more[need_more["ml_available"] > 0].copy()

    # ソート: P1 → P2、同一ティア内はスコア降順
    tier_order = {"P1": 0, "P2": 1}
    has_ml["_tier_order"] = has_ml["priority_tier"].map(tier_order)
    has_ml = has_ml.sort_values(
        ["_tier_order", "priority_score"], ascending=[True, False],
    )

    # 申請用カラム
    out = has_ml[[
        "ebird_species_code", "japanese_name", "scientific_name",
        "ebird_common_name", "residence_status", "priority_tier",
        "frequency_annual_mean", "frequency_max", "n_periods_detected",
        "recordings_total", "recordings_japan", "recordings_downloadable",
        "ml_available", "ml_japan_available",
        "recordings_with_ml", "priority_score",
    ]].copy()

    return out


def plot_gap_analysis(df: pd.DataFrame):
    """ギャップ分析の可視化。"""
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "sans-serif"]

    regular = df[~df["residence_status"].isin(["undetected", "vagrant", ""])]
    rare = df[df["residence_status"].isin(["undetected", "vagrant"])]

    # --- Figure 1: 頻度 vs 全録音数 scatter（P1/P2/P3ティア帯付き）---
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
            subset["recordings_total"] + 1,
            alpha=0.6, s=30, c=color, label=status,
        )

    ax.scatter(
        rare["frequency_annual_mean"],
        rare["recordings_total"] + 1,
        alpha=0.3, s=15, c="gray", label="vagrant/undetected",
    )

    ax.set_xlabel("eBird Frequency (Japan annual mean)")
    ax.set_ylabel("Total Recordings (all sources, log scale)")
    ax.set_yscale("log")
    ax.set_title("Recording Coverage vs eBird Frequency (Total Recordings)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # P1種で高頻度のものにラベル
    p1_regular = regular[regular["priority_tier"] == "P1"]
    top_p1 = p1_regular.nlargest(15, "frequency_annual_mean")
    for _, row in top_p1.iterrows():
        ax.annotate(
            row["japanese_name"],
            (row["frequency_annual_mean"], row["recordings_total"] + 1),
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
        ("ml_total", "#9C27B0", "ML (metadata)"),
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
    ax.set_xlabel("Total Recordings (all sources)")
    ax.set_title("P1/P2 Species: Total Recordings by Source")
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

    # ML申請用優先リスト
    ml_list = generate_ml_request_list(result)
    ml_path = STEP_DIR / "ml_request_priority.csv"
    ml_list.to_csv(ml_path, index=False, encoding="utf-8-sig")
    print(f"  → {ml_path} ({len(ml_list)} species)")

    # ── レポート ──
    print(f"\n{'='*60}")
    print("Coverage Gap Analysis Report")
    print(f"{'='*60}")

    regular = result[~result["residence_status"].isin(["undetected", "vagrant", ""])]
    print(f"\n日本で定期的に観察される種: {len(regular)}")

    # ティア別集計
    print(f"\n--- 優先度ティア（total recordings 基準）---")
    for tier, desc in [("P1", "< 50"), ("P2", "50-99"), ("P3", "≥ 100")]:
        n = (regular["priority_tier"] == tier).sum()
        print(f"  {tier} ({desc:>6}): {n} species")

    # P1トップ（最優先）
    p1_species = ml_list[ml_list["priority_tier"] == "P1"]
    p2_species = ml_list[ml_list["priority_tier"] == "P2"]

    print(f"\n--- P1 最優先種（< 50 recordings, ML録音あり）: {len(p1_species)} 種 ---")
    print(f"{'種コード':>12} {'和名':>12} {'ティア':>4} {'頻度':>6} "
          f"{'総録音':>6} {'DL済':>6} {'ML録音':>6} {'+ML後':>6}")
    print("-" * 72)
    for _, r in p1_species.head(30).iterrows():
        print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>12} "
              f"{r['priority_tier']:>4} {r['frequency_annual_mean']:>6.3f} "
              f"{r['recordings_total']:>6.0f} {r['recordings_downloadable']:>6.0f} "
              f"{r['ml_available']:>6.0f} {r['recordings_with_ml']:>6.0f}")

    print(f"\n--- P2 次優先種（50-99 recordings, ML録音あり）: {len(p2_species)} 種 ---")
    for _, r in p2_species.head(20).iterrows():
        print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>12} "
              f"{r['priority_tier']:>4} {r['frequency_annual_mean']:>6.3f} "
              f"{r['recordings_total']:>6.0f} {r['recordings_downloadable']:>6.0f} "
              f"{r['ml_available']:>6.0f} {r['recordings_with_ml']:>6.0f}")

    # MLなし P1種
    p1_no_ml = regular[
        (regular["priority_tier"] == "P1") & (regular["ml_available"] == 0)
    ]
    if len(p1_no_ml) > 0:
        print(f"\n--- P1 だがML録音なし: {len(p1_no_ml)} 種（要別途対応）---")
        for _, r in p1_no_ml.iterrows():
            print(f"  {r['ebird_species_code']:>12} {r['japanese_name']:>12} "
                  f"total={r['recordings_total']:.0f} freq={r['frequency_annual_mean']:.3f}")

    # 可視化
    print(f"\nGenerating figures...")
    plot_gap_analysis(result)

    # ML申請の概要
    print(f"\n--- ML申請用サマリ ---")
    print(f"申請対象種: {len(ml_list)} (P1: {len(p1_species)}, P2: {len(p2_species)})")
    total_ml = ml_list["ml_available"].sum()
    total_ml_jp = ml_list["ml_japan_available"].sum()
    print(f"ML録音合計: {total_ml:,.0f} (うち日本: {total_ml_jp:,.0f})")

    # ML取得後の改善見込み
    would_reach_50 = (ml_list["recordings_with_ml"] >= 50).sum()
    would_reach_100 = (ml_list["recordings_with_ml"] >= 100).sum()
    print(f"ML取得後 ≥ 50 到達: {would_reach_50}/{len(ml_list)} species")
    print(f"ML取得後 ≥ 100 到達: {would_reach_100}/{len(ml_list)} species")

    # 40K/100種の制限に合わせたバッチ分割案
    print(f"\n--- バッチ分割案（ML申請上限: 40K件/100種）---")
    batch_size = 100
    cumulative = 0
    batch_num = 1
    batch_start = 0
    for i, (_, row) in enumerate(ml_list.iterrows()):
        cumulative += row["ml_available"]
        if cumulative >= 40000 or (i - batch_start + 1) >= batch_size:
            print(f"  Batch {batch_num}: species {batch_start+1}-{i+1} "
                  f"({i - batch_start + 1} species, ~{cumulative:,.0f} recordings)")
            batch_num += 1
            batch_start = i + 1
            cumulative = 0
    if batch_start < len(ml_list):
        remaining = len(ml_list) - batch_start
        print(f"  Batch {batch_num}: species {batch_start+1}-{len(ml_list)} "
              f"({remaining} species, ~{cumulative:,.0f} recordings)")
    print(f"  Total: {batch_num} batches")


if __name__ == "__main__":
    main()
