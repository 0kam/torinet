"""
Macaulay Library 正式データ申請用のカタログ番号リストを生成する。

P1/P2種（total recordings < 100）のMLアセットIDを抽出し、
申請用CSVとして出力する。
また、ML取得後もデータが不足する種の分析を行う。

使い方:
  python generate_ml_request.py                 # メタデータベース（デフォルト）
  python generate_ml_request.py --downloaded-only  # DL済みファイルのみカウント

出力:
  - ml_request/ml_request_batch_N.csv      — バッチ別の申請用CSV（カタログ番号付き）
  - ml_request/ml_request_batch_N_ids.csv  — カタログ番号のみ（添付用）
  - ml_request/ml_request_summary.csv      — 種ごとのサマリ（取得後の見込み含む）
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils import get_target_species, load_config, nas_path

STEP_DIR = Path(__file__).resolve().parent

# ML申請制限
MAX_ASSETS_PER_BATCH = 40_000
MAX_SPECIES_PER_BATCH = 100

# 訓練データ閾値
THRESHOLD_P1 = 50   # 最優先
THRESHOLD_P2 = 100  # 十分


def load_current_counts(cfg: dict, downloaded_only: bool = False) -> pd.DataFrame:
    """XC + iNat S3 + iNat API の種ごと録音数を集計する。

    Args:
        downloaded_only: True の場合、file_path が設定済み（実際にDL済み）の
                         レコードのみカウントする。
    """
    paths = {
        "xc": nas_path(cfg, "audio/xeno-canto/metadata/xc_metadata.parquet"),
        "inat": nas_path(cfg, "audio/inat-sounds/annotations/inat_metadata.parquet"),
        "inat_api": nas_path(cfg, "audio/inat-api/metadata/inat_api_metadata.parquet"),
    }

    frames = []
    for name, path in paths.items():
        if path.exists():
            df = pq.read_table(str(path)).to_pandas()
            total_before = len(df)
            if downloaded_only and "file_path" in df.columns:
                df = df[df["file_path"].notna() & (df["file_path"] != "")]
            counts = df.groupby("ebird_species_code").size().rename(f"{name}_total")
            frames.append(counts)
            if downloaded_only:
                print(f"  {name:10s}: {len(df):>10,} downloaded / {total_before:>10,} metadata")
            else:
                print(f"  {name:10s}: {len(df):>10,} recordings (metadata)")

    if frames:
        return pd.concat(frames, axis=1).fillna(0).astype(int)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Generate ML request CSVs")
    parser.add_argument(
        "--downloaded-only", action="store_true",
        help="file_pathが設定済み（実際にDL済み）のレコードのみカウントする",
    )
    args = parser.parse_args()

    cfg = load_config()

    mode = "downloaded files only" if args.downloaded_only else "metadata (all)"
    print(f"Counting mode: {mode}")

    # ── 1. 録音数を集計 ──
    print(f"\nLoading recording counts (XC + iNat S3 + iNat API)...")
    dl_counts = load_current_counts(cfg, downloaded_only=args.downloaded_only)
    dl_counts["downloaded_total"] = dl_counts.sum(axis=1)
    print(f"  {len(dl_counts)} species with recordings")

    # ── 2. MLメタデータを読み込み ──
    print("\nLoading ML metadata...")
    ml_path = nas_path(cfg, "audio/macaulay/metadata/ml_metadata.parquet")
    ml_df = pq.read_table(str(ml_path)).to_pandas()
    print(f"  {len(ml_df)} ML recordings total")

    # ── 3. 種リストとマージ ──
    species = get_target_species(cfg)
    species = species.drop_duplicates(subset="ebird_species_code", keep="first")

    species = species.set_index("ebird_species_code")
    species["downloaded_total"] = dl_counts["downloaded_total"].reindex(species.index).fillna(0).astype(int)
    species = species.reset_index()

    # ML録音数
    ml_counts = ml_df.groupby("ebird_species_code").size().rename("ml_available")
    species = species.set_index("ebird_species_code")
    species["ml_available"] = ml_counts.reindex(species.index).fillna(0).astype(int)
    species["total_with_ml"] = species["downloaded_total"] + species["ml_available"]
    species = species.reset_index()

    # ── 4. P1/P2 種を特定 ──
    # DL済みの録音数で判定（MLはまだ手元にないので除く）
    p1 = species[species["downloaded_total"] < THRESHOLD_P1].copy()
    p2 = species[
        (species["downloaded_total"] >= THRESHOLD_P1)
        & (species["downloaded_total"] < THRESHOLD_P2)
    ].copy()
    p3 = species[species["downloaded_total"] >= THRESHOLD_P2].copy()

    p1["priority_tier"] = "P1"
    p2["priority_tier"] = "P2"

    need_ml = pd.concat([p1, p2], ignore_index=True)

    print(f"\n{'='*60}")
    print("Priority Tier Summary (based on downloaded recordings)")
    print(f"{'='*60}")
    print(f"  P1 (< {THRESHOLD_P1} downloaded):  {len(p1)} species")
    print(f"  P2 ({THRESHOLD_P1}-{THRESHOLD_P2-1} downloaded): {len(p2)} species")
    print(f"  P3 (≥ {THRESHOLD_P2} downloaded):  {len(p3)} species")

    # ── 5. ML取得後の見込み分析 ──
    print(f"\n{'='*60}")
    print("ML取得後の見込み分析")
    print(f"{'='*60}")

    # ML録音がある種
    has_ml = need_ml[need_ml["ml_available"] > 0]
    no_ml = need_ml[need_ml["ml_available"] == 0]

    print(f"\nML録音がある種: {len(has_ml)} / {len(need_ml)}")
    print(f"ML録音がない種: {len(no_ml)} / {len(need_ml)}")

    # ML取得後のティア変化
    has_ml = has_ml.copy()
    still_under_50 = has_ml[has_ml["total_with_ml"] < THRESHOLD_P1]
    reach_50 = has_ml[
        (has_ml["total_with_ml"] >= THRESHOLD_P1)
        & (has_ml["total_with_ml"] < THRESHOLD_P2)
    ]
    reach_100 = has_ml[has_ml["total_with_ml"] >= THRESHOLD_P2]

    print(f"\nML取得後の状態（ML録音がある {len(has_ml)} 種）:")
    print(f"  まだ < 50:    {len(still_under_50)} species")
    print(f"  50-99 に到達: {len(reach_50)} species")
    print(f"  ≥ 100 に到達: {len(reach_100)} species")

    # ML録音がない種の詳細
    if len(no_ml) > 0:
        print(f"\n--- ML録音なし（{len(no_ml)} 種）---")
        print(f"{'種コード':>12} {'和名':>20} {'DL済':>6} {'ティア':>4}")
        print("-" * 50)
        for _, r in no_ml.sort_values("downloaded_total").iterrows():
            print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>20} "
                  f"{r['downloaded_total']:>6} {r['priority_tier']:>4}")

    # ML取得後もまだ < 50 の種
    all_still_under_50 = pd.concat([still_under_50, no_ml], ignore_index=True)
    if len(all_still_under_50) > 0:
        print(f"\n--- ML取得後も < 50 の種（{len(all_still_under_50)} 種）---")
        print(f"{'種コード':>12} {'和名':>20} {'DL済':>6} {'ML':>6} {'合計':>6}")
        print("-" * 60)
        for _, r in all_still_under_50.sort_values("total_with_ml").iterrows():
            print(f"{r['ebird_species_code']:>12} {r['japanese_name']:>20} "
                  f"{r['downloaded_total']:>6} {r['ml_available']:>6} "
                  f"{r['total_with_ml']:>6}")

    # ── 6. 申請用アセットID抽出 ──
    print(f"\n{'='*60}")
    print("Generating ML request CSVs...")
    print(f"{'='*60}")

    # ML録音がある P1/P2 種のアセットIDを抽出
    target_species = has_ml["ebird_species_code"].tolist()
    target_ml = ml_df[ml_df["ebird_species_code"].isin(target_species)].copy()

    # 必要な列を整形
    target_ml = target_ml[[
        "ml_asset_id", "ebird_species_code", "scientific_name",
        "japanese_name", "ml_rating", "ml_rating_count",
        "ml_location", "ml_obs_date", "ml_user",
        "latitude", "longitude", "country", "is_japan",
    ]].copy()

    # カタログ番号フォーマット
    target_ml["catalog_number"] = "ML" + target_ml["ml_asset_id"].astype(str)

    # ティア情報を付与
    tier_map = has_ml.set_index("ebird_species_code")["priority_tier"].to_dict()
    target_ml["priority_tier"] = target_ml["ebird_species_code"].map(tier_map)

    # P1を先に、同一ティア内は種コード順
    tier_order = {"P1": 0, "P2": 1}
    target_ml["_tier_order"] = target_ml["priority_tier"].map(tier_order)
    target_ml = target_ml.sort_values(
        ["_tier_order", "ebird_species_code", "ml_asset_id"],
    )
    target_ml = target_ml.drop(columns=["_tier_order"])

    print(f"\n対象録音数: {len(target_ml)}")
    print(f"対象種数: {target_ml['ebird_species_code'].nunique()}")

    # ── 7. バッチ分割 ──
    out_dir = STEP_DIR / "ml_request"
    out_dir.mkdir(exist_ok=True)

    # 種ごとにグループ化してバッチに振り分け
    species_groups = list(target_ml.groupby("ebird_species_code", sort=False))

    batches = []
    current_batch = []
    current_species_count = 0
    current_asset_count = 0

    for sp_code, sp_df in species_groups:
        n_assets = len(sp_df)

        # 新しいバッチが必要？
        if (current_species_count > 0 and
            (current_species_count + 1 > MAX_SPECIES_PER_BATCH
             or current_asset_count + n_assets > MAX_ASSETS_PER_BATCH)):
            batches.append(pd.concat(current_batch, ignore_index=True))
            current_batch = []
            current_species_count = 0
            current_asset_count = 0

        current_batch.append(sp_df)
        current_species_count += 1
        current_asset_count += n_assets

    if current_batch:
        batches.append(pd.concat(current_batch, ignore_index=True))

    # バッチ別CSV出力
    for i, batch_df in enumerate(batches, 1):
        n_sp = batch_df["ebird_species_code"].nunique()
        n_p1 = batch_df[batch_df["priority_tier"] == "P1"]["ebird_species_code"].nunique()
        n_p2 = batch_df[batch_df["priority_tier"] == "P2"]["ebird_species_code"].nunique()

        # 申請用CSV（カタログ番号 + 種情報）
        batch_path = out_dir / f"ml_request_batch_{i}.csv"
        batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")

        # カタログ番号のみのリスト（添付用）
        ids_path = out_dir / f"ml_request_batch_{i}_ids.csv"
        batch_df[["catalog_number"]].to_csv(ids_path, index=False)

        print(f"\n  Batch {i}: {batch_path.name}")
        print(f"    Species: {n_sp} (P1: {n_p1}, P2: {n_p2})")
        print(f"    Assets:  {len(batch_df)}")

    # ── 8. サマリCSV ──
    summary = has_ml[[
        "ebird_species_code", "japanese_name", "scientific_name",
        "priority_tier", "downloaded_total", "ml_available", "total_with_ml",
    ]].copy()
    summary["still_needed_for_50"] = (THRESHOLD_P1 - summary["total_with_ml"]).clip(lower=0)
    summary["still_needed_for_100"] = (THRESHOLD_P2 - summary["total_with_ml"]).clip(lower=0)
    summary = summary.sort_values(
        ["priority_tier", "total_with_ml"],
        ascending=[True, True],
    )

    summary_path = out_dir / "ml_request_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n  Summary: {summary_path.name}")
    print(f"    {len(summary)} species")

    # ── 9. 最終レポート ──
    print(f"\n{'='*60}")
    print("Final Report")
    print(f"{'='*60}")
    total_assets = sum(len(b) for b in batches)
    total_species = target_ml["ebird_species_code"].nunique()
    print(f"  Batches:     {len(batches)}")
    print(f"  Total species: {total_species}")
    print(f"  Total assets:  {total_assets:,}")
    print(f"  Output dir:    {out_dir}")

    # ML取得後の全体像
    all_species = species.copy()
    all_species["post_ml_total"] = all_species["downloaded_total"] + all_species["ml_available"].clip(upper=0)
    # P1/P2でML申請する種のみml_availableを加算
    target_set = set(target_species)
    all_species["post_ml_total"] = all_species.apply(
        lambda r: r["downloaded_total"] + r["ml_available"]
        if r["ebird_species_code"] in target_set else r["downloaded_total"],
        axis=1,
    )

    post_p1 = (all_species["post_ml_total"] < THRESHOLD_P1).sum()
    post_p2 = ((all_species["post_ml_total"] >= THRESHOLD_P1) & (all_species["post_ml_total"] < THRESHOLD_P2)).sum()
    post_p3 = (all_species["post_ml_total"] >= THRESHOLD_P2).sum()

    print(f"\n  ML取得後の全体像（{len(all_species)} 種）:")
    print(f"    < 50:    {post_p1} species")
    print(f"    50-99:   {post_p2} species")
    print(f"    ≥ 100:   {post_p3} species")


if __name__ == "__main__":
    main()
