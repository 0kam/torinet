"""
全ソースのメタデータを統合し、DL済みファイルのみの統一メタデータを構築する。

入力:
  - audio/xeno-canto/metadata/xc_metadata.parquet
  - audio/inat-sounds/annotations/inat_metadata.parquet
  - audio/inat-api/metadata/inat_api_metadata.parquet
  - audio/macaulay/metadata/ml_metadata.parquet

出力:
  - metadata/unified_metadata.parquet

処理:
  1. 各ソースからDL済み録音を抽出（フィルタ条件を再適用）
  2. file_path を NAS 相対パスに正規化
  3. Macaulay は展開済みファイルから file_path を構築
  4. 統一スキーマ (UNIFIED_SCHEMA) のカラムのみ保持
  5. ファイル存在チェック（オプション）

使い方:
  python build_unified_metadata.py [--verify] [--dry-run]
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import (
    UNIFIED_SCHEMA,
    load_config,
    load_metadata,
    nas_path,
    save_metadata,
)

STEP_DIR = Path(__file__).resolve().parent


def load_xc(cfg: dict) -> pd.DataFrame:
    """XC: DLフィルタ（品質≤C, CCライセンス）かつ file_path ありの録音。"""
    df = load_metadata(nas_path(cfg, cfg["xeno_canto"]["metadata_dir"]) / "xc_metadata.parquet")

    # Apply download filter
    quality_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    df["_qnum"] = df["quality"].map(quality_order)
    cc_mask = df["license"].str.contains("creativecommons", na=False)
    dl_mask = cc_mask & (df["_qnum"] <= 3)
    df = df[dl_mask].drop(columns=["_qnum"])

    # Only include files with file_path set
    df = df[df["file_path"].notna() & (df["file_path"] != "")]

    # Deduplicate by recording_id (same XC recording appearing multiple times)
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  XC: removed {n_dedup} duplicate recording_ids")

    print(f"  XC: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_inat_s3(cfg: dict) -> pd.DataFrame:
    """iNat Sounds 2024: 全件（全てDL済み）。"""
    df = load_metadata(
        nas_path(cfg, cfg["inat_sounds"]["annotations_dir"]) / "inat_metadata.parquet"
    )

    # Normalize relative paths to absolute
    nas_base = Path(cfg["nas_base"])
    mask = ~df["file_path"].str.startswith("/")
    df.loc[mask, "file_path"] = df.loc[mask, "file_path"].apply(
        lambda p: str(nas_base / p)
    )

    print(f"  iNat S3: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_inat_api(cfg: dict) -> pd.DataFrame:
    """iNat API: 重複除外・file_path ありの録音。"""
    df = load_metadata(
        nas_path(cfg, cfg["inat_api"]["metadata_dir"]) / "inat_api_metadata.parquet"
    )

    # Exclude duplicates with iNat Sounds 2024
    if "inat_api_is_duplicate_sounds2024" in df.columns:
        df = df[df["inat_api_is_duplicate_sounds2024"] != True]  # noqa: E712

    # Only include files with file_path set
    df = df[df["file_path"].notna() & (df["file_path"] != "")]

    # Deduplicate by sound_id: same audio file can appear in multiple observations
    # for different species. Keep the first occurrence.
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  iNat API: removed {n_dedup} duplicate recording_ids (same sound, different obs)")

    print(f"  iNat API: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_macaulay(cfg: dict) -> pd.DataFrame:
    """Macaulay: 展開済みファイルからマッピングを構築。"""
    meta_path = nas_path(cfg, cfg["macaulay"]["metadata_dir"]) / "ml_metadata.parquet"
    df = load_metadata(meta_path)

    # Scan actual files on disk
    audio_dir = nas_path(cfg, cfg["macaulay"]["audio_dir"])
    file_map = {}  # asset_id -> file_path

    if audio_dir.exists():
        for species_dir in audio_dir.iterdir():
            if not species_dir.is_dir():
                continue
            for f in species_dir.iterdir():
                if f.is_file() and f.name.startswith("ml_"):
                    # ml_{asset_id}.{ext} -> asset_id
                    stem = f.stem  # ml_{asset_id}
                    asset_id = stem[3:]  # remove "ml_"
                    file_map[asset_id] = str(f)

    print(f"  Macaulay files on disk: {len(file_map)}")

    # Map asset_id to file_path
    df["ml_asset_id_str"] = df["ml_asset_id"].astype(str)
    df["file_path"] = df["ml_asset_id_str"].map(file_map).fillna("")
    df = df[df["file_path"] != ""]
    df = df.drop(columns=["ml_asset_id_str"])

    # Deduplicate by recording_id (same asset appearing in multiple metadata rows)
    n_before = len(df)
    df = df.drop_duplicates(subset=["recording_id"], keep="first")
    n_dedup = n_before - len(df)
    if n_dedup > 0:
        print(f"  Macaulay: removed {n_dedup} duplicate recording_ids")

    print(f"  Macaulay: {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def verify_files(df: pd.DataFrame, sample_size: int = 1000) -> tuple[int, int]:
    """ファイル存在をサンプルチェックする。"""
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    exists = 0
    missing = 0
    missing_examples = []

    for _, row in sample.iterrows():
        p = Path(row["file_path"])
        if p.exists():
            exists += 1
        else:
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(p))

    return exists, missing, missing_examples


def main():
    parser = argparse.ArgumentParser(description="Build unified metadata")
    parser.add_argument("--verify", action="store_true", help="Verify file existence (sample)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    cfg = load_config()

    print("Loading source metadata...")
    dfs = []
    for loader in [load_xc, load_inat_s3, load_inat_api, load_macaulay]:
        dfs.append(loader(cfg))

    # Select unified schema columns only
    unified = []
    for df in dfs:
        cols = [c for c in UNIFIED_SCHEMA if c in df.columns]
        missing_cols = set(UNIFIED_SCHEMA) - set(cols)
        sub = df[cols].copy()
        for mc in missing_cols:
            sub[mc] = None
        unified.append(sub[UNIFIED_SCHEMA])

    result = pd.concat(unified, ignore_index=True)

    # Check for cross-source duplicate recording_ids (should be rare after per-source dedup)
    dup_mask = result["recording_id"].duplicated(keep="first")
    n_dups = dup_mask.sum()
    if n_dups > 0:
        print(f"\nCross-source duplicates: {n_dups} (keeping first)")
        result = result[~dup_mask]

    print(f"\n{'='*60}")
    print(f"Unified metadata: {len(result)} recordings")
    print(f"Species: {result['ebird_species_code'].nunique()}")
    print(f"Japan recordings: {result['is_japan'].sum()}")

    # Source breakdown
    print(f"\nSource breakdown:")
    for src, grp in result.groupby("source"):
        print(f"  {src}: {len(grp)} recordings, {grp['ebird_species_code'].nunique()} species")

    # Species coverage distribution
    species_counts = result.groupby("ebird_species_code").size()
    print(f"\nRecordings per species:")
    print(f"  Mean: {species_counts.mean():.1f}")
    print(f"  Median: {species_counts.median():.0f}")
    print(f"  Min: {species_counts.min()}")
    print(f"  Max: {species_counts.max()}")

    # Tier distribution
    tier_counts = Counter()
    for n in species_counts:
        if n < 10:
            tier_counts["< 10"] += 1
        elif n < 50:
            tier_counts["10-49"] += 1
        elif n < 100:
            tier_counts["50-99"] += 1
        elif n < 500:
            tier_counts["100-499"] += 1
        else:
            tier_counts["500+"] += 1

    print(f"\nSpecies by recording count:")
    for bucket in ["< 10", "10-49", "50-99", "100-499", "500+"]:
        print(f"  {bucket:>8}: {tier_counts.get(bucket, 0)} species")

    if args.verify:
        print(f"\nVerifying file existence (sample)...")
        exists, missing, examples = verify_files(result)
        total = exists + missing
        print(f"  Sample: {exists}/{total} exist ({exists/total*100:.1f}%)")
        if examples:
            print(f"  Missing examples:")
            for ex in examples:
                print(f"    {ex}")

    if not args.dry_run:
        out_path = nas_path(cfg, cfg["unified_metadata"])
        save_metadata(result, out_path)
        print(f"\nSaved to: {out_path}")
    else:
        print("\n(dry-run: not saving)")


if __name__ == "__main__":
    main()
