"""
種の優先順位リスト作成 + テスト用サンプル選定。

Part A (priority): eBird頻度と録音数に基づく優先順位リストを生成
Part B (samples): 上位50種から各5ファイルをランダムサンプリング

使い方:
  python select_test_samples.py priority   # Part A のみ
  python select_test_samples.py samples    # Part B のみ (Part A の出力が必要)
  python select_test_samples.py all        # 両方実行
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent.parent
STEP02_DIR = STEP_DIR.parent / "02_data_collection"

NAS_BASE = Path("~/NAS/nasbi/ToriNET").expanduser()
UNIFIED_METADATA_PATH = NAS_BASE / "metadata" / "unified_metadata.parquet"
EBIRD_FREQ_PATH = STEP02_DIR / "ebird_frequency.csv"

PRIORITY_OUTPUT = STEP_DIR / "species_priority.csv"
SAMPLES_OUTPUT = STEP_DIR / "test_samples.csv"
TEST_SAMPLES_DIR = NAS_BASE / "segments" / "test_samples"

# Selection parameters
MIN_RECORDINGS = 50
TOP_N_SPECIES = 50
SAMPLES_PER_SPECIES = 5
RANDOM_SEED = 42


def load_unified_metadata() -> pd.DataFrame:
    """Load unified metadata from NAS."""
    if not UNIFIED_METADATA_PATH.exists():
        print(f"ERROR: Unified metadata not found: {UNIFIED_METADATA_PATH}")
        sys.exit(1)
    print(f"Loading unified metadata: {UNIFIED_METADATA_PATH}")
    df = pq.read_table(str(UNIFIED_METADATA_PATH)).to_pandas()
    print(f"  {len(df)} recordings, {df['ebird_species_code'].nunique()} species")
    return df


def load_ebird_frequency() -> pd.DataFrame:
    """Load eBird frequency data."""
    if not EBIRD_FREQ_PATH.exists():
        print(f"ERROR: eBird frequency file not found: {EBIRD_FREQ_PATH}")
        sys.exit(1)
    print(f"Loading eBird frequency: {EBIRD_FREQ_PATH}")
    df = pd.read_csv(EBIRD_FREQ_PATH, encoding="utf-8-sig")
    print(f"  {len(df)} species")
    return df


def build_priority_list() -> pd.DataFrame:
    """Build species priority list based on eBird frequency and recording count.

    Returns the priority DataFrame (also saved to CSV).
    """
    meta = load_unified_metadata()
    ebird = load_ebird_frequency()

    # Count recordings per species
    rec_counts = (
        meta.groupby("ebird_species_code")
        .agg(
            total_recordings=("recording_id", "size"),
            scientific_name=("scientific_name", "first"),
            japanese_name=("japanese_name", "first"),
        )
        .reset_index()
    )
    print(f"\nTotal species with recordings: {len(rec_counts)}")

    # Merge with eBird frequency (deduplicate eBird data first)
    ebird_dedup = ebird[["ebird_species_code", "frequency_annual_mean"]].drop_duplicates(
        subset=["ebird_species_code"], keep="first"
    )
    priority = rec_counts.merge(
        ebird_dedup,
        on="ebird_species_code",
        how="left",
    )

    # Species not in eBird frequency get NaN -> fill with 0 (lowest priority)
    n_no_freq = priority["frequency_annual_mean"].isna().sum()
    if n_no_freq > 0:
        print(f"Species not in eBird frequency data: {n_no_freq} (assigned priority 0)")
    priority["frequency_annual_mean"] = priority["frequency_annual_mean"].fillna(0.0)

    # Filter to species with MIN_RECORDINGS+ recordings
    enough = priority[priority["total_recordings"] >= MIN_RECORDINGS].copy()
    not_enough = priority[priority["total_recordings"] < MIN_RECORDINGS]
    print(f"Species with {MIN_RECORDINGS}+ recordings: {len(enough)}")
    print(f"Species with <{MIN_RECORDINGS} recordings: {len(not_enough)} (excluded from priority)")

    # Sort by frequency descending
    enough = enough.sort_values("frequency_annual_mean", ascending=False).reset_index(drop=True)
    enough["priority_rank"] = enough.index + 1

    # Reorder columns
    priority_df = enough[
        [
            "priority_rank",
            "ebird_species_code",
            "scientific_name",
            "japanese_name",
            "frequency_annual_mean",
            "total_recordings",
        ]
    ]

    # Save
    priority_df.to_csv(PRIORITY_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved priority list: {PRIORITY_OUTPUT}")
    print(f"  {len(priority_df)} species")

    # Summary stats
    print(f"\n--- Priority List Summary ---")
    print(f"Total species: {len(priority_df)}")
    print(f"Frequency range: {priority_df['frequency_annual_mean'].min():.4f} - "
          f"{priority_df['frequency_annual_mean'].max():.4f}")
    print(f"Recordings range: {priority_df['total_recordings'].min()} - "
          f"{priority_df['total_recordings'].max()}")
    print(f"\nTop 10 species:")
    for _, row in priority_df.head(10).iterrows():
        print(f"  {row['priority_rank']:3d}. {row['ebird_species_code']:12s} "
              f"{row['japanese_name']:8s} "
              f"freq={row['frequency_annual_mean']:.4f} "
              f"recs={row['total_recordings']}")

    return priority_df


def select_test_samples(samples_per_species: int = SAMPLES_PER_SPECIES) -> pd.DataFrame:
    """Select test samples from top species in the priority list.

    Args:
        samples_per_species: Number of samples to select per species.

    Returns the test samples DataFrame (also saved to CSV).
    """
    # Load priority list
    if not PRIORITY_OUTPUT.exists():
        print(f"ERROR: Priority list not found: {PRIORITY_OUTPUT}")
        print("  Run 'python select_test_samples.py priority' first.")
        sys.exit(1)

    priority = pd.read_csv(PRIORITY_OUTPUT, encoding="utf-8-sig")
    print(f"Loaded priority list: {len(priority)} species")

    # Take top N species
    top_species = priority.head(TOP_N_SPECIES)
    target_codes = set(top_species["ebird_species_code"])
    print(f"Target species (top {TOP_N_SPECIES}): {len(target_codes)}")

    # Load unified metadata
    meta = load_unified_metadata()

    # Filter to target species
    meta_target = meta[meta["ebird_species_code"].isin(target_codes)].copy()
    print(f"Recordings for target species: {len(meta_target)}")

    # For each species, sample SAMPLES_PER_SPECIES recordings
    # Prefer XC quality A/B recordings
    samples = []

    for species_code in sorted(target_codes):
        sp_df = meta_target[meta_target["ebird_species_code"] == species_code].copy()

        if len(sp_df) == 0:
            print(f"  WARNING: No recordings for {species_code}")
            continue

        # Assign selection priority: XC A > XC B > other
        quality_priority = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        sp_df["_quality_rank"] = sp_df["quality"].map(quality_priority).fillna(5)

        # Sort by quality (best first), then shuffle within same quality for randomness
        sp_df = sp_df.sort_values("_quality_rank")

        # Split into high-quality (A/B) and rest
        high_q = sp_df[sp_df["_quality_rank"] <= 1]
        rest = sp_df[sp_df["_quality_rank"] > 1]

        n_needed = samples_per_species

        selected = []
        # First, sample from high-quality recordings
        if len(high_q) > 0:
            n_from_high = min(n_needed, len(high_q))
            selected.append(high_q.sample(n=n_from_high, random_state=RANDOM_SEED))
            n_needed -= n_from_high

        # If still need more, sample from rest
        if n_needed > 0 and len(rest) > 0:
            n_from_rest = min(n_needed, len(rest))
            selected.append(rest.sample(n=n_from_rest, random_state=RANDOM_SEED))

        if selected:
            sp_samples = pd.concat(selected, ignore_index=True)
            samples.append(sp_samples)

    if not samples:
        print("ERROR: No samples selected!")
        sys.exit(1)

    all_samples = pd.concat(samples, ignore_index=True)

    # Copy files to test_samples directory
    print(f"\nCopying {len(all_samples)} files to {TEST_SAMPLES_DIR}...")
    copied = 0
    missing = 0
    copy_errors = 0

    for idx, row in all_samples.iterrows():
        species_code = row["ebird_species_code"]
        file_path = row["file_path"]

        # Resolve file path (NAS-relative -> absolute)
        if not file_path:
            missing += 1
            continue

        src = Path(file_path)
        if not src.is_absolute():
            src = NAS_BASE / file_path

        if not src.exists():
            print(f"  MISSING: {src}")
            missing += 1
            continue

        # Create destination directory
        dst_dir = TEST_SAMPLES_DIR / species_code
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        try:
            shutil.copy2(str(src), str(dst))
            copied += 1
        except OSError as e:
            print(f"  COPY ERROR: {src} -> {dst}: {e}")
            copy_errors += 1

    print(f"  Copied: {copied}")
    print(f"  Missing source files: {missing}")
    if copy_errors > 0:
        print(f"  Copy errors: {copy_errors}")

    # Build output DataFrame
    result = all_samples[
        [
            "ebird_species_code",
            "scientific_name",
            "recording_id",
            "source",
            "file_path",
            "quality",
        ]
    ].copy()

    # Save
    result.to_csv(SAMPLES_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved test samples list: {SAMPLES_OUTPUT}")

    # Summary
    print(f"\n--- Test Samples Summary ---")
    print(f"Total samples: {len(result)}")
    print(f"Species: {result['ebird_species_code'].nunique()}")
    print(f"Source breakdown:")
    for src, cnt in result["source"].value_counts().items():
        print(f"  {src}: {cnt}")
    print(f"Quality breakdown:")
    for q, cnt in result["quality"].value_counts(dropna=False).sort_index().items():
        label = q if pd.notna(q) else "(none)"
        print(f"  {label}: {cnt}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Create species priority list and select test samples"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("priority", help="Create species_priority.csv (Part A)")
    samples_parser = subparsers.add_parser("samples", help="Select test samples (Part B, requires Part A)")
    samples_parser.add_argument(
        "--samples-per-species", type=int, default=SAMPLES_PER_SPECIES,
        help=f"Number of samples per species (default: {SAMPLES_PER_SPECIES})",
    )
    all_parser = subparsers.add_parser("all", help="Run both Part A and Part B")
    all_parser.add_argument(
        "--samples-per-species", type=int, default=SAMPLES_PER_SPECIES,
        help=f"Number of samples per species (default: {SAMPLES_PER_SPECIES})",
    )

    args = parser.parse_args()

    if args.command == "priority":
        build_priority_list()
    elif args.command == "samples":
        select_test_samples(samples_per_species=args.samples_per_species)
    elif args.command == "all":
        build_priority_list()
        print("\n" + "=" * 60 + "\n")
        select_test_samples(samples_per_species=args.samples_per_species)


if __name__ == "__main__":
    main()
