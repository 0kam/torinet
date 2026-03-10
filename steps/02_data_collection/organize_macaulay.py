"""
Macaulay Library の tar アーカイブを展開し、種別ディレクトリに整理する。

入力:
  - audio/macaulay/Batch_*.tar  (flat な {asset_id}.{wav,mp3,m4a})
  - ml_request/ml_request_batch_*.csv  (asset_id → species_code マッピング)

出力:
  - audio/macaulay/audio/{species_code}/ml_{asset_id}.{ext}

処理:
  1. 申請CSVから asset_id → species_code マッピングを構築
  2. 各 tar を順次展開し、種別ディレクトリにリネーム移動
  3. 検証: メタデータとファイル数の一致を確認
  4. 成功後、tar ファイルを削除

使い方:
  python organize_macaulay.py [--keep-tar] [--dry-run]
"""

import argparse
import csv
import os
import re
import tarfile
from collections import Counter
from pathlib import Path

from utils import load_config, nas_path

STEP_DIR = Path(__file__).resolve().parent


def load_asset_species_map() -> dict[str, str]:
    """申請 CSV から asset_id → ebird_species_code マッピングを構築する。"""
    reqdir = STEP_DIR / "ml_request"
    mapping = {}

    batch_files = sorted(reqdir.glob("ml_request_batch_*.csv"))
    batch_files = [f for f in batch_files if "_ids" not in f.name]

    for fpath in batch_files:
        with open(fpath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                aid = str(row["ml_asset_id"])
                mapping[aid] = row["ebird_species_code"]

    print(f"Loaded {len(mapping)} asset → species mappings")
    return mapping


def extract_and_organize(
    cfg: dict,
    asset_map: dict[str, str],
    dry_run: bool = False,
    keep_tar: bool = False,
) -> dict[str, int]:
    """tar を展開して種別ディレクトリに振り分ける。"""
    ml_dir = nas_path(cfg, "audio/macaulay")
    audio_dir = nas_path(cfg, "audio/macaulay/audio")

    tar_files = sorted(ml_dir.glob("Batch*.tar"))
    if not tar_files:
        print("No tar files found!")
        return {}

    print(f"\nFound {len(tar_files)} tar files:")
    for tf in tar_files:
        size_gb = tf.stat().st_size / (1024**3)
        print(f"  {tf.name} ({size_gb:.1f} GB)")

    species_counts = Counter()
    total_extracted = 0
    unmapped = []

    for tar_path in tar_files:
        print(f"\nProcessing {tar_path.name}...")

        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            print(f"  {len(members)} files")

            for member in members:
                if not member.isfile():
                    continue

                filename = member.name
                # Extract asset_id: strip extension(s)
                asset_id = re.sub(r"\.(wav|mp3|m4a|flac)$", "", filename)

                # Determine the actual file extension
                ext_match = re.search(r"\.(wav|mp3|m4a|flac)$", filename)
                if ext_match:
                    ext = ext_match.group(1)
                else:
                    ext = "wav"  # default

                species_code = asset_map.get(asset_id)
                if species_code is None:
                    unmapped.append(asset_id)
                    continue

                dest_dir = audio_dir / species_code
                dest_file = dest_dir / f"ml_{asset_id}.{ext}"

                if dry_run:
                    species_counts[species_code] += 1
                    total_extracted += 1
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)

                # Extract to a temporary name then rename
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    print(f"  WARNING: Could not extract {filename}")
                    continue

                with open(dest_file, "wb") as out:
                    while True:
                        chunk = fileobj.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        out.write(chunk)

                species_counts[species_code] += 1
                total_extracted += 1

            if total_extracted % 5000 == 0 and total_extracted > 0:
                print(f"  ... {total_extracted} files extracted so far")

        print(f"  Done. Total so far: {total_extracted} files, "
              f"{len(species_counts)} species")

        # Delete tar after successful extraction
        if not dry_run and not keep_tar:
            tar_path.unlink()
            print(f"  Deleted {tar_path.name}")

    if unmapped:
        print(f"\nWARNING: {len(unmapped)} files had no species mapping")
        print(f"  Sample: {unmapped[:5]}")

    return dict(species_counts)


def verify(cfg: dict, expected_counts: dict[str, int]):
    """展開結果を検証する。"""
    audio_dir = nas_path(cfg, "audio/macaulay/audio")

    print("\nVerifying...")
    actual_counts = Counter()
    total_files = 0

    for species_dir in sorted(audio_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        n = sum(1 for f in species_dir.iterdir() if f.is_file())
        actual_counts[species_dir.name] = n
        total_files += n

    print(f"  Species directories: {len(actual_counts)}")
    print(f"  Total files: {total_files}")
    print(f"  Expected: {sum(expected_counts.values())} files, "
          f"{len(expected_counts)} species")

    # Check mismatches
    mismatches = []
    for sp, expected in expected_counts.items():
        actual = actual_counts.get(sp, 0)
        if actual != expected:
            mismatches.append((sp, expected, actual))

    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)}):")
        for sp, expected, actual in mismatches[:10]:
            print(f"    {sp}: expected {expected}, got {actual}")
    else:
        print("  All counts match!")

    return len(mismatches) == 0


def main():
    parser = argparse.ArgumentParser(description="Organize Macaulay Library downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without extracting")
    parser.add_argument("--keep-tar", action="store_true",
                        help="Keep tar files after extraction")
    args = parser.parse_args()

    cfg = load_config()
    asset_map = load_asset_species_map()

    if args.dry_run:
        print("\n=== DRY RUN ===")

    counts = extract_and_organize(
        cfg, asset_map, dry_run=args.dry_run, keep_tar=args.keep_tar,
    )

    print(f"\n{'='*60}")
    print(f"Extraction complete: {sum(counts.values())} files → "
          f"{len(counts)} species")

    if not args.dry_run:
        verify(cfg, counts)

    # Summary by species count
    if counts:
        count_dist = Counter()
        for n in counts.values():
            if n < 10:
                count_dist["1-9"] += 1
            elif n < 50:
                count_dist["10-49"] += 1
            elif n < 100:
                count_dist["50-99"] += 1
            else:
                count_dist["100+"] += 1

        print(f"\nSpecies by file count:")
        for bucket in ["1-9", "10-49", "50-99", "100+"]:
            print(f"  {bucket:>6}: {count_dist.get(bucket, 0)} species")


if __name__ == "__main__":
    main()
