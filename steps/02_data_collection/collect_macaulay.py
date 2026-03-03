"""
Macaulay Library 音声メタデータ収集 & ダウンロード。

Cornell Lab の Macaulay Library Search API（認証不要）を使って
eBird に投稿された鳥類音声を収集する。

使い方:
  # メタデータのみ収集（全種）
  python collect_macaulay.py --metadata-only

  # テスト: 5種だけ
  python collect_macaulay.py --metadata-only --limit 5

  # 音声ダウンロード（メタデータ収集後）
  python collect_macaulay.py --download

  # テスト: 3種だけダウンロード
  python collect_macaulay.py --download --limit 3

  # ドライラン（ダウンロード対象の確認のみ）
  python collect_macaulay.py --download --dry-run

  # 種ごとの最大録音数を指定（デフォルト: 無制限）
  python collect_macaulay.py --metadata-only --max-per-species 500
"""

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

from utils import (
    get_target_species,
    is_in_japan,
    load_config,
    load_metadata,
    nas_path,
)

SEARCH_API = "https://search.macaulaylibrary.org/api/v1/search"
PER_PAGE = 100  # API最大
PAGE_MAX_RETRIES = 3  # ページ取得の最大リトライ


def parse_args():
    parser = argparse.ArgumentParser(description="Macaulay Library データ収集")
    parser.add_argument("--metadata-only", action="store_true",
                        help="メタデータのみ収集（音声DLしない）")
    parser.add_argument("--download", action="store_true",
                        help="音声ダウンロード（メタデータ収集済み前提）")
    parser.add_argument("--limit", type=int, default=0,
                        help="テスト用: 処理する種数の上限")
    parser.add_argument("--max-per-species", type=int, default=0,
                        help="種ごとの最大録音数（0=無制限）")
    parser.add_argument("--dry-run", action="store_true",
                        help="ダウンロード対象の確認のみ（実際にはDLしない）")
    parser.add_argument("--min-rating", type=float, default=0.0,
                        help="ダウンロード時の最低レーティング（0-5, デフォルト: 0）")
    return parser.parse_args()


# ── メタデータ収集 ──────────────────────────────────────────

def fetch_species_metadata(
    taxon_code: str,
    rate_limit: float = 1.0,
    max_recordings: int = 0,
) -> tuple[list[dict], bool]:
    """1種について全ページのメタデータを取得する。

    Returns:
        (items, success): itemsは取得済みレコード、successは全ページ取得完了かどうか。
        エラーで中断した場合はsuccess=Falseで、次回リトライ対象になる。
    """
    import urllib.request
    import urllib.parse

    all_items = []
    cursor = None
    page = 0

    while True:
        params = {
            "taxonCode": taxon_code,
            "mediaType": "audio",
            "count": PER_PAGE,
        }
        if cursor:
            params["initialCursorMark"] = cursor

        url = f"{SEARCH_API}?{urllib.parse.urlencode(params)}"

        # ページ単位リトライ
        data = None
        for attempt in range(PAGE_MAX_RETRIES):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ToriNet/1.0 (bird-bioacoustics-research)"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                break
            except Exception as e:
                print(f"  Error fetching {taxon_code} page {page} "
                      f"(attempt {attempt + 1}/{PAGE_MAX_RETRIES}): {e}")
                if attempt < PAGE_MAX_RETRIES - 1:
                    time.sleep(rate_limit * (2 ** attempt) + 1)

        if data is None:
            # 全リトライ失敗 → 不完全として返す
            return all_items, False

        results = data.get("results", {})
        content = results.get("content", [])

        if not content:
            break

        all_items.extend(content)
        page += 1

        # 種ごと上限チェック
        if max_recordings > 0 and len(all_items) >= max_recordings:
            all_items = all_items[:max_recordings]
            break

        next_cursor = results.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break

        cursor = next_cursor
        time.sleep(rate_limit)

    return all_items, True


def parse_ml_item(item: dict, species_row: pd.Series, cfg: dict) -> dict:
    """ML APIレスポンスの1アイテムを統一フォーマットに変換する。"""
    asset_id = str(item.get("assetId", ""))
    lat = _safe_float(item.get("latitude"))
    lon = _safe_float(item.get("longitude"))
    rating = _safe_float(item.get("rating")) or 0.0

    return {
        "recording_id": f"ml:{asset_id}",
        "source": "macaulay",
        "ebird_species_code": species_row["ebird_species_code"],
        "scientific_name": species_row["scientific_name"],
        "japanese_name": species_row["japanese_name"],
        "latitude": lat,
        "longitude": lon,
        "country": _extract_country(item.get("locationLine2", "")),
        "is_japan": is_in_japan(lat, lon, cfg),
        "duration_sec": None,  # APIレスポンスに含まれない
        "sample_rate": None,
        "quality": "",
        "license": item.get("licenseType", ""),
        "file_path": "",
        "vocalization_type": item.get("behaviors") or "",
        # ML固有情報
        "ml_asset_id": asset_id,
        "ml_media_url": item.get("mediaUrl", ""),
        "ml_rating": rating,
        "ml_rating_count": item.get("ratingCount", "0"),
        "ml_location": item.get("location", ""),
        "ml_obs_date": item.get("obsDttm", ""),
        "ml_user": item.get("userDisplayName", ""),
        "ml_checklist_id": item.get("eBirdChecklistId", ""),
        "ml_source": item.get("source", ""),
    }


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extract_country(location_line2: str) -> str:
    """locationLine2 ('Prefecture, Country') から国名を抽出する。"""
    if not location_line2:
        return ""
    parts = location_line2.split(",")
    return parts[-1].strip() if parts else ""


def _save_metadata_atomic(df: pd.DataFrame, metadata_path: Path) -> None:
    """Parquetファイルをアトミックに保存する（tmp + rename）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(tmp), compression="snappy")
    tmp.rename(metadata_path)


def collect_metadata(cfg: dict, args) -> pd.DataFrame:
    """全対象種のMLメタデータを収集する。"""
    ml_cfg = cfg["macaulay"]
    rate_limit = ml_cfg["rate_limit_sec"]
    max_per_species = args.max_per_species

    species_df = get_target_species(cfg)
    if args.limit > 0:
        species_df = species_df.head(args.limit)

    print(f"Collecting Macaulay Library metadata for {len(species_df)} species...")
    if max_per_species > 0:
        print(f"  Max per species: {max_per_species}")

    # 進捗ファイル（再開用）
    metadata_dir = nas_path(cfg, ml_cfg["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    progress_path = metadata_dir / "collection_progress.json"
    metadata_path = metadata_dir / "ml_metadata.parquet"

    progress = {"completed_species": [], "species_counts": {}}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)

    completed_species = set(progress.get("completed_species", []))
    species_counts = progress.get("species_counts", {})

    # 既存メタデータ読み込み
    existing_records = []
    if metadata_path.exists():
        existing_df = load_metadata(metadata_path)
        existing_records = existing_df.to_dict("records")
        print(f"  Resuming: {len(completed_species)} species done, "
              f"{len(existing_records)} recordings collected")

    all_records = existing_records
    no_hit_species = []
    error_species = []

    # シグナルハンドリング
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForce exit")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nShutdown requested — saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    remaining = [
        (idx, row) for idx, row in species_df.iterrows()
        if row["ebird_species_code"] not in completed_species
    ]

    skipped = len(species_df) - len(remaining)
    if skipped > 0:
        print(f"  Skipping {skipped} already-completed species")

    for idx, row in tqdm(remaining, desc="Species", unit="sp"):
        if shutdown_requested:
            break

        sp_code = row["ebird_species_code"]
        items, success = fetch_species_metadata(
            sp_code, rate_limit, max_per_species,
        )

        if not items and success:
            # 正常に取得したが録音なし
            no_hit_species.append({
                "species_code": sp_code,
                "scientific_name": row["scientific_name"],
                "japanese_name": row["japanese_name"],
            })
        elif not success:
            # APIエラーで中断 → 部分データは保存するが完了マークしない
            error_species.append(sp_code)
            if items:
                print(f"  {sp_code}: API error after {len(items)} records "
                      f"(will retry next run)")

        if items:
            for item in items:
                parsed = parse_ml_item(item, row, cfg)
                all_records.append(parsed)

        species_counts[sp_code] = len(items)

        # 成功した種のみ完了マーク（エラー種は次回リトライ）
        if success:
            completed_species.add(sp_code)

        # 進捗保存（Parquet + JSON を一緒にアトミック保存）
        df = pd.DataFrame(all_records)
        if len(df) > 0:
            _save_metadata_atomic(df, metadata_path)
        _save_progress(progress_path, {
            "completed_species": sorted(completed_species),
            "species_counts": species_counts,
        })

        time.sleep(rate_limit)

    # 最終保存
    df = pd.DataFrame(all_records)
    if len(df) > 0:
        _save_metadata_atomic(df, metadata_path)
        print(f"\nSaved {len(df)} rows → {metadata_path}")

    # ── 統計レポート ──
    print(f"\n{'='*60}")
    print("Macaulay Library Metadata Collection Report")
    print(f"{'='*60}")
    total_species_found = df["ebird_species_code"].nunique() if len(df) > 0 else 0
    total_recordings = len(df)
    print(f"Total species with recordings: {total_species_found}/{len(species_df)}")
    print(f"Total recordings: {total_recordings}")

    if len(df) > 0:
        japan_count = df["is_japan"].sum()
        print(f"Japan recordings: {japan_count} ({japan_count/total_recordings*100:.1f}%)")

        per_species = df.groupby("ebird_species_code").size()
        print(f"\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

        # レーティング分布
        if "ml_rating" in df.columns:
            ratings = pd.to_numeric(df["ml_rating"], errors="coerce")
            print(f"\nRating distribution:")
            print(f"  Mean: {ratings.mean():.2f}")
            for threshold in [1, 2, 3, 4]:
                n = (ratings >= threshold).sum()
                print(f"  ≥{threshold}: {n} ({n/total_recordings*100:.1f}%)")

        # 鳴き声タイプ
        voc = df["vocalization_type"].value_counts().head(10)
        if len(voc) > 0:
            print(f"\nVocalization types (top 10):")
            for v, cnt in voc.items():
                print(f"  {v or '(unknown)'}: {cnt}")

    if no_hit_species:
        print(f"\nSpecies with NO recordings ({len(no_hit_species)}):")
        for sp in no_hit_species[:20]:
            print(f"  {sp['scientific_name']} ({sp['japanese_name']})")
        if len(no_hit_species) > 20:
            print(f"  ... and {len(no_hit_species) - 20} more")

    if error_species:
        print(f"\nSpecies with API errors ({len(error_species)}, will retry):")
        for sp in error_species[:10]:
            print(f"  {sp}")
        if len(error_species) > 10:
            print(f"  ... and {len(error_species) - 10} more")

    if shutdown_requested:
        print(f"\n  Interrupted — {len(completed_species)}/{len(species_df)} species done.")
        print(f"  Run again to continue.")
        sys.exit(1)

    # エラー種が残っている場合は非ゼロ終了（リトライ用）
    if error_species:
        print(f"\n  {len(error_species)} species had API errors — run again to retry.")
        sys.exit(2)

    return df


# ── 音声ダウンロード ────────────────────────────────────────

def _ml_audio_path(cfg: dict, species_code: str, asset_id: str) -> Path:
    """ML音声ファイルのNAS保存パスを生成する。"""
    audio_dir = nas_path(cfg, cfg["macaulay"]["audio_dir"])
    return audio_dir / species_code / f"ml_{asset_id}.mp3"


async def download_one(
    session: aiohttp.ClientSession,
    row: dict,
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    max_retries: int = 3,
) -> dict:
    """1つの録音をダウンロードする。"""
    async with semaphore:
        asset_id = row["ml_asset_id"]
        media_url = row["ml_media_url"]
        species_code = row["ebird_species_code"]

        if not media_url:
            return {"asset_id": asset_id, "status": "no_url"}

        out_path = _ml_audio_path(cfg, species_code, asset_id)
        if out_path.exists():
            return {"asset_id": asset_id, "status": "exists", "path": str(out_path)}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        last_status = "error"
        for attempt in range(max_retries):
            try:
                async with session.get(
                    media_url,
                    headers={"User-Agent": "ToriNet/1.0 (bird-bioacoustics-research)"},
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        break
                    last_status = f"http_{resp.status}"
                    if resp.status == 429:
                        # Rate limited — longer backoff
                        wait = rate_limit * (4 ** attempt) + 5
                        await asyncio.sleep(wait)
                        continue
                    if resp.status < 500:
                        return {"asset_id": asset_id, "status": last_status}
            except Exception as e:
                last_status = f"error: {e}"

            if attempt < max_retries - 1:
                await asyncio.sleep(rate_limit * (2 ** attempt) + 1)
        else:
            return {"asset_id": asset_id, "status": last_status}

        try:
            out_path.write_bytes(content)
            await asyncio.sleep(rate_limit)
            return {"asset_id": asset_id, "status": "ok", "path": str(out_path)}
        except Exception as e:
            out_path.unlink(missing_ok=True)
            return {"asset_id": asset_id, "status": f"error: {e}"}


async def download_species_batch(
    session: aiohttp.ClientSession,
    species_records: list[dict],
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
) -> list[dict]:
    """1種分の録音をバッチダウンロードする。"""
    tasks = [
        download_one(session, rec, cfg, semaphore, rate_limit)
        for rec in species_records
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


def _save_progress(progress_path: Path, data: dict):
    """進捗ファイルをアトミックに保存する。"""
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(progress_path)


async def download_all(cfg: dict, args):
    """メタデータに基づき全録音をダウンロードする。"""
    ml_cfg = cfg["macaulay"]
    metadata_dir = nas_path(cfg, ml_cfg["metadata_dir"])
    metadata_path = metadata_dir / "ml_metadata.parquet"

    if not metadata_path.exists():
        print("Error: metadata not found. Run --metadata-only first.")
        sys.exit(1)

    df = load_metadata(metadata_path)
    print(f"Loaded {len(df)} recordings from metadata")

    # レーティングフィルタ
    if args.min_rating > 0:
        before = len(df)
        df["_rating"] = pd.to_numeric(df["ml_rating"], errors="coerce").fillna(0)
        df = df[df["_rating"] >= args.min_rating].drop(columns=["_rating"])
        print(f"Rating filter (≥{args.min_rating}): {before} → {len(df)} recordings")

    # 種ごとにグループ化
    species_groups = df.groupby("ebird_species_code")
    species_list = sorted(species_groups.groups.keys())
    if args.limit > 0:
        species_list = species_list[:args.limit]

    total_recordings = sum(len(species_groups.get_group(sp)) for sp in species_list)

    # ── ドライラン ──
    if args.dry_run:
        print(f"\n[DRY RUN] Would download {total_recordings} recordings "
              f"for {len(species_list)} species")
        est_gb = total_recordings * 2.5 / 1024  # ~2.5 MB/file MP3
        print(f"  Estimated storage: ~{est_gb:.0f} GB")
        top10 = df[df["ebird_species_code"].isin(species_list)] \
            .groupby("ebird_species_code").size().nlargest(10)
        print(f"\nTop 10 species by recording count:")
        for sp, cnt in top10.items():
            print(f"  {sp}: {cnt}")
        return

    dl_cfg = ml_cfg["download"]
    max_concurrent = dl_cfg["max_concurrent"]
    rate_limit = dl_cfg["rate_limit_sec"]
    semaphore = asyncio.Semaphore(max_concurrent)

    # 進捗ファイル
    progress_path = metadata_dir / "download_progress.json"
    progress = {"files": {}, "completed_species": []}
    if progress_path.exists():
        with open(progress_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "files" in raw:
            progress = raw
        else:
            progress["files"] = raw

    all_ok_results = progress["files"]
    completed_species = set(progress["completed_species"])

    remaining_species = [sp for sp in species_list if sp not in completed_species]
    skipped_species = len(species_list) - len(remaining_species)
    remaining_recordings = sum(
        len(species_groups.get_group(sp)) for sp in remaining_species
    )

    print(f"\nDownloading {total_recordings} recordings "
          f"for {len(species_list)} species")
    print(f"  Concurrent: {max_concurrent}, Rate limit: {rate_limit}s")
    if skipped_species > 0:
        print(f"  Resuming: {skipped_species} species already completed, "
              f"{len(all_ok_results)} files tracked")
        print(f"  Remaining: {remaining_recordings} recordings "
              f"for {len(remaining_species)} species")
    print()

    if not remaining_species:
        print("All species already completed!")
        return

    # シグナルハンドリング
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForce exit (progress already saved)")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nShutdown requested — finishing current species, saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total_stats = {"ok": 0, "exists": 0, "error": 0, "http_error": 0}
    connector = aiohttp.TCPConnector(limit=max_concurrent, force_close=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(remaining_species, desc="Species", unit="sp")
        for sp_idx, sp_code in enumerate(pbar):
            if shutdown_requested:
                print(f"\nStopping after {sp_idx} species (signal received)")
                break

            sp_df = species_groups.get_group(sp_code)
            records = sp_df.to_dict("records")
            pbar.set_postfix_str(f"{sp_code} ({len(records)} recs)")

            try:
                results = await download_species_batch(
                    session, records, cfg, semaphore, rate_limit,
                )
            except Exception as e:
                print(f"\n  Error processing {sp_code}: {e}")
                total_stats["error"] += len(records)
                _save_progress(progress_path, {
                    "files": all_ok_results,
                    "completed_species": sorted(completed_species),
                })
                continue

            sp_ok = 0
            sp_skipped = 0
            for r in results:
                status = r["status"]
                if status == "ok":
                    total_stats["ok"] += 1
                    sp_ok += 1
                elif status == "exists":
                    total_stats["exists"] += 1
                    sp_ok += 1
                elif status == "no_url":
                    # URL欠損はリトライ不可 → 成功扱い
                    sp_skipped += 1
                elif status.startswith("http_"):
                    total_stats["http_error"] += 1
                else:
                    total_stats["error"] += 1

                if status in ("ok", "exists") and "path" in r:
                    all_ok_results[r["asset_id"]] = r["path"]

            # 成功 + スキップ = 全件なら種完了
            if sp_ok + sp_skipped == len(records):
                completed_species.add(sp_code)
            if sp_skipped > 0:
                total_stats["no_url"] = total_stats.get("no_url", 0) + sp_skipped

            _save_progress(progress_path, {
                "files": all_ok_results,
                "completed_species": sorted(completed_species),
            })

    # 結果サマリ
    print(f"\n{'='*60}")
    print("Macaulay Library Download Results")
    print(f"{'='*60}")
    for status, cnt in sorted(total_stats.items()):
        print(f"  {status}: {cnt}")
    print(f"  Total tracked files: {len(all_ok_results)}")
    print(f"  Completed species: {len(completed_species)}/{len(species_list)}")

    # ファイルパスをメタデータに反映（フィルタ前の全データに対して更新）
    if all_ok_results:
        full_df = load_metadata(metadata_path)
        full_df["file_path"] = full_df["ml_asset_id"].apply(
            lambda x: all_ok_results.get(str(x), "")
        )
        _save_metadata_atomic(full_df, metadata_path)

    if shutdown_requested:
        print(f"\n  Download interrupted — run again to continue.")
        sys.exit(1)
    if len(completed_species) < len(species_list):
        incomplete = len(species_list) - len(completed_species)
        print(f"\n  {incomplete} species incomplete — run again to retry.")
        sys.exit(2)


# ── Main ────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config()

    if args.metadata_only:
        collect_metadata(cfg, args)
    elif args.download:
        asyncio.run(download_all(cfg, args))
    else:
        print("Specify --metadata-only or --download")
        print("  --metadata-only: collect metadata from ML Search API")
        print("  --download:      download audio files (after metadata)")
        sys.exit(1)


if __name__ == "__main__":
    main()
