"""
Xeno-canto メタデータ収集 & 音声ダウンロード。

使い方:
  # メタデータのみ収集（全690種）
  python collect_xc.py --metadata-only

  # テスト: 5種だけ
  python collect_xc.py --metadata-only --limit 5

  # 音声ダウンロード（メタデータ収集後）
  python collect_xc.py --download

  # MP3のまま保持（WAV変換しない、ストレージ節約）
  python collect_xc.py --download --format mp3

  # テスト: 3種だけダウンロード
  python collect_xc.py --download --limit 3

  # ドライラン（ダウンロード対象の確認のみ）
  python collect_xc.py --download --dry-run
"""

import argparse
import asyncio
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

from utils import (
    audio_file_path,
    get_target_species,
    is_in_japan,
    load_api_key,
    load_config,
    load_metadata,
    nas_path,
    save_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Xeno-canto データ収集")
    parser.add_argument("--metadata-only", action="store_true",
                        help="メタデータのみ収集（音声DLしない）")
    parser.add_argument("--download", action="store_true",
                        help="音声ダウンロード（メタデータ収集済み前提）")
    parser.add_argument("--limit", type=int, default=0,
                        help="テスト用: 処理する種数の上限")
    parser.add_argument("--resume", action="store_true",
                        help="既存メタデータから再開")
    parser.add_argument("--format", choices=["wav", "mp3"], default="mp3",
                        help="保存フォーマット（mp3=元ファイル保持, wav=変換）")
    parser.add_argument("--dry-run", action="store_true",
                        help="ダウンロード対象の確認のみ（実際にはDLしない）")
    return parser.parse_args()


# ── メタデータ収集 ──────────────────────────────────────────

def fetch_species_metadata(
    scientific_name: str,
    api_key: str,
    api_url: str,
    per_page: int = 500,
    rate_limit: float = 1.0,
) -> list[dict]:
    """1種について全ページのメタデータを取得する。"""
    import urllib.request
    import urllib.parse

    all_recordings = []
    page = 1

    while True:
        params = urllib.parse.urlencode({
            "query": f'sp:"{scientific_name}"',
            "key": api_key,
            "per_page": per_page,
            "page": page,
        })
        url = f"{api_url}?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  Error fetching {scientific_name} page {page}: {e}")
            break

        recordings = data.get("recordings", [])
        if not recordings:
            break

        all_recordings.extend(recordings)

        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break

        page += 1
        time.sleep(rate_limit)

    return all_recordings


def parse_xc_recording(rec: dict, species_row: pd.Series, cfg: dict) -> dict:
    """XC APIレスポンスの1録音を統一フォーマットに変換する。"""
    xc_id = str(rec.get("id", ""))
    lat = _safe_float(rec.get("lat"))
    lon = _safe_float(rec.get("lon"))

    # サンプルレートをパース（"44100" や "48000 (Hz)" 等）
    sr_raw = rec.get("smp", "")
    sample_rate = _parse_sample_rate(sr_raw)

    # 録音長（秒）
    length_raw = rec.get("length", "0:00")
    duration_sec = _parse_duration(length_raw)

    return {
        "recording_id": f"xc:{xc_id}",
        "source": "xeno-canto",
        "ebird_species_code": species_row["ebird_species_code"],
        "scientific_name": species_row["scientific_name"],
        "japanese_name": species_row["japanese_name"],
        "latitude": lat,
        "longitude": lon,
        "country": rec.get("cnt", ""),
        "is_japan": is_in_japan(lat, lon, cfg),
        "duration_sec": duration_sec,
        "sample_rate": sample_rate,
        "quality": rec.get("q", ""),
        "license": rec.get("lic", ""),
        "file_path": "",  # DL時に設定
        "vocalization_type": rec.get("type", ""),
        # XC固有の追加情報
        "xc_id": xc_id,
        "xc_url": rec.get("url", ""),
        "xc_file_url": rec.get("file", ""),
        "xc_file_name": rec.get("file-name", ""),
        "xc_sono_url": rec.get("sono", {}).get("small", "") if isinstance(rec.get("sono"), dict) else "",
        "xc_gen": rec.get("gen", ""),
        "xc_sp": rec.get("sp", ""),
        "xc_subspecies": rec.get("ssp", ""),
        "xc_recordist": rec.get("rec", ""),
        "xc_date": rec.get("date", ""),
        "xc_time": rec.get("time", ""),
        "xc_loc": rec.get("loc", ""),
        "xc_rmk": rec.get("rmk", ""),
        "xc_also": rec.get("also", []),
    }


def _safe_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_sample_rate(sr_raw: str) -> int | None:
    """サンプルレート文字列をパース。"""
    if not sr_raw:
        return None
    # "44100", "48000 (Hz)" 等
    import re
    m = re.search(r"(\d+)", str(sr_raw))
    return int(m.group(1)) if m else None


def _parse_duration(length_str: str) -> float:
    """XCの録音長文字列 "m:ss" or "h:mm:ss" を秒に変換。"""
    if not length_str:
        return 0.0
    parts = length_str.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return 0.0


def collect_metadata(cfg: dict, args) -> pd.DataFrame:
    """全対象種のXCメタデータを収集する。"""
    api_key = load_api_key(cfg)
    xc_cfg = cfg["xeno_canto"]
    api_url = xc_cfg["api_url"]
    per_page = xc_cfg["per_page"]
    rate_limit = xc_cfg["rate_limit_sec"]

    species_df = get_target_species(cfg)
    if args.limit > 0:
        species_df = species_df.head(args.limit)

    print(f"Collecting XC metadata for {len(species_df)} species...")

    # 既存メタデータの読み込み（再開用）
    metadata_path = nas_path(cfg, xc_cfg["metadata_dir"]) / "xc_metadata.parquet"
    existing_species = set()
    existing_records = []
    if args.resume and metadata_path.exists():
        existing_df = load_metadata(metadata_path)
        existing_species = set(existing_df["ebird_species_code"].unique())
        existing_records = existing_df.to_dict("records")
        print(f"  Resuming: {len(existing_species)} species already collected")

    all_records = existing_records
    no_hit_species = []
    errors = []

    for idx, row in tqdm(species_df.iterrows(), total=len(species_df), desc="Species"):
        sp_code = row["ebird_species_code"]
        if sp_code in existing_species:
            continue

        # まず scientific_name（第8版の学名）で検索
        recordings = fetch_species_metadata(
            row["scientific_name"], api_key, api_url, per_page, rate_limit,
        )

        # ヒットしなければ ebird_sciname でリトライ
        if not recordings and row["ebird_sciname"] != row["scientific_name"]:
            time.sleep(rate_limit)
            recordings = fetch_species_metadata(
                row["ebird_sciname"], api_key, api_url, per_page, rate_limit,
            )

        if not recordings:
            no_hit_species.append({
                "species_code": sp_code,
                "scientific_name": row["scientific_name"],
                "japanese_name": row["japanese_name"],
            })
            continue

        for rec in recordings:
            parsed = parse_xc_recording(rec, row, cfg)
            all_records.append(parsed)

        time.sleep(rate_limit)

    # DataFrame化
    df = pd.DataFrame(all_records)

    # xc_also はリスト型 → JSON文字列化（Parquet保存用）
    if "xc_also" in df.columns:
        df["xc_also"] = df["xc_also"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else str(x)
        )

    # 保存
    save_metadata(df, metadata_path)

    # ── 統計レポート ──
    print("\n" + "=" * 60)
    print("XC Metadata Collection Report")
    print("=" * 60)
    total_species_found = df["ebird_species_code"].nunique() if len(df) > 0 else 0
    total_recordings = len(df)
    print(f"Total species with recordings: {total_species_found}/{len(species_df)}")
    print(f"Total recordings: {total_recordings}")

    if len(df) > 0:
        # 品質分布
        print(f"\nQuality distribution:")
        for q, cnt in df["quality"].value_counts().sort_index().items():
            print(f"  {q}: {cnt} ({cnt/total_recordings*100:.1f}%)")

        # 国内/国外
        japan_count = df["is_japan"].sum()
        print(f"\nJapan recordings: {japan_count} ({japan_count/total_recordings*100:.1f}%)")
        print(f"Overseas recordings: {total_recordings - japan_count}")

        # 種ごとの録音数分布
        per_species = df.groupby("ebird_species_code").size()
        print(f"\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

        # 録音数の少ない種 top 10
        bottom10 = per_species.nsmallest(10)
        print(f"\nSpecies with fewest recordings:")
        for sp_code, cnt in bottom10.items():
            sp_row = species_df[species_df["ebird_species_code"] == sp_code]
            if len(sp_row) > 0:
                jp = sp_row.iloc[0]["japanese_name"]
                print(f"  {sp_code}: {cnt} recordings ({jp})")

    if no_hit_species:
        print(f"\nSpecies with NO recordings ({len(no_hit_species)}):")
        for sp in no_hit_species:
            print(f"  {sp['scientific_name']} ({sp['japanese_name']})")

    return df


# ── 音声ダウンロード ────────────────────────────────────────

def filter_for_download(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ダウンロード対象をフィルタする（品質・ライセンス）。"""
    xc_cfg = cfg["xeno_canto"]
    dl_filter = xc_cfg["download_filter"]

    quality_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "": 6}
    min_q = dl_filter["min_quality"]
    max_q_val = quality_order.get(min_q, 3)

    # 品質フィルタ
    df = df.copy()
    df["_q_val"] = df["quality"].map(quality_order).fillna(6)
    filtered = df[df["_q_val"] <= max_q_val].copy()
    filtered = filtered.drop(columns=["_q_val"])

    # ライセンスフィルタ
    allowed = dl_filter["allowed_licenses"]
    if allowed:
        license_mask = filtered["license"].apply(
            lambda lic: any(pat in str(lic) for pat in allowed)
        )
        filtered = filtered[license_mask]

    before = len(df)
    after = len(filtered)
    print(f"Download filter: {before} → {after} recordings "
          f"(quality ≤ {min_q}, CC license)")

    return filtered


async def download_one(
    session: aiohttp.ClientSession,
    row: dict,
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    out_ext: str = "mp3",
    max_retries: int = 3,
) -> dict | None:
    """1つの録音をダウンロードする。HTTPエラー時はリトライ。"""
    async with semaphore:
        xc_id = row["xc_id"]
        file_url = row["xc_file_url"]
        species_code = row["ebird_species_code"]

        if not file_url:
            return None

        # 出力パス
        out_path = audio_file_path(
            cfg, "xeno-canto", species_code, f"xc_{xc_id}", ext=out_ext,
        )
        if out_path.exists():
            return {"xc_id": xc_id, "status": "exists", "path": str(out_path)}

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if file_url.startswith("//"):
            file_url = "https:" + file_url

        last_status = "error"
        for attempt in range(max_retries):
            try:
                async with session.get(file_url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        break
                    last_status = f"http_{resp.status}"
                    # 5xx系はリトライ、4xx系は即失敗
                    if resp.status < 500:
                        return {"xc_id": xc_id, "status": last_status}
            except Exception as e:
                last_status = f"error: {e}"

            # リトライ前に待機（指数バックオフ）
            if attempt < max_retries - 1:
                await asyncio.sleep(rate_limit * (2 ** attempt) + 1)
        else:
            # 全リトライ失敗
            return {"xc_id": xc_id, "status": last_status}

        try:
            if out_ext == "mp3":
                out_path.write_bytes(content)
            else:
                # MP3→WAV変換
                mp3_path = out_path.with_suffix(".mp3")
                mp3_path.write_bytes(content)
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3_path),
                     "-acodec", "pcm_s16le", str(out_path)],
                    capture_output=True, timeout=60,
                )
                mp3_path.unlink(missing_ok=True)
                if result.returncode != 0:
                    out_path.unlink(missing_ok=True)
                    return {"xc_id": xc_id, "status": "ffmpeg_error"}

            await asyncio.sleep(rate_limit)
            return {"xc_id": xc_id, "status": "ok", "path": str(out_path)}

        except Exception as e:
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".mp3").unlink(missing_ok=True)
            return {"xc_id": xc_id, "status": f"error: {e}"}


async def download_species_batch(
    session: aiohttp.ClientSession,
    species_records: list[dict],
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    out_ext: str,
) -> list[dict]:
    """1種分の録音をバッチダウンロードする。"""
    tasks = [
        download_one(session, rec, cfg, semaphore, rate_limit, out_ext)
        for rec in species_records
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
    return results


def _save_progress(progress_path: Path, data: dict):
    """進捗ファイルをアトミックに保存する。"""
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(progress_path)


async def download_all(cfg: dict, args):
    """メタデータに基づき全録音をダウンロードする。種ごとにバッチ処理。"""
    xc_cfg = cfg["xeno_canto"]
    metadata_path = nas_path(cfg, xc_cfg["metadata_dir"]) / "xc_metadata.parquet"

    if not metadata_path.exists():
        print("Error: metadata not found. Run --metadata-only first.")
        sys.exit(1)

    df = load_metadata(metadata_path)
    filtered = filter_for_download(df, cfg)

    out_ext = args.format

    # 種ごとにグループ化
    species_groups = filtered.groupby("ebird_species_code")
    species_list = sorted(species_groups.groups.keys())
    if args.limit > 0:
        species_list = species_list[:args.limit]

    total_recordings = sum(len(species_groups.get_group(sp)) for sp in species_list)

    # ── ドライラン ──
    if args.dry_run:
        print(f"\n[DRY RUN] Would download {total_recordings} recordings "
              f"for {len(species_list)} species")
        print(f"  Format: {out_ext}")
        if out_ext == "mp3":
            est_gb = total_recordings * 2.5 / 1024
        else:
            est_gb = total_recordings * 12 / 1024
        print(f"  Estimated storage: ~{est_gb:.0f} GB")
        print(f"\nTop 10 species by recording count:")
        top10 = filtered.groupby("ebird_species_code").size().nlargest(10)
        for sp, cnt in top10.items():
            print(f"  {sp}: {cnt}")
        return

    # 依存チェック
    if out_ext == "wav":
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            print("Error: ffmpeg is required for WAV conversion. "
                  "Install it or use --format mp3")
            sys.exit(1)

    dl_cfg = xc_cfg["download"]
    max_concurrent = dl_cfg["max_concurrent"]
    rate_limit = dl_cfg["rate_limit_sec"]
    semaphore = asyncio.Semaphore(max_concurrent)

    # 進捗ファイル（DL済みファイル + 完了種の記録）
    progress_path = nas_path(cfg, xc_cfg["metadata_dir"]) / "download_progress.json"
    progress = {"files": {}, "completed_species": []}
    if progress_path.exists():
        with open(progress_path) as f:
            raw = json.load(f)
        # 旧フォーマット互換（dict直書き → 新フォーマットに移行）
        if isinstance(raw, dict) and "files" in raw:
            progress = raw
        else:
            progress["files"] = raw

    all_ok_results = progress["files"]
    completed_species = set(progress["completed_species"])

    # 完了済み種をスキップして残りを算出
    remaining_species = [sp for sp in species_list if sp not in completed_species]
    skipped_species = len(species_list) - len(remaining_species)
    remaining_recordings = sum(
        len(species_groups.get_group(sp)) for sp in remaining_species
    )

    print(f"\nDownloading {total_recordings} recordings "
          f"for {len(species_list)} species")
    print(f"  Format: {out_ext}, Concurrent: {max_concurrent}")
    if skipped_species > 0:
        print(f"  Resuming: {skipped_species} species already completed, "
              f"{len(all_ok_results)} files tracked")
        print(f"  Remaining: {remaining_recordings} recordings "
              f"for {len(remaining_species)} species")
    print(f"  Existing files will be skipped")
    print()

    if not remaining_species:
        print("All species already completed!")
        return

    # シグナルハンドリング（Ctrl+C で進捗保存して終了）
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
                    session, records, cfg, semaphore, rate_limit, out_ext,
                )
            except Exception as e:
                print(f"\n  Error processing {sp_code}: {e}")
                print(f"  Skipping to next species...")
                total_stats["error"] += len(records)
                # 進捗保存してから次へ
                _save_progress(progress_path, {
                    "files": all_ok_results,
                    "completed_species": sorted(completed_species),
                })
                continue

            # 結果集計
            sp_ok = 0
            for r in results:
                status = r["status"]
                if status == "ok":
                    total_stats["ok"] += 1
                    sp_ok += 1
                elif status == "exists":
                    total_stats["exists"] += 1
                    sp_ok += 1
                elif status.startswith("http_"):
                    total_stats["http_error"] += 1
                else:
                    total_stats["error"] += 1

                if status in ("ok", "exists") and "path" in r:
                    all_ok_results[r["xc_id"]] = r["path"]

            # 種の完了判定（エラーなし or 全ファイルが ok/exists）
            if sp_ok == len(records):
                completed_species.add(sp_code)

            # 毎種ごとに進捗保存
            _save_progress(progress_path, {
                "files": all_ok_results,
                "completed_species": sorted(completed_species),
            })

    # 結果サマリ
    print(f"\n{'='*60}")
    print("XC Download Results")
    print(f"{'='*60}")
    for status, cnt in sorted(total_stats.items()):
        print(f"  {status}: {cnt}")
    print(f"  Total tracked files: {len(all_ok_results)}")
    print(f"  Completed species: {len(completed_species)}/{len(species_list)}")

    # ファイルパスをメタデータに反映
    if all_ok_results:
        df["file_path"] = df["xc_id"].apply(lambda x: all_ok_results.get(x, ""))
        save_metadata(df, metadata_path)

    # 未完了種がある場合は非ゼロ終了（シェルスクリプトがリトライ）
    if shutdown_requested:
        print(f"\n  Download interrupted — run again to continue.")
        sys.exit(1)
    if len(completed_species) < len(species_list):
        incomplete = len(species_list) - len(completed_species)
        print(f"\n  {incomplete} species incomplete (HTTP errors etc.) — "
              f"run again to retry.")
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
        print("  --metadata-only: collect metadata from XC API")
        print("  --download:      download audio files (after metadata)")
        sys.exit(1)


if __name__ == "__main__":
    main()
