"""
iNaturalist API メタデータ収集 & 音声ダウンロード。

iNat Sounds 2024 コンペデータを補完するため、
iNaturalist API から直接 research-grade の鳥類音声を収集する。

使い方:
  # メタデータのみ収集（全688種）
  python collect_inat_api.py --metadata-only

  # テスト: 3種だけ
  python collect_inat_api.py --metadata-only --limit 3

  # 音声ダウンロード（メタデータ収集後）
  python collect_inat_api.py --download

  # ドライラン（ダウンロード対象の確認のみ）
  python collect_inat_api.py --download --dry-run

  # テスト: 1種だけダウンロード
  python collect_inat_api.py --download --limit 1
"""

import argparse
import asyncio
import json
import signal
import sys
import time
import urllib.parse
import urllib.request
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
    save_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="iNaturalist API データ収集")
    parser.add_argument("--metadata-only", action="store_true",
                        help="メタデータのみ収集（音声DLしない）")
    parser.add_argument("--download", action="store_true",
                        help="音声ダウンロード（メタデータ収集済み前提）")
    parser.add_argument("--limit", type=int, default=0,
                        help="テスト用: 処理する種数の上限")
    parser.add_argument("--dry-run", action="store_true",
                        help="ダウンロード対象の確認のみ（実際にはDLしない）")
    return parser.parse_args()


# ── 重複チェック ─────────────────────────────────────────────

def load_existing_inat_sound_ids(cfg: dict) -> set:
    """既存 iNat Sounds 2024 メタデータから audio ID セットを取得する。"""
    inat_cfg = cfg.get("inat_sounds", {})
    ann_dir = inat_cfg.get("annotations_dir", "")
    if not ann_dir:
        return set()

    metadata_path = nas_path(cfg, ann_dir) / "inat_metadata.parquet"
    if not metadata_path.exists():
        return set()

    try:
        df = load_metadata(metadata_path)
        if "inat_audio_id" in df.columns:
            return set(df["inat_audio_id"].dropna().astype(int))
    except Exception as e:
        print(f"  Warning: could not load existing iNat metadata: {e}")

    return set()


# ── メタデータ収集 ──────────────────────────────────────────

def fetch_species_observations(
    scientific_name: str,
    api_url: str,
    per_page: int = 200,
    rate_limit: float = 1.0,
    max_retries: int = 3,
) -> list[dict]:
    """1種について全 research-grade 音声付き観察を取得する。

    id_above パラメータで 10K 制限を回避してページネーション。
    """
    all_observations = []
    id_above = 0

    while True:
        params = {
            "sounds": "true",
            "quality_grade": "research",
            "taxon_name": scientific_name,
            "per_page": per_page,
            "order": "asc",
            "order_by": "id",
        }
        if id_above > 0:
            params["id_above"] = id_above

        url = f"{api_url}?{urllib.parse.urlencode(params)}"

        data = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ToriNet/1.0 (bird bioacoustics research)"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited (429), waiting {wait}s...")
                    time.sleep(wait)
                elif e.code == 422:
                    # Unprocessable — taxon name not found
                    return all_observations
                else:
                    print(f"  HTTP {e.code} for {scientific_name} "
                          f"(id_above={id_above}), attempt {attempt+1}")
                    time.sleep(rate_limit * 2)
            except Exception as e:
                print(f"  Error fetching {scientific_name} "
                      f"(id_above={id_above}): {e}, attempt {attempt+1}")
                time.sleep(rate_limit * 2)

        if data is None:
            break

        results = data.get("results", [])
        if not results:
            break

        all_observations.extend(results)

        # 次のページへ
        last_id = results[-1]["id"]
        id_above = last_id

        if len(results) < per_page:
            break

        time.sleep(rate_limit)

    return all_observations


def parse_observation_sounds(
    obs: dict,
    species_row: pd.Series,
    cfg: dict,
    existing_sound_ids: set,
) -> list[dict]:
    """1つの観察から全サウンドのメタデータを抽出する。"""
    records = []
    obs_id = obs.get("id", 0)

    # 位置情報（"lat,lon" 文字列）
    location = obs.get("location")
    lat, lon = None, None
    if location:
        try:
            parts = str(location).split(",")
            lat = float(parts[0])
            lon = float(parts[1])
        except (ValueError, IndexError):
            pass

    sounds = obs.get("sounds", [])
    for sound in sounds:
        sound_id = sound.get("id")
        if sound_id is None:
            continue

        is_duplicate = int(sound_id) in existing_sound_ids

        file_url = sound.get("file_url", "")
        license_code = sound.get("license_code", "")

        records.append({
            "recording_id": f"inat_api:{sound_id}",
            "source": "inat-api",
            "ebird_species_code": species_row["ebird_species_code"],
            "scientific_name": species_row["scientific_name"],
            "japanese_name": species_row["japanese_name"],
            "latitude": lat,
            "longitude": lon,
            "country": "",
            "is_japan": is_in_japan(lat, lon, cfg),
            "duration_sec": None,
            "sample_rate": None,
            "quality": "",
            "license": license_code,
            "file_path": "",
            "vocalization_type": "",
            # iNat API 固有
            "inat_api_sound_id": int(sound_id),
            "inat_api_obs_id": int(obs_id),
            "inat_api_file_url": file_url,
            "inat_api_attribution": sound.get("attribution", ""),
            "inat_api_license_code": license_code,
            "inat_api_observer": obs.get("user", {}).get("login", ""),
            "inat_api_observed_on": obs.get("observed_on", ""),
            "inat_api_place_guess": obs.get("place_guess", ""),
            "inat_api_taxon_id": obs.get("taxon", {}).get("id"),
            "inat_api_is_duplicate_sounds2024": is_duplicate,
        })

    return records


def _save_progress(progress_path: Path, data: dict):
    """進捗ファイルをアトミックに保存する。"""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(progress_path)


def collect_metadata(cfg: dict, args) -> pd.DataFrame:
    """全対象種の iNat API メタデータを収集する。"""
    api_cfg = cfg["inat_api"]
    api_url = api_cfg["api_url"]
    per_page = api_cfg["per_page"]
    rate_limit = api_cfg["rate_limit_sec"]

    species_df = get_target_species(cfg)
    if args.limit > 0:
        species_df = species_df.head(args.limit)

    print(f"Collecting iNat API metadata for {len(species_df)} species...")

    # 既存 iNat Sounds 2024 のサウンドID（重複チェック用）
    existing_sound_ids = load_existing_inat_sound_ids(cfg)
    if existing_sound_ids:
        print(f"  Loaded {len(existing_sound_ids)} existing iNat Sounds 2024 "
              f"audio IDs for dedup")

    # 進捗ファイル（再開用）
    metadata_path = (
        nas_path(cfg, api_cfg["metadata_dir"]) / "inat_api_metadata.parquet"
    )
    progress_path = metadata_path.parent / "collection_progress.json"

    completed_species: set[str] = set()
    existing_records: list[dict] = []

    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        completed_species = set(progress.get("completed_species", []))
        print(f"  Resuming: {len(completed_species)} species already collected")

    if metadata_path.exists() and completed_species:
        existing_df = load_metadata(metadata_path)
        existing_records = existing_df.to_dict("records")

    all_records = existing_records
    no_hit_species: list[dict] = []

    # シグナルハンドリング
    shutdown_requested = False

    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n\nForce exit (progress already saved)")
            sys.exit(1)
        shutdown_requested = True
        print("\n\nShutdown requested — finishing current species, "
              "saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    remaining = [
        (idx, row)
        for idx, row in species_df.iterrows()
        if row["ebird_species_code"] not in completed_species
    ]

    print(f"  Remaining: {len(remaining)} species to process")

    pbar = tqdm(remaining, desc="Species", unit="sp")
    for _idx, row in pbar:
        if shutdown_requested:
            break

        sp_code = row["ebird_species_code"]
        pbar.set_postfix_str(sp_code)

        # まず scientific_name（第8版の学名）で検索
        observations = fetch_species_observations(
            row["scientific_name"], api_url, per_page, rate_limit,
        )

        # ヒットしなければ ebird_sciname でリトライ
        if not observations and row["ebird_sciname"] != row["scientific_name"]:
            time.sleep(rate_limit)
            observations = fetch_species_observations(
                row["ebird_sciname"], api_url, per_page, rate_limit,
            )

        if not observations:
            no_hit_species.append({
                "species_code": sp_code,
                "scientific_name": row["scientific_name"],
                "japanese_name": row["japanese_name"],
            })
        else:
            for obs in observations:
                records = parse_observation_sounds(
                    obs, row, cfg, existing_sound_ids,
                )
                all_records.extend(records)

        completed_species.add(sp_code)

        # 進捗保存（種ごと）
        _save_progress(progress_path, {
            "completed_species": sorted(completed_species),
            "total_records": len(all_records),
        })

        time.sleep(rate_limit)

    # DataFrame化 & 保存
    df = pd.DataFrame(all_records)
    if len(df) > 0:
        save_metadata(df, metadata_path)

    # ── 統計レポート ──
    print("\n" + "=" * 60)
    print("iNat API Metadata Collection Report")
    print("=" * 60)

    total_recordings = len(df)
    total_species = (
        df["ebird_species_code"].nunique() if total_recordings > 0 else 0
    )
    print(f"Total species with recordings: {total_species}/{len(species_df)}")
    print(f"Total sound recordings: {total_recordings}")

    if total_recordings > 0:
        # 重複統計
        dup_count = int(df["inat_api_is_duplicate_sounds2024"].sum())
        unique_count = total_recordings - dup_count
        print(f"\nDuplication with iNat Sounds 2024:")
        print(f"  Duplicates: {dup_count}")
        print(f"  New (unique): {unique_count}")

        # ライセンス分布
        print("\nLicense distribution:")
        for lic, cnt in df["license"].value_counts().head(10).items():
            print(f"  {lic}: {cnt} ({cnt/total_recordings*100:.1f}%)")

        # 国内/国外
        japan_count = int(df["is_japan"].sum())
        print(f"\nJapan recordings: {japan_count} "
              f"({japan_count/total_recordings*100:.1f}%)")

        # 種ごとの録音数分布
        per_species = df.groupby("ebird_species_code").size()
        print("\nRecordings per species:")
        print(f"  Mean:   {per_species.mean():.1f}")
        print(f"  Median: {per_species.median():.0f}")
        print(f"  Min:    {per_species.min()}")
        print(f"  Max:    {per_species.max()}")

    if no_hit_species:
        print(f"\nSpecies with NO recordings ({len(no_hit_species)}):")
        for sp in no_hit_species[:30]:
            print(f"  {sp['scientific_name']} ({sp['japanese_name']})")
        if len(no_hit_species) > 30:
            print(f"  ... and {len(no_hit_species) - 30} more")

    if shutdown_requested:
        print("\n  Collection interrupted — run again to continue.")
        sys.exit(1)

    return df


# ── 音声ダウンロード ────────────────────────────────────────

def filter_for_download(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ダウンロード対象をフィルタする（ライセンス、重複除外）。"""
    api_cfg = cfg["inat_api"]
    allowed = api_cfg.get("allowed_licenses", [])

    filtered = df.copy()

    # iNat Sounds 2024 との重複を除外
    if "inat_api_is_duplicate_sounds2024" in filtered.columns:
        before_dedup = len(filtered)
        filtered = filtered[~filtered["inat_api_is_duplicate_sounds2024"]].copy()
        dedup_removed = before_dedup - len(filtered)
        if dedup_removed > 0:
            print(f"Dedup: removed {dedup_removed} duplicates "
                  f"with iNat Sounds 2024")

    # ライセンスフィルタ
    if allowed:
        before_lic = len(filtered)
        license_mask = filtered["license"].apply(
            lambda lic: (
                any(a in str(lic).lower() for a in allowed)
                if pd.notna(lic)
                else False
            )
        )
        filtered = filtered[license_mask].copy()
        lic_removed = before_lic - len(filtered)
        if lic_removed > 0:
            print(f"License filter: removed {lic_removed} non-CC recordings")

    # file_url が空のものを除外
    if "inat_api_file_url" in filtered.columns:
        filtered = filtered[
            filtered["inat_api_file_url"].str.len() > 0
        ].copy()

    print(f"Download target: {len(filtered)} recordings "
          f"(from {len(df)} total)")
    return filtered


class BandwidthTracker:
    """5GB/hour メディアダウンロード制限を管理する。"""

    def __init__(self, hourly_limit_gb: float = 4.5):
        self.hourly_limit_bytes = int(hourly_limit_gb * 1024**3)
        self.window_start = time.monotonic()
        self.window_bytes = 0

    def add(self, nbytes: int):
        self._maybe_reset_window()
        self.window_bytes += nbytes

    def _maybe_reset_window(self):
        elapsed = time.monotonic() - self.window_start
        if elapsed >= 3600:
            self.window_start = time.monotonic()
            self.window_bytes = 0

    def should_pause(self) -> bool:
        self._maybe_reset_window()
        return self.window_bytes >= self.hourly_limit_bytes

    def seconds_until_reset(self) -> float:
        elapsed = time.monotonic() - self.window_start
        return max(0, 3600 - elapsed)

    def used_gb(self) -> float:
        return self.window_bytes / 1024**3


async def download_one(
    session: aiohttp.ClientSession,
    row: dict,
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    max_retries: int = 5,
) -> dict:
    """1つの録音をダウンロードする（リトライ付き）。"""
    async with semaphore:
        sound_id = row["inat_api_sound_id"]
        file_url = row["inat_api_file_url"]
        species_code = row["ebird_species_code"]

        if not file_url:
            return {"sound_id": sound_id, "status": "no_url", "size": 0}

        # 出力パス: audio/inat-api/audio/{species_code}/inatapi_{sound_id}.m4a
        audio_dir = nas_path(cfg, cfg["inat_api"]["audio_dir"])
        ext = Path(file_url.split("?")[0]).suffix or ".m4a"
        out_path = audio_dir / species_code / f"inatapi_{sound_id}{ext}"

        if out_path.exists():
            return {
                "sound_id": sound_id,
                "status": "exists",
                "path": str(out_path),
                "size": 0,
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Retryable status codes (rate limiting / server errors)
        _RETRYABLE = {403, 429, 500, 502, 503, 504}

        last_status = None
        for attempt in range(max_retries):
            try:
                async with session.get(file_url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        out_path.write_bytes(content)
                        await asyncio.sleep(rate_limit)
                        return {
                            "sound_id": sound_id,
                            "status": "ok",
                            "path": str(out_path),
                            "size": len(content),
                        }
                    last_status = resp.status
                    if resp.status in _RETRYABLE:
                        if resp.status in (403, 429):
                            wait = 60 * (attempt + 1)
                        else:
                            wait = rate_limit * (2 ** attempt) + 1
                        if attempt == 0:
                            print(f"\n  HTTP {resp.status} (id={sound_id}), "
                                  f"retry {attempt+1}/{max_retries}, "
                                  f"waiting {wait:.0f}s...")
                        await asyncio.sleep(wait)
                    else:
                        # Non-retryable 4xx: permanent failure
                        return {
                            "sound_id": sound_id,
                            "status": f"http_{resp.status}",
                            "size": 0,
                        }

            except Exception as e:
                last_status = str(e)
                if attempt < max_retries - 1:
                    wait = rate_limit * (2 ** attempt) + 1
                    print(f"\n  Error (id={sound_id}): {e}, "
                          f"retry {attempt+1}/{max_retries}, "
                          f"waiting {wait:.0f}s...")
                    await asyncio.sleep(wait)

        out_path.unlink(missing_ok=True)
        return {
            "sound_id": sound_id,
            "status": f"http_{last_status}" if last_status else "error",
                "size": 0,
            }


def _check_exists(rec: dict, cfg: dict) -> dict | None:
    """ファイルが既にDL済みか同期的にチェックする。"""
    sound_id = rec["inat_api_sound_id"]
    file_url = rec.get("inat_api_file_url", "")
    species_code = rec["ebird_species_code"]

    if not file_url:
        return {"sound_id": sound_id, "status": "no_url", "size": 0}

    audio_dir = nas_path(cfg, cfg["inat_api"]["audio_dir"])
    ext = Path(file_url.split("?")[0]).suffix or ".m4a"
    out_path = audio_dir / species_code / f"inatapi_{sound_id}{ext}"

    if out_path.exists():
        return {
            "sound_id": sound_id,
            "status": "exists",
            "path": str(out_path),
            "size": 0,
        }
    return None


async def download_species_batch(
    session: aiohttp.ClientSession,
    species_records: list[dict],
    cfg: dict,
    semaphore: asyncio.Semaphore,
    rate_limit: float,
    progress_callback=None,
) -> list[dict]:
    """1種分の録音をバッチダウンロードする。"""
    # 既存ファイルを同期的に高速チェック（semaphore/ネットワーク不要）
    results = []
    need_download = []
    for rec in species_records:
        existing = _check_exists(rec, cfg)
        if existing:
            results.append(existing)
        else:
            need_download.append(rec)

    total = len(species_records)
    if progress_callback and results:
        progress_callback(len(results), total, results[-1])

    # 未DL分のみネットワークDL
    if need_download:
        tasks = [
            download_one(session, rec, cfg, semaphore, rate_limit)
            for rec in need_download
        ]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
                if progress_callback:
                    progress_callback(len(results), total, result)

    return results


async def download_all(cfg: dict, args):
    """メタデータに基づき全録音をダウンロードする。種ごとにバッチ処理。"""
    api_cfg = cfg["inat_api"]
    metadata_path = (
        nas_path(cfg, api_cfg["metadata_dir"]) / "inat_api_metadata.parquet"
    )

    if not metadata_path.exists():
        print("Error: metadata not found. Run --metadata-only first.")
        sys.exit(1)

    df = load_metadata(metadata_path)
    filtered = filter_for_download(df, cfg)

    # 種ごとにグループ化
    species_groups = filtered.groupby("ebird_species_code")
    species_list = sorted(species_groups.groups.keys())
    if args.limit > 0:
        species_list = species_list[:args.limit]

    total_recordings = sum(
        len(species_groups.get_group(sp)) for sp in species_list
    )

    # ── ドライラン ──
    if args.dry_run:
        print(f"\n[DRY RUN] Would download {total_recordings} recordings "
              f"for {len(species_list)} species")
        est_gb = total_recordings * 0.5 / 1024  # m4a 平均 ~0.5MB
        print(f"  Estimated storage: ~{est_gb:.0f} GB")
        print(f"  At 5GB/hour limit: ~{est_gb / 5:.0f} hours")
        print(f"\nTop 10 species by recording count:")
        top10 = filtered.groupby("ebird_species_code").size().nlargest(10)
        for sp, cnt in top10.items():
            print(f"  {sp}: {cnt}")
        return

    dl_cfg = api_cfg["download"]
    max_concurrent = dl_cfg["max_concurrent"]
    rate_limit = dl_cfg["rate_limit_sec"]
    hourly_limit_gb = dl_cfg["hourly_limit_gb"]
    semaphore = asyncio.Semaphore(max_concurrent)

    bandwidth = BandwidthTracker(hourly_limit_gb)

    # 進捗ファイル
    progress_path = (
        nas_path(cfg, api_cfg["metadata_dir"]) / "download_progress.json"
    )
    progress: dict = {"files": {}, "completed_species": []}
    if progress_path.exists():
        with open(progress_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "files" in raw:
            progress = raw
        else:
            progress["files"] = raw

    all_ok_results: dict = progress["files"]
    completed_species: set = set(progress["completed_species"])

    remaining_species = [
        sp for sp in species_list if sp not in completed_species
    ]
    skipped_species = len(species_list) - len(remaining_species)
    remaining_recordings = sum(
        len(species_groups.get_group(sp)) for sp in remaining_species
    )

    print(f"\nDownloading {total_recordings} recordings "
          f"for {len(species_list)} species")
    print(f"  Concurrent: {max_concurrent}, Rate limit: {rate_limit}s")
    print(f"  Bandwidth limit: {hourly_limit_gb} GB/hour")
    if skipped_species > 0:
        print(f"  Resuming: {skipped_species} species already completed")
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
        print("\n\nShutdown requested — finishing current species, "
              "saving progress...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    total_stats = {
        "ok": 0, "exists": 0, "error": 0,
        "http_error": 0, "http_error_codes": {},
    }
    total_downloaded_bytes = 0
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(remaining_species, desc="Species", unit="sp")
        for sp_code in pbar:
            if shutdown_requested:
                break

            # 帯域制限チェック
            if bandwidth.should_pause():
                wait_sec = bandwidth.seconds_until_reset()
                print(f"\n  Bandwidth limit reached "
                      f"({bandwidth.used_gb():.2f} GB). "
                      f"Pausing {wait_sec:.0f}s...")
                await asyncio.sleep(wait_sec + 5)

            sp_df = species_groups.get_group(sp_code)
            records = sp_df.to_dict("records")
            n_recs = len(records)

            sp_downloaded_bytes = 0

            def _progress(done, total, result):
                nonlocal sp_downloaded_bytes
                sp_downloaded_bytes += result.get("size", 0)
                dl_gb = (total_downloaded_bytes + sp_downloaded_bytes) / (1024**3)
                pbar.set_postfix_str(
                    f"{sp_code} {done}/{total} "
                    f"(DL {dl_gb:.2f}GB)"
                )

            dl_gb = total_downloaded_bytes / (1024**3)
            pbar.set_postfix_str(
                f"{sp_code} 0/{n_recs} "
                f"(DL {dl_gb:.2f}GB)"
            )

            try:
                results = await download_species_batch(
                    session, records, cfg, semaphore, rate_limit,
                    progress_callback=_progress,
                )
            except Exception as e:
                print(f"\n  Error processing {sp_code}: {e}")
                _save_progress(progress_path, {
                    "files": all_ok_results,
                    "completed_species": sorted(completed_species),
                })
                continue

            sp_ok = 0
            sp_skipped = 0
            for r in results:
                status = r["status"]
                dl_size = r.get("size", 0)

                if status == "ok":
                    total_stats["ok"] += 1
                    sp_ok += 1
                    total_downloaded_bytes += dl_size
                    bandwidth.add(dl_size)
                elif status == "exists":
                    total_stats["exists"] += 1
                    sp_ok += 1
                elif status == "no_url":
                    sp_skipped += 1
                elif status.startswith("http_"):
                    total_stats["http_error"] += 1
                    code = status.split("_", 1)[1]
                    total_stats["http_error_codes"][code] = (
                        total_stats["http_error_codes"].get(code, 0) + 1
                    )
                    # Only truly permanent errors count as skipped
                    # (404 Not Found, 410 Gone, etc.)
                    if code.isdigit():
                        c = int(code)
                        if c in (404, 410, 451):
                            sp_skipped += 1
                else:
                    total_stats["error"] += 1

                if status in ("ok", "exists") and "path" in r:
                    all_ok_results[str(r["sound_id"])] = r["path"]

            if sp_ok + sp_skipped == len(records):
                completed_species.add(sp_code)

            _save_progress(progress_path, {
                "files": all_ok_results,
                "completed_species": sorted(completed_species),
            })

    # 結果サマリ
    print(f"\n{'=' * 60}")
    print("iNat API Download Results")
    print(f"{'=' * 60}")
    for status, cnt in sorted(total_stats.items()):
        if status == "http_error_codes":
            if cnt:
                print(f"  http_error breakdown:")
                for code, n in sorted(cnt.items(),
                                      key=lambda x: -x[1]):
                    print(f"    HTTP {code}: {n}")
        else:
            print(f"  {status}: {cnt}")
    print(f"  Total downloaded: "
          f"{total_downloaded_bytes / 1024**3:.2f} GB")
    print(f"  Total tracked files: {len(all_ok_results)}")
    print(f"  Completed species: "
          f"{len(completed_species)}/{len(species_list)}")

    if shutdown_requested:
        print("\n  Download interrupted — run again to continue.")
        sys.exit(1)

    # ファイルパスをメタデータに反映
    if all_ok_results:
        df["file_path"] = df["inat_api_sound_id"].apply(
            lambda x: (
                all_ok_results.get(str(int(x)), "")
                if pd.notna(x)
                else ""
            )
        )
        save_metadata(df, metadata_path)


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
        print("  --metadata-only: collect metadata from iNat API")
        print("  --download:      download audio files (after metadata)")
        sys.exit(1)


if __name__ == "__main__":
    main()
