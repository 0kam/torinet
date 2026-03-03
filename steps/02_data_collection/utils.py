"""
Step 02 共通ユーティリティ。

種リスト読み込み、日本国内判定、メタデータ保存、パス生成など。
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEP_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    """config.yaml を読み込む。"""
    cfg_path = STEP_DIR / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # NASベースパスを展開
    cfg["nas_base"] = str(Path(cfg["nas_base"]).expanduser())
    return cfg


def nas_path(cfg: dict, relative: str) -> Path:
    """NASベースパスからの相対パスを解決する。"""
    return Path(cfg["nas_base"]) / relative


def load_species_list(cfg: dict | None = None) -> pd.DataFrame:
    """species_list.csv を読み込む。"""
    if cfg is None:
        cfg = load_config()
    csv_path = PROJECT_ROOT / cfg["species_list"]
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    return df


def get_target_species(cfg: dict | None = None) -> pd.DataFrame:
    """eBirdマッチ済みの種のみ返す（ebird_species_code, scientific_name, ebird_sciname）。"""
    df = load_species_list(cfg)
    matched = df[df["ebird_matched"] == True].copy()  # noqa: E712
    cols = [
        "species_num", "scientific_name", "japanese_name",
        "ebird_sciname", "ebird_common_name", "ebird_species_code",
        "order", "family",
    ]
    return matched[cols].reset_index(drop=True)


def is_in_japan(lat: float | None, lon: float | None, cfg: dict | None = None) -> bool:
    """緯度経度が日本国内かどうか判定する。"""
    if lat is None or lon is None:
        return False
    if cfg is None:
        cfg = load_config()
    bounds = cfg["japan_bounds"]
    return (
        bounds["lat_min"] <= lat <= bounds["lat_max"]
        and bounds["lon_min"] <= lon <= bounds["lon_max"]
    )


def save_metadata(df: pd.DataFrame, path: str | Path) -> None:
    """DataFrame を Parquet 形式で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path), compression="snappy")
    print(f"Saved {len(df)} rows → {path}")


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Parquet ファイルを読み込む。"""
    return pq.read_table(str(path)).to_pandas()


def audio_file_path(
    cfg: dict,
    source: str,
    species_code: str,
    recording_id: str,
    ext: str = "wav",
) -> Path:
    """音声ファイルのNAS保存パスを生成する。

    例: audio/xeno-canto/wav/leucig1/xc_12345.wav
    """
    if source == "xeno-canto":
        base = cfg["xeno_canto"]["wav_dir"]
    elif source == "inat":
        base = cfg["inat_sounds"]["filtered_dir"]
    elif source == "inat-api":
        base = cfg["inat_api"]["audio_dir"]
    elif source == "macaulay":
        base = cfg["macaulay"]["audio_dir"]
    else:
        raise ValueError(f"Unknown source: {source}")

    return nas_path(cfg, base) / species_code / f"{recording_id}.{ext}"


def load_api_key(cfg: dict) -> str:
    """XC APIキーを読み込む。"""
    key_path = PROJECT_ROOT / cfg["xc_api_key_file"]
    return key_path.read_text().strip()


# ── 統一メタデータスキーマ ──

UNIFIED_SCHEMA = [
    "recording_id",       # "xc:12345", "inat:67890"
    "source",             # "xeno-canto" / "inat" / "fsd50k"
    "ebird_species_code",
    "scientific_name",
    "japanese_name",
    "latitude",
    "longitude",
    "country",
    "is_japan",
    "duration_sec",
    "sample_rate",
    "quality",            # XC品質 A-E（iNatは空）
    "license",
    "file_path",          # NAS相対パス
    "vocalization_type",  # song / call / alarm 等
]
