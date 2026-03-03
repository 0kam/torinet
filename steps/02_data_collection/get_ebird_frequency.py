#!/usr/bin/env python
"""eBird 頻度データ取得スクリプト。

日本国内のeBirdチェックリスト報告頻度（detection frequency）を取得し、
各種の「よく見られる種 vs 迷鳥」判定に使う頻度スコアを算出する。

── 2つのデータ取得方法 ──

方法1: バーチャートTSVデータ（推奨・最も正確）
  eBirdウェブサイトからダウンロードしたヒストグラムTSVを解析。
  週別（年48期間）の検出頻度を持ち、季節性も判定できる。

方法2: eBird API v2（自動・近似値）
  APIから日本の種リストと最近の観察データを取得し、頻度を推定。
  30日間の観察回数ベースなので季節バイアスがある。

── 使い方 ──

  # 方法1: TSVファイルから（推奨）
  python get_ebird_frequency.py --from-tsv ebird_JP_barchart.tsv

  # 方法2: APIから自動取得
  python get_ebird_frequency.py --from-api

  # TSVダウンロードURLを表示するだけ
  python get_ebird_frequency.py --show-url
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_target_species, load_config, STEP_DIR

# ── 定数 ──

EBIRD_API_BASE = "https://api.ebird.org/v2"
BARCHART_URL = "https://ebird.org/barchartData"
REGION_CODE = "JP"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = STEP_DIR / "ebird_frequency.csv"

# 48期間のラベル（各月4期間）
MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
PERIOD_LABELS = [f"{m}-{w}" for m in MONTH_NAMES for w in range(1, 5)]

# 季節判定用の月グループ（1-indexed）
SEASON_MONTHS = {
    "breeding": {4, 5, 6, 7, 8},       # 4-8月
    "winter": {11, 12, 1, 2, 3},        # 11-3月
    "spring_passage": {3, 4, 5},        # 3-5月
    "autumn_passage": {8, 9, 10, 11},   # 8-11月
}


# ── eBird APIキー管理 ──

def load_ebird_api_key() -> str | None:
    """eBird APIキーを読み込む。keys/ebird_api.key から。"""
    key_path = PROJECT_ROOT / "keys" / "ebird_api.key"
    if key_path.exists():
        return key_path.read_text().strip()
    return None


def ebird_api_headers(api_key: str) -> dict:
    """eBird API v2 リクエストヘッダ。"""
    return {"x-ebirdapitoken": api_key}


# ── 方法1: バーチャートTSV解析 ──

def get_barchart_download_url(
    region: str = REGION_CODE,
    byr: int = 1900,
    eyr: int = 2026,
) -> str:
    """eBirdバーチャートTSVのダウンロードURL生成。"""
    params = {
        "r": region,
        "bmo": 1,
        "emo": 12,
        "byr": byr,
        "eyr": eyr,
        "fmt": "tsv",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{BARCHART_URL}?{qs}"


def parse_barchart_tsv(tsv_path: str | Path) -> pd.DataFrame:
    """eBirdバーチャートTSVファイルを解析する。

    TSV形式:
      - ヘッダ行（列名は使わず位置ベースで解析）
      - 「Sample Size」行にチェックリスト数
      - 以降: 種名 + 48期間の検出頻度（0.0-1.0）

    Returns:
        種ごとの検出頻度DataFrame
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSVファイルが見つかりません: {tsv_path}")

    raw = tsv_path.read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    if len(lines) < 3:
        raise ValueError("TSVファイルの形式が不正です（行数不足）")

    # ヘッダ行を解析してデータ列の数を確認
    header = lines[0].split("\t")

    # Sample Size行を探す（最初の数行以内にあるはず）
    sample_sizes = None
    data_start = 1
    for i, line in enumerate(lines[1:6], start=1):
        cols = line.split("\t")
        if cols and "sample size" in cols[0].lower():
            # 数値列を抽出
            sample_sizes = []
            for v in cols[1:49]:
                try:
                    sample_sizes.append(int(float(v)))
                except (ValueError, IndexError):
                    sample_sizes.append(0)
            data_start = i + 1
            break

    # 種データの解析
    records = []
    for line in lines[data_start:]:
        cols = line.split("\t")
        if not cols or not cols[0].strip():
            continue

        species_name = cols[0].strip()

        # eBirdのTSVでは種名の後に48期間の頻度値が並ぶ
        freqs = []
        for v in cols[1:49]:
            v = v.strip()
            if v == "" or v == "—" or v == "-":
                freqs.append(0.0)
            else:
                try:
                    freqs.append(float(v))
                except ValueError:
                    freqs.append(0.0)

        # 48期間に満たない場合はゼロ埋め
        while len(freqs) < 48:
            freqs.append(0.0)

        records.append({"species_name": species_name, "freqs": freqs})

    if not records:
        raise ValueError("TSVから種データを抽出できませんでした")

    # DataFrameに変換
    rows = []
    for rec in records:
        row = {"species_name": rec["species_name"]}
        for j, label in enumerate(PERIOD_LABELS):
            row[label] = rec["freqs"][j]
        rows.append(row)

    df = pd.DataFrame(rows)

    # 年間平均頻度と統計を算出
    freq_cols = PERIOD_LABELS
    df["frequency_annual_mean"] = df[freq_cols].mean(axis=1)
    df["frequency_max"] = df[freq_cols].max(axis=1)
    df["frequency_min"] = df[freq_cols].min(axis=1)
    df["n_periods_detected"] = (df[freq_cols] > 0).sum(axis=1)

    print(f"バーチャートTSV: {len(df)} 種を解析")
    if sample_sizes:
        total_checklists = sum(sample_sizes)
        print(f"チェックリスト総数: {total_checklists:,}")

    return df


def classify_residence_status(row: pd.Series) -> str:
    """月別頻度パターンから滞在ステータスを推定する。

    Returns:
        resident / summer / winter / passage / vagrant / undetected
    """
    freq_cols = PERIOD_LABELS
    freqs = row[freq_cols].values

    if all(f == 0 for f in freqs):
        return "undetected"

    annual_mean = row["frequency_annual_mean"]

    # 非常に低頻度 → 迷鳥
    if annual_mean < 0.001 and row["n_periods_detected"] <= 4:
        return "vagrant"

    # 月別の平均頻度を計算（4期間ずつ）
    monthly_means = []
    for m in range(12):
        start = m * 4
        month_freqs = freqs[start : start + 4]
        monthly_means.append(sum(month_freqs) / 4)

    # 繁殖期・越冬期の平均頻度
    breeding_freq = sum(monthly_means[m - 1] for m in SEASON_MONTHS["breeding"]) / len(
        SEASON_MONTHS["breeding"]
    )
    winter_freq = sum(monthly_means[m - 1] for m in SEASON_MONTHS["winter"]) / len(
        SEASON_MONTHS["winter"]
    )

    max_monthly = max(monthly_means)
    if max_monthly == 0:
        return "undetected"

    # 通年で検出 → 留鳥
    n_months_detected = sum(1 for m in monthly_means if m > max_monthly * 0.1)
    if n_months_detected >= 10:
        return "resident"

    # 繁殖期に偏る → 夏鳥
    if breeding_freq > winter_freq * 3 and breeding_freq > 0.001:
        return "summer"

    # 越冬期に偏る → 冬鳥
    if winter_freq > breeding_freq * 3 and winter_freq > 0.001:
        return "winter"

    # 上記に当てはまらず限られた月のみ → 旅鳥
    if n_months_detected <= 6:
        return "passage"

    return "resident"


# ── 方法2: eBird API v2 ──

def api_get_species_list(api_key: str, region: str = REGION_CODE) -> list[str]:
    """eBird API: 地域の種リスト取得。"""
    url = f"{EBIRD_API_BASE}/product/spplist/{region}"
    resp = requests.get(url, headers=ebird_api_headers(api_key), timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_get_recent_observations(
    api_key: str,
    region: str = REGION_CODE,
    back: int = 30,
) -> list[dict]:
    """eBird API: 最近の観察データ取得。"""
    url = f"{EBIRD_API_BASE}/data/obs/{region}/recent"
    params = {"back": back}
    resp = requests.get(
        url, headers=ebird_api_headers(api_key), params=params, timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def api_get_historic_observations(
    api_key: str,
    region: str,
    year: int,
    month: int,
    day: int,
) -> list[dict]:
    """eBird API: 特定日の観察データ取得。"""
    url = f"{EBIRD_API_BASE}/data/obs/{region}/historic/{year}/{month}/{day}"
    resp = requests.get(url, headers=ebird_api_headers(api_key), timeout=60)
    resp.raise_for_status()
    return resp.json()


def frequency_from_api(api_key: str) -> pd.DataFrame:
    """eBird APIから頻度データを推定する。

    1. 日本の種リスト取得（/product/spplist/JP）
    2. 最近30日間の観察取得
    3. 種ごとの観察回数をカウント
    """
    print("eBird API: 日本の種リスト取得中...")
    jp_species = api_get_species_list(api_key)
    print(f"  → {len(jp_species)} 種")
    time.sleep(1)

    print("eBird API: 最近30日間の観察データ取得中...")
    observations = api_get_recent_observations(api_key, back=30)
    print(f"  → {len(observations)} 件の観察")

    # 種ごとの観察回数と最大個体数を集計
    species_counts: dict[str, dict] = {}
    for obs in observations:
        code = obs.get("speciesCode", "")
        if not code:
            continue
        if code not in species_counts:
            species_counts[code] = {
                "obs_count": 0,
                "total_individuals": 0,
                "comName": obs.get("comName", ""),
                "sciName": obs.get("sciName", ""),
            }
        species_counts[code]["obs_count"] += 1
        how_many = obs.get("howMany")
        if how_many is not None:
            species_counts[code]["total_individuals"] += how_many

    # DataFrameに変換
    max_obs = max((v["obs_count"] for v in species_counts.values()), default=1)

    rows = []
    for code in jp_species:
        info = species_counts.get(code, {})
        obs_count = info.get("obs_count", 0)
        rows.append({
            "ebird_species_code": code,
            "scientific_name_ebird": info.get("sciName", ""),
            "common_name_ebird": info.get("comName", ""),
            "observation_count_30d": obs_count,
            "frequency_score": obs_count / max_obs if max_obs > 0 else 0,
            "in_recent_30d": obs_count > 0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("observation_count_30d", ascending=False).reset_index(drop=True)

    return df


# ── 結果統合 ──

def merge_with_target_species(
    freq_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """頻度データをターゲット種リストとマージする。"""
    cfg = load_config()
    target = get_target_species(cfg)

    if method == "tsv":
        # TSVの種名から学名と和名を抽出してマッチに使う
        freq_df = freq_df.copy()
        freq_df["sci_extracted"] = freq_df["species_name"].apply(
            lambda x: _extract_scientific_name(x) if isinstance(x, str) else ""
        )
        freq_df["jpn_extracted"] = freq_df["species_name"].apply(
            lambda x: _extract_japanese_name(x) if isinstance(x, str) else ""
        )

        # 学名でマッチ（scientific_name → sci_extracted）
        merged = target.merge(
            freq_df,
            left_on="scientific_name",
            right_on="sci_extracted",
            how="left",
        )

        # 未マッチ種を ebird_sciname でリトライ
        unmatched_mask = merged["species_name"].isna()
        if unmatched_mask.sum() > 0:
            unmatched_codes = merged.loc[unmatched_mask, "ebird_species_code"].values
            retry = target[target["ebird_species_code"].isin(unmatched_codes)].merge(
                freq_df,
                left_on="ebird_sciname",
                right_on="sci_extracted",
                how="left",
                suffixes=("", "_retry"),
            )
            for _, row in retry[retry["species_name"].notna()].iterrows():
                idx = merged.index[
                    merged["ebird_species_code"] == row["ebird_species_code"]
                ]
                if len(idx) > 0:
                    for col in freq_df.columns:
                        if col in row.index:
                            merged.loc[idx[0], col] = row[col]

        # さらに未マッチ種を和名でリトライ
        unmatched_mask = merged["species_name"].isna()
        if unmatched_mask.sum() > 0:
            unmatched_codes = merged.loc[unmatched_mask, "ebird_species_code"].values
            retry2 = target[target["ebird_species_code"].isin(unmatched_codes)].merge(
                freq_df,
                left_on="japanese_name",
                right_on="jpn_extracted",
                how="left",
                suffixes=("", "_retry2"),
            )
            for _, row in retry2[retry2["species_name"].notna()].iterrows():
                idx = merged.index[
                    merged["ebird_species_code"] == row["ebird_species_code"]
                ]
                if len(idx) > 0:
                    for col in freq_df.columns:
                        if col in row.index:
                            merged.loc[idx[0], col] = row[col]

        n_matched = merged["species_name"].notna().sum()
        print(f"種マッチング: {n_matched}/{len(target)} "
              f"({n_matched/len(target)*100:.1f}%)")

        # 滞在ステータス判定
        freq_cols = [c for c in PERIOD_LABELS if c in merged.columns]
        if freq_cols:
            merged["residence_status"] = merged.apply(
                lambda r: classify_residence_status(r) if pd.notna(r.get("species_name")) else "undetected",
                axis=1,
            )

        # 出力列の整理
        output_cols = [
            "ebird_species_code", "scientific_name", "japanese_name",
            "ebird_common_name", "frequency_annual_mean", "frequency_max",
            "n_periods_detected", "residence_status",
        ]
        output_cols = [c for c in output_cols if c in merged.columns]
        result = merged[output_cols].copy()
        result = result.sort_values("frequency_annual_mean", ascending=False, na_position="last")

    elif method == "api":
        merged = target.merge(
            freq_df,
            left_on="ebird_species_code",
            right_on="ebird_species_code",
            how="left",
        )
        output_cols = [
            "ebird_species_code", "scientific_name", "japanese_name",
            "ebird_common_name", "frequency_score",
            "observation_count_30d", "in_recent_30d",
        ]
        output_cols = [c for c in output_cols if c in merged.columns]
        result = merged[output_cols].copy()
        result = result.sort_values("frequency_score", ascending=False, na_position="last")

    result = result.reset_index(drop=True)
    return result


def _extract_scientific_name(text: str) -> str:
    """種名文字列から学名を抽出する。

    対応形式:
      - 'Common Name (Scientific name)' — 英語TSV
      - '和名 (<em class="sci">Genus species</em>)' — 日本語TSV (HTMLタグ付き)
    """
    # HTMLタグ内の学名を抽出
    m = re.search(r'<em[^>]*>([A-Z][a-z]+ [a-z]+[^<]*)</em>', text)
    if m:
        return m.group(1).strip()
    # 括弧内の学名
    m = re.search(r"\(([A-Z][a-z]+ [a-z]+)\)", text)
    if m:
        return m.group(1)
    return ""


def _extract_japanese_name(text: str) -> str:
    """種名文字列から和名を抽出する。'和名 (<em...>...</em>)' 形式。"""
    # HTMLタグの前の部分が和名
    m = re.match(r'^(.+?)\s*[\(<]', text)
    if m:
        name = m.group(1).strip()
        # 雑種/sp.表記を除外するフラグは呼び出し側で判定
        return name
    return text.strip()


# ── 出力 ──

def print_summary(df: pd.DataFrame, method: str) -> None:
    """結果のサマリを表示する。"""
    total = len(df)

    if method == "tsv":
        freq_col = "frequency_annual_mean"
        has_data = df[freq_col].notna() & (df[freq_col] > 0)
    else:
        freq_col = "frequency_score"
        has_data = df[freq_col].notna() & (df[freq_col] > 0)

    n_with_data = has_data.sum()
    n_no_data = total - n_with_data

    print("\n" + "=" * 70)
    print("eBird 頻度データ サマリ")
    print("=" * 70)
    print(f"対象種数:     {total}")
    print(f"データあり:   {n_with_data} ({n_with_data/total*100:.1f}%)")
    print(f"データなし:   {n_no_data} ({n_no_data/total*100:.1f}%)")

    if method == "tsv" and "residence_status" in df.columns:
        print("\n── 滞在ステータス ──")
        status_counts = df["residence_status"].value_counts()
        for status, count in status_counts.items():
            print(f"  {status:12s}: {count:4d}")

    # Top 20
    top = df[has_data].head(20)
    print(f"\n── 頻度 Top 20 ──")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row.get("japanese_name", "")
        sci = row.get("scientific_name", "")
        freq = row[freq_col]
        freq_str = f"{freq:.4f}" if pd.notna(freq) else "N/A"
        status = row.get("residence_status", "")
        status_str = f" [{status}]" if status else ""
        print(f"  {i:2d}. {name} ({sci}) freq={freq_str}{status_str}")

    # Bottom 20 (with data)
    bottom = df[has_data].tail(20)
    print(f"\n── 頻度 Bottom 20（データあり最低頻度）──")
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        name = row.get("japanese_name", "")
        sci = row.get("scientific_name", "")
        freq = row[freq_col]
        freq_str = f"{freq:.6f}" if pd.notna(freq) else "N/A"
        status = row.get("residence_status", "")
        status_str = f" [{status}]" if status else ""
        print(f"  {i:2d}. {name} ({sci}) freq={freq_str}{status_str}")

    # Species not found
    if n_no_data > 0:
        print(f"\n── eBirdデータなし ({n_no_data}種) ──")
        no_data = df[~has_data]
        for _, row in no_data.head(30).iterrows():
            name = row.get("japanese_name", "")
            sci = row.get("scientific_name", "")
            print(f"  - {name} ({sci})")
        if n_no_data > 30:
            print(f"  ... 他 {n_no_data - 30} 種")


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """結果をCSVに保存。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n保存: {output_path} ({len(df)} 行)")


# ── メイン ──

def main():
    parser = argparse.ArgumentParser(
        description="eBird頻度データ取得",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--from-tsv",
        type=str,
        metavar="TSV_FILE",
        help="eBirdバーチャートTSVファイルから解析",
    )
    group.add_argument(
        "--from-api",
        action="store_true",
        help="eBird API v2 から自動取得（近似値）",
    )
    group.add_argument(
        "--show-url",
        action="store_true",
        help="TSVダウンロードURLを表示するだけ",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_PATH),
        help=f"出力CSVパス（デフォルト: {OUTPUT_PATH}）",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=REGION_CODE,
        help="eBird地域コード（デフォルト: JP）",
    )

    args = parser.parse_args()

    # TSVダウンロードURL表示
    if args.show_url:
        url = get_barchart_download_url(region=args.region)
        print("eBirdバーチャートデータのダウンロード手順:")
        print()
        print("1. eBirdにログイン: https://ebird.org/home")
        print("2. 以下のURLをブラウザで開く:")
        print(f"   {url}")
        print("3. ダウンロードされたTSVファイルを保存")
        print("4. 本スクリプトで解析:")
        print(f"   python {Path(__file__).name} --from-tsv <保存したファイル>")
        return

    # 方法1: TSV解析
    if args.from_tsv:
        print(f"方法1: バーチャートTSV解析 ({args.from_tsv})")
        freq_df = parse_barchart_tsv(args.from_tsv)
        result = merge_with_target_species(freq_df, method="tsv")
        print_summary(result, method="tsv")
        save_results(result, Path(args.output))
        return

    # 方法2: API
    if args.from_api:
        api_key = load_ebird_api_key()
        if api_key is None:
            print("エラー: eBird APIキーが見つかりません。", file=sys.stderr)
            print(file=sys.stderr)
            print("以下の手順でAPIキーを取得してください:", file=sys.stderr)
            print("  1. https://ebird.org/api/keygen にアクセス", file=sys.stderr)
            print("  2. APIキーを取得", file=sys.stderr)
            print("  3. keys/ebird_api.key に保存:", file=sys.stderr)
            print(f"     echo 'YOUR_API_KEY' > {PROJECT_ROOT / 'keys' / 'ebird_api.key'}", file=sys.stderr)
            print(file=sys.stderr)
            print("または、TSV方式を使用:", file=sys.stderr)
            print(f"  python {Path(__file__).name} --show-url", file=sys.stderr)
            sys.exit(1)

        print(f"方法2: eBird API v2（地域: {args.region}）")
        freq_df = frequency_from_api(api_key)
        result = merge_with_target_species(freq_df, method="api")
        print_summary(result, method="api")
        save_results(result, Path(args.output))
        return


if __name__ == "__main__":
    main()
