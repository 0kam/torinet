"""
日本鳥類目録第8版をマスターリストとし、eBird/Clements taxonomyおよび
BirdNET V2.4ラベルとのマッピングテーブルを構築する。

出力:
  steps/01_species_list/species_list.csv — 統合種リスト
"""

import csv
import re
from pathlib import Path

import openpyxl
import pandas as pd

DATA_DIR = Path(__file__).parent
OUT_CSV = DATA_DIR / "species_list.csv"

# ── 0. 分類体系間のシノニムテーブル ────────────────────────────
# 日本鳥類目録第8版の学名 → eBird/Clements v2024 の学名
# 主に属レベルの分割(split)・統合(lump)による差異
CHECKLIST8_TO_EBIRD_SYNONYMS = {
    # スペル差異
    "Anser cygnoid": "Anser cygnoides",
    # Charadrius → Thinornis / Anarhynchus (チドリ属の再編)
    "Charadrius placidus": "Thinornis placidus",
    "Charadrius dubius": "Thinornis dubius",
    "Charadrius alexandrinus": "Anarhynchus alexandrinus",
    "Charadrius leschenaultii": "Anarhynchus leschenaultii",
    "Charadrius mongolus": "Anarhynchus mongolus",
    "Charadrius veredus": "Anarhynchus veredus",
    # Ixobrychus → Botaurus (ヨシゴイ類)
    "Ixobrychus sinensis": "Botaurus sinensis",
    "Ixobrychus eurhythmus": "Botaurus eurhythmus",
    "Ixobrychus cinnamomeus": "Botaurus cinnamomeus",
    "Ixobrychus flavicollis": "Botaurus flavicollis",
    # Bubulcus → Ardea (アマサギ → 東アマサギ)
    "Bubulcus ibis": "Ardea coromanda",
    # Accipiter → Astur / Tachyspiza (タカ属の再編)
    "Accipiter gentilis": "Astur gentilis",
    "Accipiter soloensis": "Tachyspiza soloensis",
    "Accipiter gularis": "Tachyspiza gularis",
    # Corvus → Coloeus (コクマルガラス類)
    "Corvus monedula": "Coloeus monedula",
    "Corvus dauuricus": "Coloeus dauuricus",
    # Pardaliparus → Periparus
    "Pardaliparus venustulus": "Periparus venustulus",
    # Locustella → Helopsaltes (センニュウ類)
    "Locustella amnicola": "Helopsaltes amnicola",
    "Locustella pryeri": "Helopsaltes pryeri",
    "Locustella certhiola": "Helopsaltes certhiola",
    "Locustella ochotensis": "Helopsaltes ochotensis",
    "Locustella pleskei": "Helopsaltes pleskei",
    # Hypotaenidia → Gallirallus (ヤンバルクイナ)
    "Hypotaenidia okinawae": "Gallirallus okinawae",
    # 絶滅種 — eBirdでの学名
    "Cichlopasser terrestris": "Zoothera terrestris",  # オガサワラガビチョウ (Bonin Thrush)
    # 注: Chloris kittlitzi (オガサワラカワラヒワ) と Todiramphus miyakoensis (ミヤコショウビン) は
    # eBirdにも存在しない絶滅種
}

# eBird名がBirdNETで異なる場合のシノニム（種小名マッチで対応できないもの）
EBIRD_TO_BIRDNET_SYNONYMS = {
    # 必要に応じて追加
}

# ── 1. 日本鳥類目録第8版のパース ──────────────────────────────

def parse_checklist_8ed(xlsx_path: str) -> pd.DataFrame:
    """公式Excelファイルから種レベルの情報を抽出する。"""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)

    # Sheet2 がデータ本体
    ws = wb.worksheets[1]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    # ヘッダー行を探す
    header = None
    data_rows = []
    for row in rows:
        vals = [str(v).strip() if v is not None else "" for v in row]
        if any("掲載順" in v for v in vals):
            header = vals
            continue
        if header is not None:
            data_rows.append(vals)

    if header is None:
        raise ValueError("Header row not found in sheet2")

    df = pd.DataFrame(data_rows, columns=range(len(header)))

    # 列番号でアクセス（ヘッダーに読み仮名が混在するため）
    # 0: 掲載順, 1: Part, 2: カテゴリ, 3: 種番号, 4: 亜種番号,
    # 5: 学名, 6: 著者, 7: 和名
    COL_ORDER = 0
    COL_PART = 1
    COL_CATEGORY = 2
    COL_SPECIES_NUM = 3
    COL_SUBSPECIES_NUM = 4
    COL_SCINAME = 5
    COL_AUTHOR = 6
    COL_JPNAME = 7

    # 種レベルのみ抽出（「種」を含み「亜種」を含まない）
    species_rows = []
    current_order = ""
    current_family = ""

    for _, row in df.iterrows():
        category = str(row[COL_CATEGORY]).strip()
        if "目" in category and "亜" not in category:
            current_order = str(row[COL_SCINAME]).strip() if row[COL_SCINAME] else ""
        elif "科" in category and "亜" not in category:
            current_family = str(row[COL_SCINAME]).strip() if row[COL_SCINAME] else ""
        elif "種" in category and "亜種" not in category:
            sci_name = str(row[COL_SCINAME]).strip() if row[COL_SCINAME] else ""
            jp_name = str(row[COL_JPNAME]).strip() if row[COL_JPNAME] else ""
            part = str(row[COL_PART]).strip() if row[COL_PART] else ""
            sp_num = str(row[COL_SPECIES_NUM]).strip() if row[COL_SPECIES_NUM] else ""

            if sci_name and jp_name:
                species_rows.append({
                    "checklist8_order": int(row[COL_ORDER]) if row[COL_ORDER] else 0,
                    "part": part,
                    "species_num": int(sp_num) if sp_num.isdigit() else 0,
                    "scientific_name": sci_name,
                    "japanese_name": jp_name,
                    "order": current_order,
                    "family": current_family,
                })

    return pd.DataFrame(species_rows)


# ── 2. eBird taxonomy のロード ─────────────────────────────────

def load_ebird_taxonomy(csv_path: str) -> pd.DataFrame:
    """eBird taxonomy CSVから種レベルのレコードを抽出する。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # species カテゴリのみ
    species = df[df["CATEGORY"] == "species"].copy()
    species = species.rename(columns={
        "SCI_NAME": "ebird_sciname",
        "PRIMARY_COM_NAME": "ebird_common_name",
        "SPECIES_CODE": "ebird_species_code",
        "ORDER": "ebird_order",
        "FAMILY": "ebird_family",
    })
    return species[["ebird_sciname", "ebird_common_name", "ebird_species_code",
                     "ebird_order", "ebird_family"]]


# ── 3. BirdNET ラベルのロード ──────────────────────────────────

def load_birdnet_labels(en_path: str, ja_path: str) -> pd.DataFrame:
    """BirdNET V2.4のラベルファイルをパースする。"""
    records = []
    with open(en_path) as f_en, open(ja_path) as f_ja:
        for line_en, line_ja in zip(f_en, f_ja):
            line_en = line_en.strip()
            line_ja = line_ja.strip()
            if "_" in line_en:
                sci_en, common_en = line_en.split("_", 1)
                sci_ja, common_ja = line_ja.split("_", 1)
                records.append({
                    "birdnet_sciname": sci_en,
                    "birdnet_common_name_en": common_en,
                    "birdnet_common_name_ja": common_ja,
                })
    return pd.DataFrame(records)


# ── 4. 学名の正規化とマッチング ───────────────────────────────

def normalize_sciname(name: str) -> str:
    """学名を正規化（属名 + 種小名のみ、小文字）。"""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0].lower()} {parts[1].lower()}"
    return name.lower().strip()


def apply_synonyms(checklist: pd.DataFrame) -> pd.DataFrame:
    """第8版の学名にeBird用のシノニムキーを追加する。"""
    checklist = checklist.copy()

    def get_ebird_key(sci_name: str) -> str:
        syn = CHECKLIST8_TO_EBIRD_SYNONYMS.get(sci_name)
        if syn:
            return normalize_sciname(syn)
        return normalize_sciname(sci_name)

    checklist["_ebird_key"] = checklist["scientific_name"].apply(get_ebird_key)
    # BirdNETキーは、まずeBirdシノニム経由、それでもダメなら元の学名
    checklist["_birdnet_key"] = checklist["scientific_name"].apply(normalize_sciname)
    return checklist


def build_mapping(checklist: pd.DataFrame, ebird: pd.DataFrame,
                  birdnet: pd.DataFrame) -> pd.DataFrame:
    """第8版マスターリストにeBirdとBirdNETの情報をマッピングする。"""
    # シノニム適用
    checklist = apply_synonyms(checklist)

    ebird = ebird.copy()
    ebird["_key"] = ebird["ebird_sciname"].apply(normalize_sciname)
    ebird = ebird.drop_duplicates(subset="_key", keep="first")

    birdnet = birdnet.copy()
    birdnet["_key"] = birdnet["birdnet_sciname"].apply(normalize_sciname)
    birdnet = birdnet.drop_duplicates(subset="_key", keep="first")

    # eBird: シノニム経由でJOIN
    merged = checklist.merge(
        ebird, left_on="_ebird_key", right_on="_key", how="left", suffixes=("", "_eb"),
    )

    # BirdNET: まず元の学名で、次にeBirdシノニム経由で
    merged = merged.merge(
        birdnet, left_on="_birdnet_key", right_on="_key", how="left", suffixes=("", "_bn"),
    )
    # BirdNETにマッチしなかった行はeBirdシノニム経由で再試行
    unmatched_mask = merged["birdnet_sciname"].isna()
    if unmatched_mask.any():
        retry = merged.loc[unmatched_mask].drop(
            columns=[c for c in merged.columns if c.startswith("birdnet_")],
        )
        retry = retry.merge(
            birdnet, left_on="_ebird_key", right_on="_key", how="left", suffixes=("", "_bn2"),
        )
        for col in ["birdnet_sciname", "birdnet_common_name_en", "birdnet_common_name_ja"]:
            if col in retry.columns:
                merged.loc[unmatched_mask, col] = retry[col].values

    # マッチ状況の列を追加
    merged["ebird_matched"] = merged["ebird_sciname"].notna()
    merged["birdnet_matched"] = merged["birdnet_sciname"].notna()

    # 整理
    cols = [
        "checklist8_order", "part", "species_num",
        "scientific_name", "japanese_name", "order", "family",
        "ebird_matched", "ebird_sciname", "ebird_common_name", "ebird_species_code",
        "birdnet_matched", "birdnet_sciname", "birdnet_common_name_en", "birdnet_common_name_ja",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]
    merged = merged.sort_values("checklist8_order").reset_index(drop=True)

    return merged


# ── Main ──────────────────────────────────────────────────────

def main():
    nas_dir = Path.home() / "NAS/nasbi/ToriNET/taxonomy"
    xlsx_path = str(nas_dir / "jpbirdlist8ed.xlsx")
    ebird_csv = str(nas_dir / "ebird_taxonomy_v2024.csv")
    birdnet_en = str(nas_dir / "birdnet_labels_en.txt")
    birdnet_ja = str(nas_dir / "birdnet_labels_ja.txt")

    print("Parsing checklist 8th edition...")
    checklist = parse_checklist_8ed(xlsx_path)
    print(f"  → {len(checklist)} species ({checklist[checklist['part']=='A'].shape[0]} native, "
          f"{checklist[checklist['part']=='B'].shape[0]} introduced)")

    print("Loading eBird taxonomy...")
    ebird = load_ebird_taxonomy(ebird_csv)
    print(f"  → {len(ebird)} species worldwide")

    print("Loading BirdNET labels...")
    birdnet = load_birdnet_labels(birdnet_en, birdnet_ja)
    print(f"  → {len(birdnet)} classes")

    print("Building mapping...")
    result = build_mapping(checklist, ebird, birdnet)

    ebird_hit = result["ebird_matched"].sum()
    birdnet_hit = result["birdnet_matched"].sum()
    total = len(result)
    print(f"  → eBird matched:   {ebird_hit}/{total} ({ebird_hit/total*100:.1f}%)")
    print(f"  → BirdNET matched: {birdnet_hit}/{total} ({birdnet_hit/total*100:.1f}%)")

    # マッチしなかった種を表示
    ebird_miss = result[~result["ebird_matched"]]
    if len(ebird_miss) > 0:
        print(f"\n  eBird unmatched ({len(ebird_miss)}):")
        for _, row in ebird_miss.iterrows():
            print(f"    {row['scientific_name']} ({row['japanese_name']})")

    birdnet_miss = result[~result["birdnet_matched"]]
    if len(birdnet_miss) > 0 and len(birdnet_miss) <= 50:
        print(f"\n  BirdNET unmatched ({len(birdnet_miss)}):")
        for _, row in birdnet_miss.iterrows():
            print(f"    {row['scientific_name']} ({row['japanese_name']})")

    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
