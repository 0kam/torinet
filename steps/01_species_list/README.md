# Step 01: 種リストの作成

## 目的

日本鳥類目録改訂第8版をマスターとし、eBird/Clements taxonomy・BirdNETラベルとの
マッピングテーブルを構築する。訓練データ収集やモデル評価の基盤となる。

## データソース

| ソース | ファイル (NAS: `taxonomy/`) | 備考 |
|--------|----------------------------|------|
| 日本鳥類目録改訂第8版 | `jpbirdlist8ed.xlsx` | [日本鳥学会公式](https://ornithology.jp/checklist.html) |
| eBird/Clements Taxonomy v2024 | `ebird_taxonomy_v2024.csv` | [Cornell Lab](https://www.birds.cornell.edu/clementschecklist/) |
| BirdNET V2.4 Labels | `birdnet_labels_en.txt`, `birdnet_labels_ja.txt` | [GitHub](https://github.com/birdnet-team/BirdNET-Analyzer) |

## 成果物

- **`species_list.csv`** — 統合種リスト (690種)

## 結果サマリ

- 日本鳥類目録第8版: 690種 (自然分布644 + 外来46)
- eBird マッピング: 688/690 (99.7%)
- BirdNET マッピング: 605/690 (87.7%)
- eBird 未対応: ミヤコショウビン、オガサワラカワラヒワ（絶滅種）
- BirdNET 未対応: 85種（希少種・固有種・絶滅種中心）

### 分類体系間の主な差異

日本鳥学会とeBird/Clements間で27件の属レベルの差異をシノニムテーブルで解決。
詳細は `build_species_list.py` の `CHECKLIST8_TO_EBIRD_SYNONYMS` を参照。

代表例:
- チドリ属の再編: Charadrius → Thinornis / Anarhynchus
- ヨシゴイ属の統合: Ixobrychus → Botaurus
- タカ属の再編: Accipiter → Astur / Tachyspiza
- センニュウ属の分離: Locustella → Helopsaltes

## 実行方法

```bash
.venv/bin/python steps/01_species_list/build_species_list.py
```
