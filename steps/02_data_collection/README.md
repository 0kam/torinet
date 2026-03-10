# Step 02: 訓練用データの収集

## 目的

Step 01の種リスト（690種）に基づき、モデル訓練に必要な音声データを収集する。
Perch 2.0を参考に、Xeno-canto + iNat Sounds 2024 を主力データソースとする。

## データソース

| ソース | 種カバレッジ | メタデータ録音数 | DL可能 | 日本録音 |
|--------|------------|----------------|--------|---------|
| **Xeno-canto** | 658/688 (95.6%) | 290,810件 | 290,810件 | 2,730件/239種 |
| **iNat Sounds S3** | 412/688 (59.9%) | 33,917件 | 33,917件 | 381件/105種 |
| **iNat API** | 567/688 (82.4%) | 335,657件 | 335,657件 | — |
| **Macaulay Library** | 646/688 (93.9%) | 503,206件 | 25,968件 (304種) | 3,082件 |
| FSD50K | — (環境音) | 51K clips | — | — |
| **合計 (DL可能)** | **668/688 (97.1%)** | — | **686,352件** | — |

## 実行方法

### 環境準備

```bash
pip install pyarrow aiohttp pyyaml tqdm soundfile

# 音声DLに必要（XC WAV変換時のみ）
sudo apt install ffmpeg

# iNat S3ダウンロードに必要
pip install awscli
```

### Xeno-canto メタデータ収集

```bash
# APIキーを keys/xenocant_api.key に配置すること

# メタデータ収集（全688種、約4時間）
python steps/02_data_collection/collect_xc.py --metadata-only

# テスト: 5種だけ
python steps/02_data_collection/collect_xc.py --metadata-only --limit 5
```

### Xeno-canto 音声ダウンロード

```bash
# ドライラン（対象件数の確認）
python steps/02_data_collection/collect_xc.py --download --dry-run

# MP3のまま保存（推奨: ~651GB、WAV変換は訓練パイプラインで実施）
python steps/02_data_collection/collect_xc.py --download --format mp3

# WAV変換して保存（~3.1TB、要ffmpeg）
python steps/02_data_collection/collect_xc.py --download --format wav

# テスト: 3種だけ
python steps/02_data_collection/collect_xc.py --download --limit 3

# 中断再開: 既にDL済みファイルは自動スキップ
python steps/02_data_collection/collect_xc.py --download --format mp3
```

### iNat Sounds 2024

```bash
# アノテーション解析（JSON DL + 種マッチング）
python steps/02_data_collection/collect_inat.py --annotations-only

# ドライラン
python steps/02_data_collection/collect_inat.py --download --dry-run

# 音声ダウンロード（S3から、要 aws cli、約92GB）
python steps/02_data_collection/collect_inat.py --download
```

### 一括ダウンロード（推奨）

```bash
# XC + iNat 一括（リトライ・ログ付き、tmux推奨）
tmux new -s torinet-dl 'bash steps/02_data_collection/download_all.sh'

# XCのみ
bash steps/02_data_collection/download_all.sh --xc-only

# iNatのみ
bash steps/02_data_collection/download_all.sh --inat-only

# WAV変換あり（~3.1TB、要ffmpeg）
bash steps/02_data_collection/download_all.sh --wav
```

- Ctrl+C で中断しても進捗は保存され、再実行で自動再開
- ログは `steps/02_data_collection/logs/` に保存

## 結果

### iNat Sounds 2024 アノテーション解析

- **マッチ種数**: 412/688 (59.9%)
- **総録音数**: 33,917件
  - train: 24,971件 (412種)
  - val: 8,946件 (196種)
- **日本録音**: 381件 (105種)
- **種ごとの録音数**:
  - Mean: 82.3, Median: 8, Min: 1, Max: 1,100
- **未マッチ種**: 270/688（多くは日本固有種・希少種）

### Xeno-canto メタデータ収集

- **種カバレッジ**: 658/688 (95.6%)
- **総録音数**: 290,810件
- **品質分布**:
  - A: 46,026 (15.8%)
  - B: 135,571 (46.6%)
  - C: 84,871 (29.2%)
  - D: 20,035 (6.9%)
  - E: 4,105 (1.4%)
- **日本録音**: 2,730件 (239種)
- **種ごとの録音数**:
  - Mean: 442.0, Median: 102, Min: 1, Max: 8,359
- **未カバー種**: 21/688（海鳥・絶滅種・希少種が中心）
- **注意**: API v3のlongitudeフィールド名修正済み（`lng` → `lon`）。既存データのis_japanはcountryフィールドから再計算

### Macaulay Library

- **メタデータ**: 503,206件 (646/688種, 98.7% of eBird-matched)
- **ダウンロード済み**: 25,968件 (304種) — P1/P2優先種向けに正式申請
  - WAV: 20,078件, MP3: 3,224件, M4A: 2,595件
  - 合計容量: 176 GB (5バッチ)
  - 日本録音: 3,082件
- **注意**: Cornell Lab研究ライセンス（eBird Any Lab Use）

### 全ソース合算カバレッジ（DL可能ベース）

- **合計種カバレッジ**: 668/688 (97.1%)
- **合計DL可能録音数**: 686,352件
- **優先度ティア分布（定期観察種515種）**:
  - P1 (DL可能 < 50): 72種
  - P2 (DL可能 50-99): 68種
  - P3 (DL可能 ≥ 100): 375種

### 統合メタデータ

- **総録音数**: 556,914件（DL済み・重複除去後）
- **種カバレッジ**: 667種
- **日本録音**: 9,171件
- **ソース内訳**:
  - Xeno-canto: 262,038件 (642種)
  - iNat API: 235,062件 (528種)
  - iNat Sounds S3: 33,917件 (412種)
  - Macaulay: 25,897件 (304種)
- **種ごとの録音数**: Mean 835, Median 181, Min 1, Max 39,826
- **重複除去**:
  - XC: 2,762件（同一レコーディングの重複行）
  - iNat API: 12,651件（同一音声ファイルが複数観察に紐づくケース）
  - Macaulay: 71件（メタデータ内の重複）
- **ファイル存在率**: 100%（1,000件サンプルチェック）

## ファイル構成

```
steps/02_data_collection/
├── README.md                # 本ファイル
├── config.yaml              # 設定（NASパス、APIエンドポイント等）
├── utils.py                 # 共通ユーティリティ
├── collect_xc.py            # XC メタデータ収集 + 音声DL
├── collect_inat.py          # iNat アノテーション解析 + 音声DL
├── collect_inat_api.py      # iNat API 直接検索
├── collect_macaulay.py      # ML メタデータ収集 + 音声DL
├── generate_ml_request.py   # ML申請用カタログリスト生成
├── analyze_coverage_gaps.py # カバレッジギャップ分析
├── visualize_coverage.py    # カバレッジ可視化
├── build_unified_metadata.py # 統合メタデータ構築
├── organize_macaulay.py     # ML tar展開 → 種別ディレクトリ整理
├── download_all.sh          # 一括DL（リトライ・ログ付き）
├── ml_request/              # ML申請用CSV（4バッチ）
├── figures/                 # 分析結果の可視化
└── logs/                    # DLログ（自動生成）
```

## NASデータ構成

```
~/NAS/nasbi/ToriNET/
├── audio/
│   ├── xeno-canto/
│   │   ├── metadata/            # xc_metadata.parquet
│   │   └── wav/{species_code}/  # WAV変換後
│   ├── inat-sounds/
│   │   ├── annotations/         # train.json, val.json, inat_metadata.parquet
│   │   ├── raw/                 # S3からの元データ
│   │   └── filtered/{species_code}/  # マッチ種のみ
│   ├── inat-api/
│   │   ├── metadata/            # inat_api_metadata.parquet
│   │   └── audio/{species_code}/
│   ├── macaulay/
│   │   ├── metadata/            # ml_metadata.parquet, collection_progress.json
│   │   └── audio/{species_code}/ # 展開済み音声 (25,897件, 304種)
│   └── fsd50k/                  # 後続
└── metadata/
    └── unified_metadata.parquet # 統合メタデータ (556,914件, 667種)
```

## TODO

- [x] 共通ユーティリティ・設定ファイル作成
- [x] iNat Sounds 2024 アノテーション解析
- [x] Xeno-canto APIキー配置 → メタデータ収集
- [x] iNat API 直接検索
- [x] Macaulay Library メタデータ収集
- [x] Macaulay Library 正式申請 → ダウンロード完了 (25,968件, 304種)
- [x] カバレッジギャップ分析更新
- [x] Macaulay tar 展開 → 種別ディレクトリに整理 (25,897件, 304種)
- [x] 全ソース音声ダウンロード（XC + iNat S3 + iNat API）完了
- [x] 統合メタデータ作成 (556,914件, 667種, ファイル存在率100%)
- [ ] FSD50K 環境音の取得（後続ステップ）
