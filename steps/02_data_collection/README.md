# Step 02: 訓練用データの収集

## 目的

Step 01の種リスト（690種）に基づき、モデル訓練に必要な音声データを収集する。
Perch 2.0を参考に、Xeno-canto + iNat Sounds 2024 を主力データソースとする。

## データソース

| ソース | 種カバレッジ | 推定録音数 | 日本録音 | 優先度 |
|--------|------------|-----------|---------|--------|
| **Xeno-canto** | 658/688 (95.6%) | 290,810件 | 2,730件/239種 | **1** |
| **iNat Sounds 2024** | 412/688 (59.9%) | 33,917件 | 381件/105種 | **2** |
| FSD50K | — (環境音) | 51K clips | — | 3 |

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

### XC + iNat 合算カバレッジ

- **合計種カバレッジ**: XC 658種 + iNat 412種 = 合算 660/688 (95.9%)（重複410種）
- **合計録音数**: 290,810 + 33,917 = 324,727件
- **合計日本録音**: 2,730 + 381 = 3,111件

## ファイル構成

```
steps/02_data_collection/
├── README.md           # 本ファイル
├── config.yaml         # 設定（NASパス、APIエンドポイント等）
├── utils.py            # 共通ユーティリティ
├── collect_xc.py       # XC メタデータ収集 + 音声DL
├── collect_inat.py     # iNat アノテーション解析 + 音声DL
├── download_all.sh     # 一括DL（リトライ・ログ付き）
└── logs/               # DLログ（自動生成）
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
│   └── fsd50k/                  # 後続
└── metadata/
    └── unified_metadata.parquet # XC + iNat 統合（後続）
```

## TODO

- [x] 共通ユーティリティ・設定ファイル作成
- [x] iNat Sounds 2024 アノテーション解析
- [x] Xeno-canto APIキー配置 → メタデータ収集
- [ ] メタデータ確認後、音声ダウンロード判断
- [ ] FSD50K 環境音の取得（後続ステップ）
