# ToriNet - Project Guide for Claude

## Overview

日本国内の生物音響を高精度に認識する汎用モデル（BirdNET/Google Perch相当）の開発プロジェクト。

## Key Decisions

- **タスク**: マルチラベル・セグメンテーション（時間軸上で複数種を同時検出）
- **訓練データ**: ラベル付きフォーカル録音 + サウンドスケープシミュレーションによる強いaugmentation
- **テストデータ**: 実際のサウンドスケープ録音（PAM等）
- **データ保存先**: `~/NAS/nasbi/ToriNET/`

## Repository Structure

```
torinet/
├── CLAUDE.md
├── README.md
├── steps/
│   ├── 01_species_list/    # 種リストの構築
│   │   ├── README.md           # 作業記録・結果サマリ
│   │   ├── build_species_list.py
│   │   └── species_list.csv
│   ├── 02_data_collection/ # 訓練データの収集
│   │   └── README.md
│   ├── 03_.../             # (future steps)
│   ...
└── .venv/
```

各ステップは `steps/NN_name/` に格納し、それぞれに README.md で作業記録を残す。

## NAS Data Structure (`~/NAS/nasbi/ToriNET/`)

```
ToriNET/
├── taxonomy/           # 分類リスト・ラベルファイル
│   ├── jpbirdlist8ed.xlsx          # 日本鳥類目録第8版
│   ├── ebird_taxonomy_v2024.csv    # eBird/Clements taxonomy
│   ├── birdnet_labels_en.txt       # BirdNET V2.4 英語ラベル
│   └── birdnet_labels_ja.txt       # BirdNET V2.4 日本語ラベル
├── audio/
│   ├── xeno-canto/     # Xeno-cantoからのフォーカル録音
│   ├── environment/    # 背景ノイズ・環境音
│   └── soundscape/     # PAMサウンドスケープ（テスト用）
└── gbif_ebird/         # GBIF eBird export（既存、巨大）
```

## Species List

- マスターリスト: 日本鳥類目録改訂第8版（日本鳥学会, 690種）
- eBird/Clements v2024 とのマッピング: 688/690 (99.7%)
- BirdNET V2.4 とのマッピング: 605/690 (87.7%)
- 分類体系間のシノニムテーブルは `steps/01_species_list/build_species_list.py` に定義

## Conventions

- Language: Python
- Config format: YAML (Hydra or similar)
- Audio processing: librosa / torchaudio
- Code style: ruff format + lint
- Docstrings: 日本語OK、コード・変数名は英語
- 各ステップに README.md で作業記録を残す

## Important Notes

- NAS `gbif_ebird/occurrence.txt` は ~1.7TB。直接全読みしないこと
- Xeno-canto API v3 はAPIキーが必要（v2は廃止済み）
