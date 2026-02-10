# ToriNet — 日本向け汎用生物音響認識モデル

BirdNETやGoogle Perchのような大規模生物音響認識モデルを、日本の生物・環境音に特化して構築するプロジェクト。

## 目標

- 日本国内に生息する鳥類・両生類・哺乳類・昆虫等の鳴き声を高精度に検出・識別
    - とりあえずは鳥をターゲットにするが、入手できそうなデータは多分類群でも集めておく
- 日本の環境音（風、雨、川、車両、人声など）のクラスも含めた汎用モデル
- 実際のサウンドスケープ録音（PAM: Passive Acoustic Monitoring）で実用的な精度を達成

## 設計方針

### タスク定義

- **マルチラベル・セグメンテーション**: 音声を時間フレームに分割し、各フレームに対して複数ラベルを付与
- フレーム長は3秒程度を想定（BirdNET準拠）、ただし要検討

### 訓練戦略

```
フォーカル録音（ラベル付き）
    ↓
サウンドスケープシミュレーション（augmentation）
    ↓  ・複数種の音声を重畳
    ↓  ・背景ノイズの付加
    ↓  ・SNR、タイミング、空間特性のランダム化
    ↓
擬似サウンドスケープ（訓練用）
```

- フォーカル録音はクリーンなラベルを持つため、シミュレーションで混合しても正確なラベルが得られる
- 実際のPAM録音環境を模擬することで、domain gapを軽減

### 評価

- テストには実際のサウンドスケープ録音（人手アノテーション済み）を使用
- 訓練データとテストデータのドメインが異なる（focal vs. soundscape）ことが本質的な課題

## ステップ

| Step | ディレクトリ | 内容 | 状態 |
|------|-------------|------|------|
| 01 | `steps/01_species_list/` | 種リストの構築 | 完了 |
| 02 | `steps/02_data_collection/` | 訓練用データの収集 | 作業中 |
| 03 | — | 前処理・サウンドスケープシミュレーション | 未着手 |
| 04 | — | モデル訓練・評価 | 未着手 |

各ステップの詳細は `steps/NN_xxx/README.md` を参照。

## モデルアーキテクチャ（検討中）

### 候補

1. **BirdNET方式**: ResNet系backbone + 分類ヘッド（実績あり、ベースライン）
2. **Perch方式**: EfficientNet backbone + embedding（転移学習に強い）
3. **Audio Spectrogram Transformer (AST)**: Transformer系（高精度だが計算コスト大）
4. **BEATs / Audio-MAE**: 自己教師あり事前学習 + fine-tuning

### 入力表現

- メルスペクトログラム（標準的）
- サンプルレート: 32kHz or 48kHz（鳥類の高周波をカバー）
- 周波数範囲: 0–16kHz 程度（生物音の主要帯域）

## 評価指標（検討中）

- フレームレベル: mAP、AUROC、F1（種ごと & マクロ平均）
- イベントレベル: segment-based precision/recall
- 種の検出有無: recording-level metrics

## 参考文献

- Kahl et al. (2021) "BirdNET: A deep learning solution for avian diversity monitoring"
- Ghani et al. (2023) "Global birdsong embeddings enable superior transfer learning for bioacoustic classification" (Perch)
- Gong et al. (2021) "AST: Audio Spectrogram Transformer"
- Chen et al. (2022) "BEATs: Audio Pre-Training with Acoustic Tokenizers"
