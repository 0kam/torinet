# Phase A: TweetyNet bootstrap (履歴記録)

> この文書は Phase A（TweetyNet 初期学習 + self-training）の作業記録です。
> 成果物として `models/tweetynet_r2_best.pt` が得られ、現行の v3 パイプライン
> （[README.md](README.md)）で固定モデルとして使用されています。Phase A 自体
> は再実行しません。関連スクリプト・疑似ラベル・ラウンドごとの学習データは
> `legacy/` 以下に保存されています。

## 概要

TweetyNet (Cohen et al., eLife 2022) のアーキテクチャを PyTorch で直接実装し、
信号処理アンサンブルの疑似ラベルで自己学習するパイプライン。

vakフレームワークを通さず直接実装した理由:
1. bird vs background の **2クラス問題** であり、vakの多クラスパイプラインは不要
2. 7種の信号処理手法のアンサンブル **疑似ラベル** で自己学習する独自ワークフロー
3. 既存の `prototype_segmentation.py` と統一的に比較したい
4. 設定ファイル管理やデータセット形式変換のオーバーヘッドを避けたい

## アーキテクチャ

```
Input: (batch, 1, 128, time_steps) — log-mel spectrogram
  ↓
Conv2d(1→32, 5×5) → BN → ReLU → MaxPool(8,1)    # (B, 32, 16, T)
  ↓
Conv2d(32→64, 5×5) → BN → ReLU → MaxPool(8,1)   # (B, 64, 2, T)
  ↓
Reshape → (B, T, 128)
  ↓
BiLSTM(128 hidden, 2 layers, dropout=0.2)          # (B, T, 256)
  ↓
Dropout(0.2) → Linear(256→2)                       # (B, T, 2)
  ↓
Output: (batch, 2, time_steps) — [background, bird] logits
```

| パラメータ | 値 |
|-----------|-----|
| 入力 | 128-bin mel spectrogram (150–12000Hz) |
| サンプリングレート | 32kHz (リサンプリング) |
| FFT窓長 | 1024 |
| ホップ長 | 320 (10ms/frame) |
| コンテキスト窓 | 200 frames (2s) |
| 総パラメータ | 712,258 |

## ファイル構成

```
prototype_tweetynet.py     # メインスクリプト
pseudo_labels/             # 初期疑似ラベル (npz)
  └── {species}/
      └── {safe_id}.npz   # keys: spectrogram (128, T), labels (T,)
models/
  ├── tweetynet_best.pt    # 初期学習ベストモデル
  ├── tweetynet_final.pt   # 初期学習最終モデル
  ├── tweetynet_r0_best.pt # Self-training Round 0 ベスト
  ├── tweetynet_r1_best.pt # Self-training Round 1 ベスト
  ├── tweetynet_r2_best.pt # Self-training Round 2 ベスト
  └── *.json / *.png       # 学習履歴 + 曲線
```

## ワークフロー

### Phase 1: 初期疑似ラベル生成

```bash
python prototype_tweetynet.py generate-labels
```

信号処理7手法のOR-アンサンブルでフレームレベル疑似ラベルを生成:
- 各フレームに対し7手法が独立に検出を実行
- **≥2手法が検出** → bird ラベル
- 形態学的後処理: 50ms gap fill, 30ms minimum duration
- npz形式で保存（spectrogram + labels）

### Phase 2: 初期学習

```bash
python prototype_tweetynet.py train --epochs 30 --device cuda
```

- 種レベル 80/20 train/val 分割（種間汎化を評価）
- 2秒パッチのランダムクロップ
- データ拡張: time masking, freq masking, additive noise
- クラス重み付き CrossEntropyLoss（不均衡対応）
- AdamW (lr=1e-3, weight_decay=1e-4)
- ReduceLROnPlateau + Early stopping (patience=5)

### Phase 3: Self-training による精製

```bash
# 教師メソッドの評価・選定
python self_training.py evaluate-teachers --f1-threshold 0.4

# 精製ラベル生成（選定教師のみで投票）
python self_training.py generate-refined-labels --min-votes 2

# 3ラウンドの自己学習
python self_training.py self-train --rounds 3 --device cuda
```

#### Self-training ラウンド構成

| Round | ラベル | Epochs | LR | 信頼度フィルタ |
|-------|--------|--------|-----|---------------|
| 0 | 精製疑似ラベル | 30 | 1e-3 | なし（全ラベル使用） |
| 1 | Round 0 予測 | 20 | 5e-4 | P>0.9=bird, P<0.1=bg, 他はマスク |
| 2 | Round 1 予測 | 15 | 2e-4 | P>0.85=bird, P<0.15=bg, 他はマスク |

#### 教師評価 (Teacher Evaluation)

各SPメソッドの検出品質をフレームレベルP/R/F1で評価。
F1 ≥ 閾値（デフォルト0.4）のメソッドを教師として選定し、精製ラベル生成に使用。

#### 崩壊防止

- bird-frame比率を各ラウンドで監視（50%超変化で中断）
- mask比率を監視（80%超で警告）

### Phase 4: 予測

```bash
python prototype_tweetynet.py predict --device cuda
```

- スライディングウィンドウ（50%オーバーラップ）で全長予測
- フレーム確率の重み付き平均 → 閾値(0.5) → 形態学後処理 → セグメント

### 追加オプション

```bash
# チェックポイントから再開
python prototype_tweetynet.py train --resume models/tweetynet_r0_best.pt --lr 5e-4

# カスタムラベルディレクトリ
python prototype_tweetynet.py train --labels-dir refined_pseudo_labels

# モデル名指定
python prototype_tweetynet.py train --model-name tweetynet_custom
```

## 実験結果

### 初期学習（疑似ラベル, 250ファイル）

| 指標 | 値 |
|------|-----|
| 学習データ | 200ファイル / 40種 (3,582パッチ) |
| 検証データ | 50ファイル / 10種 (1,116パッチ) |
| ベスト val_loss | 0.4134 (epoch 2) |
| ベスト val_acc | ~0.80 |
| Early stopping | epoch 7 |

### Self-training 結果

| Round | ラベル | Best Epoch | Val Loss | Val Acc |
|-------|--------|-----------|----------|---------|
| 0 | 精製疑似ラベル | 9 | 0.3479 | 86.3% |
| 1 | R0予測 (mask P∈[0.1,0.9]) | 11 | 0.0104 | 99.8% |
| 2 | R1予測 (mask P∈[0.15,0.85]) | 14 | 0.1034 | 97.7% |

精製モデル（R2）による最終予測: 11,173セグメント (250ファイル, 平均44.7/ファイル)

### Teacher評価結果

| Method | Precision | Recall | F1 | 選定 |
|--------|-----------|--------|----|------|
| M1 Band Energy Hysteresis | 0.654 | 0.610 | 0.631 | ✓ |
| M2 PCEN Connected Components | 0.586 | 0.604 | 0.595 | ✓ |
| M3 Flux-Anchor Boundary | 0.347 | 0.868 | 0.496 | ✓ |
| M6 REPET-lite Foreground | 0.633 | 0.764 | 0.692 | ✓ |
| M8 2-State HMM (Viterbi) | 0.597 | 0.763 | 0.670 | ✓ |
| M12 Spectral Entropy | 0.639 | 0.215 | 0.322 | ✗ |
| M14 Median Clipping (Lasseck) | 0.492 | 0.234 | 0.317 | ✗ |

M12（Spectral Entropy）とM14（Median Clipping）は低recall (0.2前後) のため教師から除外。
5メソッド (M1, M2, M3, M6, M8) を教師として精製ラベル生成に使用。

## 参考文献

- Cohen, Y. et al. (2022). Automated annotation of birdsong with a neural network that segments spectrograms. eLife, 11, e63853.
- TweetyNet GitHub: https://github.com/yardencsGitHub/tweetynet
- vak GitHub: https://github.com/vocalpy/vak
