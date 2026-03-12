# TweetyNet によるフレームセグメンテーション実験

## 調査結果

### vak/TweetyNet について

- **vak** (v1.0.5): 音声コミュニケーション研究向けのニューラルネットワークフレームワーク
  - TOML設定ファイルベースのCLIワークフロー (`vak prep`, `vak train`, `vak predict`)
  - 依存: PyTorch >= 2.7.0, PyTorch Lightning, crowsetta, dask 等
  - 現環境 (PyTorch 2.10.0+cu128, Python 3.12) と**互換性あり**
  - ただし多数のシラブルクラス分類を想定した設計で、2クラス問題には過剰

- **TweetyNet**: スペクトログラムのフレームレベルセグメンテーション用ニューラルネット
  - 論文: Cohen et al., eLife 2022 ("Automated annotation of birdsong with a neural network that segments spectrograms")
  - アーキテクチャ: 2x (Conv2d → ReLU → MaxPool) → BiLSTM → Linear
  - 入力: `(batch, 1, n_freq, n_time)` スペクトログラム
  - 出力: `(batch, n_classes, n_time)` フレームごとのクラスロジット
  - パラメータ数: 約712K（軽量）

- **crowsetta**: アノテーション形式変換ライブラリ
  - `simple-seq` 形式: CSV with `onset_s, offset_s, label` カラム
  - 1音声ファイル = 1アノテーションファイルの対応

### 採用方針

vakフレームワークを通さず、**TweetyNetアーキテクチャをPyTorchで直接実装**した。理由:

1. 本プロジェクトは **bird vs background の2クラス問題**であり、vakの多クラスシラブル分類パイプラインは不要
2. 既存の信号処理手法7種の結果をアンサンブルした**疑似ラベル**で自己学習する独自ワークフローが必要
3. 推論結果を既存の `prototype_segmentation.py` と統一的に比較したい
4. vakの設定ファイル管理やデータセット形式変換のオーバーヘッドを避けたい

## 実装

### ファイル構成

```
steps/03_segment_extraction/
├── prototype_tweetynet.py    # 本スクリプト（全機能統合）
├── pseudo_labels/            # 疑似ラベル (npz形式)
│   └── {species_code}/
│       └── {recording_id}.npz  # spectrogram + labels
├── models/
│   ├── tweetynet_best.pt     # ベストモデル (val_loss基準)
│   ├── tweetynet_final.pt    # 最終エポックモデル
│   ├── training_history.json # 学習曲線データ
│   └── training_curves.png   # 学習曲線プロット
├── tweetynet_results.csv     # 予測結果サマリ
└── README_tweetynet.md       # 本ファイル
```

出力先: `~/NAS/nasbi/ToriNET/segments/test_samples_results_tweetynet/`

### ワークフロー

#### Phase 1: 疑似ラベル生成

```bash
python prototype_tweetynet.py generate-labels
```

- 既存の信号処理手法7種 (`prototype_segmentation.py`) をOR-アンサンブル
- 各フレームについて **2手法以上が検出** → bird ラベル
- 形態学的後処理 (50ms gap fill, 30ms minimum duration)
- スペクトログラムとラベルを npz で保存

#### Phase 2: TweetyNetの学習

```bash
python prototype_tweetynet.py train --epochs 30 --device cuda
```

- 種レベルで 80/20 の train/val 分割（種間汎化を評価）
- 2秒パッチのランダムクロップで学習
- データ拡張: time/freq masking, additive noise
- クラス重み付き CrossEntropyLoss（不均衡対応）
- Early stopping (patience=5)

#### Phase 3: 予測

```bash
python prototype_tweetynet.py predict --device cuda
```

- スライディングウィンドウ（50%オーバーラップ）で全長予測
- フレーム確率 → 閾値 → 形態学後処理 → (onset, offset) セグメント

#### Phase 4: 可視化

```bash
python prototype_tweetynet.py visualize [--species jabwar] [--limit 20]
```

- スペクトログラム + P(bird)確率 + セグメント + 疑似ラベル比較

### モデル仕様

| パラメータ | 値 |
|-----------|-----|
| 入力 | 128-bin mel spectrogram (150-12000Hz) |
| サンプリングレート | 32kHz |
| ホップ長 | 320 (10ms/frame) |
| コンテキスト窓 | 200 frames (2s) |
| Conv1 | 32 filters, 5x5, MaxPool 8x1 |
| Conv2 | 64 filters, 5x5, MaxPool 8x1 |
| BiLSTM | 128 hidden, 2 layers |
| 出力 | 2クラス (background, bird) |
| 総パラメータ | 712,258 |

### 実験結果

| 指標 | 値 |
|------|-----|
| 学習データ | 200ファイル / 40種 (3,582パッチ) |
| 検証データ | 50ファイル / 10種 (1,116パッチ) |
| ベスト val_loss | 0.4134 (epoch 2) |
| ベスト val_acc | ~0.80 |
| Early stopping | epoch 7 |
| 予測セグメント数 | 12,668 (250ファイル) |
| 平均セグメント/ファイル | 50.7 |

### 考察と次のステップ

1. **疑似ラベルの品質**: 信号処理アンサンブルからの疑似ラベルは完璧ではないため、val_acc ~80% は疑似ラベルとの一致度であって絶対的な精度ではない
2. **種間汎化**: 検証は未知の10種で行っており、一定の汎化性能がある
3. **改善案**:
   - 人手でアノテーションした少量のデータで fine-tuning（半教師あり）
   - 複数ラウンドの self-training（予測→フィルタ→再学習）
   - BirdNET/Perch embedding をTweetyNetの入力特徴量に追加
4. **統合**: 信号処理手法のアンサンブルよりTweetyNetの方が**滑らかで一貫した**セグメント境界を出す傾向がある。最終的なPhase 2の全種抽出パイプラインに組み込む候補

## 参考文献

- Cohen, Y. et al. (2022). Automated annotation of birdsong with a neural network that segments spectrograms. eLife, 11, e63853.
- TweetyNet GitHub: https://github.com/yardencsGitHub/tweetynet
- vak GitHub: https://github.com/vocalpy/vak
- crowsetta: https://crowsetta.readthedocs.io/
