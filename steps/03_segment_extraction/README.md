# Step 03: クリーンセグメント抽出 (Pipeline v3)

## 目的

Step 02 で収集した録音から、各種の鳴き声を正確に切り出した **クリーンセグメント** を作成し、
後続のサウンドスケープシミュレーション（セグメンテーションタスク）で訓練データとして使う。

## ゴール

- 各種 **最低 50 個、最大 100 個** のクリーンセグメントを用意
- 対象種の鳴き声のみを含み、他種の鳴き声や過度な雑音がないこと
- セグメント長は鳴き声の長さに一致（固定長ではない）
- 各セグメントの元録音 ID を記録（train/eval 分割でリーク防止）

## 設計の肝 — Early routing

BirdNET で十分に検出できる種（B-1）には Perch プロトタイプは不要。重い処理
（Perch 埋め込み・HDBSCAN クラスタリング）は **B-2 種にのみ** 限定する。
振り分けは `species_routing.csv` を経由して明示化され、閾値は CLI から変更可能。

| 判定 | 条件 | 選定ゲート |
|---|---|---|
| **B-1** | BirdNET 登録 & ヒット率 ≥ 0.3 & ヒット数 ≥ 10 | BirdNET conf ≥ 0.2 |
| **B-2** | BirdNET 非登録 / ヒット不足 | Perch プロトタイプ cosine ≥ 0.3 |

ランキングはいずれも `gate_score × bioacoustic_quality`。Bioacoustic quality は
`σ(0.3·(SNR−10)) × max((NDSI+1)/2, 0.3)` で SNR を対数的に、NDSI を低周波種
保護のため下限クリップしたもの。

## パイプライン全体図

```
┌──────────────────────────────────────────────────────────────┐
│ 1. separate                    Bird-MixIT 4-source separation │
│                                → birdmixit_sources/           │
│                                                               │
│ 2. compute-acoustic-features   SNR / NDSI / bird_ratio        │
│                                (librosa + TweetyNet r2)       │
│                                → acoustic_features/           │
│                                                               │
│ 3. birdnet-score               BirdNET v2.4 × 4ch             │
│                                (登録種のみ, subprocess)        │
│                                → birdnet_scores/              │
│                                                               │
│ 4. route-species               B-1 / B-2 判定                 │
│                                → species_routing.csv          │
├──────────────────────────────────────────────────────────────┤
│ 5. compute-perch-embeddings    Perch v2 埋め込み (B-2 のみ)   │
│                                → perch_embeddings/            │
│                                                               │
│ 6. build-prototypes            HDBSCAN (B-2 のみ)             │
│                                → species_prototypes/          │
├──────────────────────────────────────────────────────────────┤
│ 7. channel-select              focal_channel gate+rank         │
│                                → channel_selection/           │
│                                                               │
│ 8. segment                     TweetyNet r2 で focal_ch 予測  │
│                                note → bout                    │
│                                → tweetynet_segments/          │
│                                                               │
│ 9. select                      rank_score 上位を export       │
│                                → birdmixit_pipeline_results.csv│
│                                → birdmixit_selected/          │
└──────────────────────────────────────────────────────────────┘
```

`select` サブコマンドは 3〜9 までを必要に応じて自動で走らせ（キャッシュ有効）、
`all` は 1〜9 を一気通貫で実行する。

## ファイル構成

### 現役（v3 パイプライン）

```
steps/03_segment_extraction/
├── README.md                    本ファイル
├── README_phaseA.md             Phase A (TweetyNet bootstrap) の作業記録
│
├── birdmixit_pipeline.py        v3 パイプライン本体（9 サブコマンド）
├── prototype_tweetynet.py       TweetyNet の学習・予測（新規学習用、現行は推論のみ使用）
├── taxonomy_maps.py             eBird/BirdNET 分類マッピング
├── cluster_selector.py          B-2 種のクラスタ目視選定 Web UI
├── separation_viewer.py         分離結果・選定 bout の Web UI
├── select_test_samples.py       種優先度 + テストサンプリング
│
├── species_priority.csv         種の優先順位
├── test_samples.csv             選定された録音一覧
├── birdmixit_pipeline_results.csv  最終選定 bout (latest run)
│
├── models/                      Phase A 由来の TweetyNet 重み
│   └── tweetynet_r2_best.pt     現行使用モデル
└── species_prototypes/          build-prototypes の出力（B-2 種のみ）
```

### Legacy（v2 および Phase A 作業成果物）

```
legacy/
├── self_training.py, prototype_segmentation.py, ...
├── bout_pipeline.py, bout_embedding.py, ...
├── results_viewer.py, method_viewer.py, method_comparison.py
├── pseudo_labels/, refined_pseudo_labels/, self_train_labels_r{1,2}/
├── cluster_visualizations/, note_cluster_visualizations/
├── precompute_features.py       （birdmixit_pipeline.py に統合済み）
└── *.csv                        旧手法の比較結果
```

## NAS 上の出力先 (`~/NAS/nasbi/ToriNET/segments/`)

```
segments/
├── birdmixit_sources/           1. 4ch 分離 WAV
├── acoustic_features/           2. SNR/NDSI/bird_ratio npz
├── birdnet_scores/              3. BirdNET 4ch conf CSV
├── species_routing.csv          4. B-1 / B-2 判定結果
├── perch_embeddings/            5. Perch v2 1280-d npz (B-2 種のみ)
├── channel_selection/           7. 採用 focal_channel per recording
├── tweetynet_segments/          8. bout-level note grouping
└── birdmixit_selected/          9. 最終選定 bout WAV
```

## npz フォーマット

### `acoustic_features/{sp}/{rec}_src{ch}.npz`

| キー | 型 | 形状 | 内容 |
|---|---|---|---|
| `snr` | float32 | (n_windows,) | 窓ごとの SNR (dB) |
| `ndsi` | float32 | (n_windows,) | NDSI (−1 ~ +1) |
| `bird_ratio` | float32 | (n_windows,) | P(bird)>0.5 のフレーム比率 |
| `window_starts` | int32 | (n_windows,) | 先頭サンプル位置（MIXIT_SR=22050基準）|

### `perch_embeddings/{sp}/{rec}_src{ch}.npz`

| キー | 型 | 形状 | 内容 |
|---|---|---|---|
| `embeddings` | float32 | (n_windows, 1280) | L2 正規化済み Perch v2 埋め込み |
| `window_starts` | int32 | (n_windows,) | `acoustic_features` と同じ基準 |

## 使い方

### 代表的なコマンド

```bash
# 全パイプライン (1→9)
python birdmixit_pipeline.py all

# stage 3-9 のみ（分離と acoustic features は既存想定）
python birdmixit_pipeline.py select

# ルーティング閾値を変更して再判定
python birdmixit_pipeline.py route-species \
    --birdnet-hit-rate-min 0.5 --birdnet-hit-count-min 15

# B-2 種の目視でクラスタ選定（B-2 にルーティングされた種のみ対象）
python cluster_selector.py --port 8053

# 最終選定結果の閲覧
python separation_viewer.py --port 8052
```

### 1 種だけ試す

```bash
python birdmixit_pipeline.py all --species azwmag2
```

`--species` は `separate` 以外のすべてのサブコマンドでも有効。

### 途中から再開

各ステージは出力の存在をチェックするため、同じコマンドを再実行すれば未処理分
のみが走る。`--force` ですべて再計算。

## 成果物フォーマット

### `birdmixit_pipeline_results.csv` 主要カラム

| カラム | 説明 |
|---|---|
| `species_code`, `scientific_name` | eBird コード / 学名 |
| `recording_id`, `safe_id`, `focal_channel` | 元録音と採用 channel |
| `method` | `B1` / `B2` |
| `species_score` | BirdNET conf（B-1）または proto cos sim（B-2）|
| `bout_snr`, `bout_ndsi`, `bout_quality` | note 連結音声の品質 |
| `rank_score` | `species_score × bout_quality` |
| `bout_onset`, `bout_offset`, `bout_duration` | bout 時刻 (秒) |
| `notes_json` | bout 内の note (onset, offset) リスト |

### `species_routing.csv`

| カラム | 説明 |
|---|---|
| `route` | `B1` or `B2` |
| `reason` | `birdnet_sufficient` / `birdnet_unregistered` / `birdnet_low_hit_rate` / `birdnet_low_hit_count` / `birdnet_not_scored` |
| `birdnet_coverage` | `both` / `birdnet` / `ebird` / `none` |
| `n_recordings`, `n_valid_scored`, `n_above_gate`, `hit_rate` | 判定根拠 |
| `hit_rate_threshold`, `hit_count_threshold` | 使用した閾値（監査用）|

## 実装上の注意

- **BirdNET TFLite は FD リーク** するので必ず subprocess で呼ぶ（`birdmixit_pipeline.py` 内で spawn process + batch=20）。
- **Perch v2 は TF2 Hub から動的ロード**。初回は ~数十秒。
- **Bird-MixIT は TF1 CPU 限定**。TF1 ワーカープロセスでは `CUDA_VISIBLE_DEVICES=-1` を設定し、PyTorch GPU の可用性に影響しないよう隔離。
- **長尺録音は 10 分で打ち切り**（`MAX_SEPARATE_DURATION_S`）。Bird-MixIT は 45 分級の入力で TDCN++ 中間テンソルが爆発して OOM するため、`librosa.load(..., duration=...)` で先頭のみを読み込む。各録音から 50–100 本の bout を取れれば十分なので 10 分で事足りる。
- **SNR/NDSI と Perch 埋め込みは独立 npz** に分離（v3 でリファクタ済み）。B-1 種では Perch 埋め込みを生成しない。
- **低周波種保護**：`_bioacoustic_quality` で `ndsi_norm` を 0.3 でクリップし、1-2 kHz の anthrophony と鳴き声がオーバーラップする種でも quality がゼロに潰れないように調整。

## Phase A (Bootstrap) 履歴

現行の `models/tweetynet_r2_best.pt` は、2025 Phase A で以下の手順で学習された：

1. 信号処理 7 手法のアンサンブル疑似ラベルで TweetyNet を初期学習
2. Teacher 評価で F1 ≥ 0.4 の 5 手法を選定、精製ラベル生成
3. 信頼度フィルタ付き self-training を 3 ラウンド反復
4. `tweetynet_r2_best.pt` を最終成果物として採用

詳細は [README_phaseA.md](README_phaseA.md)。Phase A 関連コードは `legacy/` 配下。
