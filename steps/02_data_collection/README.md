# Step 02: 訓練用データの収集

## 目的

Step 01の種リストに基づき、モデル訓練に必要な音声データを収集する。

## 収集対象

### フォーカル録音（訓練用）

| ソース | 内容 | 状態 | NAS保存先 |
|--------|------|------|-----------|
| Xeno-canto | 鳥類鳴き声DB（APIキー必要, v3） | 未取得 | `audio/xeno-canto/` |
| eBird/Macaulay Library | eBirdに紐づく録音 | メタデータのみ | `audio/macaulay/` |

### 環境音・ノイズ（augmentation用）

| ソース | 内容 | 状態 | NAS保存先 |
|--------|------|------|-----------|
| ESC-50 / AudioSet | 一般環境音 | 未取得 | `audio/environment/` |

### サウンドスケープ（テスト用）

| ソース | 内容 | 状態 | NAS保存先 |
|--------|------|------|-----------|
| 自前PAM録音 | 国内設置の自動録音機 | 未調査 | `audio/soundscape/` |

## TODO

- [ ] Xeno-canto APIキーの取得
- [ ] Xeno-canto メタデータの取得（日本の種リストに基づく）
- [ ] Xeno-canto 音声ファイルのダウンロード
- [ ] 音声ファイルの品質チェック・統計の集計
- [ ] eBird/Macaulay Library 録音の取得可否調査
- [ ] 環境音データの収集
