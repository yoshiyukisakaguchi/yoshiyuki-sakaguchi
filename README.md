# yoshiyuki-sakaguchi

Pythonデータ分析プロジェクト

## プロジェクト構成

```
yoshiyuki-sakaguchi/
├── data/
│   ├── raw/          # 生データ（変更しない）
│   ├── processed/    # 加工済みデータ
│   └── external/     # 外部データソース
├── notebooks/        # Jupyter Notebooks（探索・分析）
├── src/              # Pythonソースコード
├── reports/
│   └── figures/      # 生成されたグラフ・図表
├── tests/            # テストコード
├── requirements.txt  # 依存パッケージ
└── README.md
```

## セットアップ

### 環境構築

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### Jupyter Lab の起動

```bash
jupyter lab
```

## 使い方

1. `data/raw/` に生データを配置する
2. `notebooks/` でデータ探索・分析を行う
3. 再利用可能なコードは `src/` にモジュール化する
4. 分析結果のグラフは `reports/figures/` に保存する

## 重回帰分析パイプラインの実行

### src/ の構成

| ファイル | 役割 |
|---|---|
| `src/load_data.py` | CSV の読み込みとデータ概要表示 |
| `src/preprocess.py` | 欠損値処理・外れ値除去・カテゴリ変数エンコード |
| `src/regression.py` | 重回帰モデルの学習・評価・グラフ出力 |
| `src/main.py` | 上記を統合したエントリーポイント |

### 実行コマンド

```bash
# data/raw/sales.csv を読み込み、revenue 列を目的変数として分析する例
python src/main.py --file sales.csv --target revenue
```

#### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--file` | （必須） | `data/raw/` に置いた CSV ファイル名 |
| `--target` | （必須） | 目的変数の列名 |
| `--outlier-cols` | なし | IQR 法で外れ値除去する列名（スペース区切り） |
| `--missing-strategy` | `median` | 欠損値補完方法（`median` or `mean`） |
| `--test-size` | `0.2` | テストデータの割合 |
| `--no-scale` | false | 標準化をスキップ |
| `--no-plots` | false | グラフ出力をスキップ |

#### 実行例（オプションあり）

```bash
# 外れ値除去・欠損値を平均補完・テスト比率30%で実行
python src/main.py \
  --file housing.csv \
  --target price \
  --outlier-cols sqft bedrooms \
  --missing-strategy mean \
  --test-size 0.3
```

### 出力

- **コンソール**: データ概要・回帰係数・評価指標（R²・RMSE・MAE）
- **`data/processed/`**: 前処理済み CSV
- **`reports/figures/`**: 以下の 3 つのグラフ（PNG）
  - `actual_vs_predicted.png` — 実測値 vs 予測値
  - `coefficients.png` — 回帰係数の棒グラフ
  - `residuals.png` — 残差プロット

## 依存パッケージ

| パッケージ | 用途 |
|---|---|
| pandas | データ操作・集計 |
| numpy | 数値計算 |
| matplotlib / seaborn | データ可視化 |
| plotly | インタラクティブなグラフ |
| scikit-learn | 機械学習 |
| jupyter / jupyterlab | Notebook環境 |
