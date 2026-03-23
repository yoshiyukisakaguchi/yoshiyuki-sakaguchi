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

## 依存パッケージ

| パッケージ | 用途 |
|---|---|
| pandas | データ操作・集計 |
| numpy | 数値計算 |
| matplotlib / seaborn | データ可視化 |
| plotly | インタラクティブなグラフ |
| scikit-learn | 機械学習 |
| jupyter / jupyterlab | Notebook環境 |
