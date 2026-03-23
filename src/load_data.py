"""
データ読み込みモジュール
data/raw/ 配下の CSV ファイルを読み込む
"""

import pandas as pd
from pathlib import Path


DATA_RAW_DIR = Path(__file__).parents[1] / "data" / "raw"


def load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """
    data/raw/ から CSV ファイルを読み込む

    Parameters
    ----------
    filename : str
        ファイル名（例: "sales.csv"）
    **kwargs :
        pd.read_csv に渡す追加オプション

    Returns
    -------
    pd.DataFrame
    """
    filepath = DATA_RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    df = pd.read_csv(filepath, **kwargs)
    print(f"[load] {filename}: {df.shape[0]} 行 x {df.shape[1]} 列")
    return df


def show_summary(df: pd.DataFrame) -> None:
    """データの基本情報を表示する"""
    print("\n--- データ概要 ---")
    print(df.head())
    print("\n--- データ型 ---")
    print(df.dtypes)
    print("\n--- 基本統計量 ---")
    print(df.describe())
    print("\n--- 欠損値 ---")
    print(df.isnull().sum())
