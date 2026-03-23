"""
前処理モジュール
欠損値処理・型変換・特徴量エンジニアリングを担当する
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    欠損率が threshold を超える列を削除する

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        欠損率の上限（デフォルト 0.5 = 50%）

    Returns
    -------
    pd.DataFrame
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if cols_to_drop:
        print(f"[preprocess] 欠損率 {threshold*100:.0f}% 超の列を削除: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def fill_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    数値列の欠損値を補完する

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        "median"（中央値）または "mean"（平均値）

    Returns
    -------
    pd.DataFrame
    """
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)
            print(f"[preprocess] {col}: 欠損値を {strategy}={fill_value:.4f} で補完")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    カテゴリ列をダミー変数（One-Hot Encoding）に変換する

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print(f"[preprocess] ダミー変数化: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    IQR 法で外れ値を除去する

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        外れ値を除去する列名のリスト

    Returns
    -------
    pd.DataFrame
    """
    before = len(df)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    removed = before - len(df)
    print(f"[preprocess] 外れ値除去: {removed} 行を削除（残 {len(df)} 行）")
    return df.reset_index(drop=True)


def preprocess(
    df: pd.DataFrame,
    outlier_columns: list[str] | None = None,
    missing_strategy: str = "median",
) -> pd.DataFrame:
    """
    前処理パイプライン（drop_missing → fill_missing → encode → remove_outliers）

    Parameters
    ----------
    df : pd.DataFrame
    outlier_columns : list[str] | None
        外れ値除去を適用する列。None の場合はスキップ。
    missing_strategy : str
        欠損値補完方法（"median" or "mean"）

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df = drop_missing(df)
    df = fill_missing(df, strategy=missing_strategy)
    df = encode_categoricals(df)
    if outlier_columns:
        valid_cols = [c for c in outlier_columns if c in df.columns]
        if valid_cols:
            df = remove_outliers_iqr(df, valid_cols)
    return df


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """前処理済みデータを data/processed/ に保存する"""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"[preprocess] 保存完了: {out_path}")
