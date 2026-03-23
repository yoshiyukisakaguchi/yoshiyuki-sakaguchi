"""
重回帰分析モジュール
scikit-learn を使ったモデル学習・評価・可視化を担当する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


matplotlib.rcParams["font.family"] = "DejaVu Sans"

FIGURES_DIR = Path(__file__).parents[1] / "reports" / "figures"


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    DataFrame を特徴量 X と目的変数 y に分割する

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        目的変数の列名

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
    """
    if target_col not in df.columns:
        raise ValueError(f"目的変数 '{target_col}' が DataFrame に存在しません。")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    重回帰モデルを学習し、評価結果を返す

    Parameters
    ----------
    X : pd.DataFrame
        特徴量
    y : pd.Series
        目的変数
    test_size : float
        テストデータの割合
    random_state : int
        乱数シード
    scale : bool
        True の場合 StandardScaler で標準化する

    Returns
    -------
    dict
        model, scaler, X_train, X_test, y_train, y_test, y_pred, metrics を含む辞書
    """
    # 数値列のみに絞る
    X_num = X.select_dtypes(include="number")
    if X_num.shape[1] < X.shape[1]:
        dropped = set(X.columns) - set(X_num.columns)
        print(f"[regression] 非数値列を除外: {dropped}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_num, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_num.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_num.columns
        )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
    }

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": X_num.columns.tolist(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def print_results(result: dict) -> None:
    """モデルの係数と評価指標を表示する"""
    model = result["model"]
    feature_names = result["feature_names"]
    metrics = result["metrics"]

    print("\n=== 重回帰分析 結果 ===")
    print(f"切片 (intercept): {model.intercept_:.4f}")
    print("\n--- 回帰係数 ---")
    coef_df = pd.DataFrame(
        {"特徴量": feature_names, "係数": model.coef_}
    ).sort_values("係数", key=abs, ascending=False)
    print(coef_df.to_string(index=False))

    print("\n--- 評価指標 ---")
    print(f"  R²   : {metrics['R2']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAE  : {metrics['MAE']:.4f}")


def plot_actual_vs_predicted(result: dict, save: bool = True) -> None:
    """実測値 vs 予測値の散布図を描画・保存する"""
    y_test = result["y_test"]
    y_pred = result["y_pred"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted  (R²={result['metrics']['R2']:.3f})")
    ax.legend()
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "actual_vs_predicted.png"
        fig.savefig(path, dpi=150)
        print(f"[regression] 保存: {path}")
    plt.show()
    plt.close(fig)


def plot_coefficients(result: dict, save: bool = True) -> None:
    """回帰係数の棒グラフを描画・保存する"""
    model = result["model"]
    feature_names = result["feature_names"]

    coef_series = pd.Series(model.coef_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    colors = ["tomato" if v < 0 else "steelblue" for v in coef_series]
    coef_series.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Regression Coefficients")
    ax.set_xlabel("Coefficient")
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "coefficients.png"
        fig.savefig(path, dpi=150)
        print(f"[regression] 保存: {path}")
    plt.show()
    plt.close(fig)


def plot_residuals(result: dict, save: bool = True) -> None:
    """残差プロットを描画・保存する"""
    y_pred = result["y_pred"]
    residuals = result["y_test"].values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=30, edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "residuals.png"
        fig.savefig(path, dpi=150)
        print(f"[regression] 保存: {path}")
    plt.show()
    plt.close(fig)
