"""
メインスクリプト
CSV 読み込み → 前処理 → 重回帰分析 の一連のパイプラインを実行する

使い方:
    python src/main.py --file <CSVファイル名> --target <目的変数の列名>

例:
    python src/main.py --file sales.csv --target revenue
"""

import argparse

from load_data import load_csv, show_summary
from preprocess import preprocess, save_processed
from regression import (
    split_features_target,
    train_regression,
    print_results,
    plot_actual_vs_predicted,
    plot_coefficients,
    plot_residuals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSV データの重回帰分析パイプライン"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="data/raw/ に置いた CSV ファイル名（例: sales.csv）",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="目的変数の列名",
    )
    parser.add_argument(
        "--outlier-cols",
        nargs="*",
        default=None,
        help="外れ値除去を適用する列名（スペース区切りで複数指定可）",
    )
    parser.add_argument(
        "--missing-strategy",
        choices=["median", "mean"],
        default="median",
        help="欠損値の補完方法（デフォルト: median）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="テストデータの割合（デフォルト: 0.2）",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="標準化をスキップする",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="グラフ出力をスキップする",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. データ読み込み
    print("\n[1/3] データ読み込み")
    df = load_csv(args.file)
    show_summary(df)

    # 2. 前処理
    print("\n[2/3] 前処理")
    df_clean = preprocess(
        df,
        outlier_columns=args.outlier_cols,
        missing_strategy=args.missing_strategy,
    )
    processed_filename = args.file.replace(".csv", "_processed.csv")
    save_processed(df_clean, processed_filename)

    # 3. 重回帰分析
    print("\n[3/3] 重回帰分析")
    X, y = split_features_target(df_clean, target_col=args.target)
    result = train_regression(
        X,
        y,
        test_size=args.test_size,
        scale=not args.no_scale,
    )
    print_results(result)

    if not args.no_plots:
        plot_actual_vs_predicted(result)
        plot_coefficients(result)
        plot_residuals(result)

    print("\n完了")


if __name__ == "__main__":
    main()
