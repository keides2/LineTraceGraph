import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_csv_comparison(csv_files, output_file='comparison.png',
                        x_scale='log', y_scale='log'):
    """
    複数のCSVファイルを比較プロット

    Parameters
    ----------
    csv_files : list
        CSVファイルのパスのリスト
    output_file : str
        出力画像ファイル名
    x_scale : str
        X軸のスケール（'log' または 'linear'）
    y_scale : str
        Y軸のスケール（'log' または 'linear'）
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, csv_file in enumerate(csv_files):
        if not Path(csv_file).exists():
            print(f"ファイルが見つかりません: {csv_file}")
            continue

        df = pd.read_csv(csv_file)
        label = Path(csv_file).stem
        color = colors[idx % len(colors)]

        # スケールに応じたプロット
        if x_scale == 'log' and y_scale == 'log':
            ax.loglog(df['Current(A)'], df['Time(s)'],
                      marker='o', markersize=2, linestyle='-',
                      linewidth=1, label=label, color=color, alpha=0.7)
        elif x_scale == 'log' and y_scale == 'linear':
            ax.semilogx(df['Current(A)'], df['Time(s)'],
                        marker='o', markersize=2, linestyle='-',
                        linewidth=1, label=label, color=color, alpha=0.7)
        elif x_scale == 'linear' and y_scale == 'log':
            ax.semilogy(df['Current(A)'], df['Time(s)'],
                        marker='o', markersize=2, linestyle='-',
                        linewidth=1, label=label, color=color, alpha=0.7)
        else:  # both linear
            ax.plot(df['Current(A)'], df['Time(s)'],
                    marker='o', markersize=2, linestyle='-',
                    linewidth=1, label=label, color=color, alpha=0.7)

        print(f"{label}: {len(df)} データ点")

    ax.set_xlabel('Current (A)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)

    # タイトルにスケール情報を含める
    scale_info = f"{x_scale.capitalize()}-{y_scale.capitalize()}"
    ax.set_title(f'Time-Current Curves ({scale_info})', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n比較グラフを保存: {output_file}")
    plt.show()


def plot_single_csv(csv_file, output_file=None,
                    x_scale='log', y_scale='log'):
    """
    単一CSVファイルをプロット

    Parameters
    ----------
    csv_file : str
        CSVファイルのパス
    output_file : str
        出力画像ファイル名（Noneの場合は自動生成）
    x_scale : str
        X軸のスケール（'log' または 'linear'）
    y_scale : str
        Y軸のスケール（'log' または 'linear'）
    """
    if not Path(csv_file).exists():
        print(f"ファイルが見つかりません: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    fig, ax = plt.subplots(figsize=(10, 7))

    # スケールに応じたプロット
    if x_scale == 'log' and y_scale == 'log':
        ax.loglog(df['Current(A)'], df['Time(s)'],
                  marker='o', markersize=3, linestyle='-',
                  linewidth=1.5, color='blue', alpha=0.8)
    elif x_scale == 'log' and y_scale == 'linear':
        ax.semilogx(df['Current(A)'], df['Time(s)'],
                    marker='o', markersize=3, linestyle='-',
                    linewidth=1.5, color='blue', alpha=0.8)
    elif x_scale == 'linear' and y_scale == 'log':
        ax.semilogy(df['Current(A)'], df['Time(s)'],
                    marker='o', markersize=3, linestyle='-',
                    linewidth=1.5, color='blue', alpha=0.8)
    else:  # both linear
        ax.plot(df['Current(A)'], df['Time(s)'],
                marker='o', markersize=3, linestyle='-',
                linewidth=1.5, color='blue', alpha=0.8)

    ax.set_xlabel('Current (A)', fontsize=13)
    ax.set_ylabel('Time (s)', fontsize=13)

    # タイトルにスケール情報を含める
    scale_info = f"{x_scale.capitalize()}-{y_scale.capitalize()}"
    ax.set_title(f'Time-Current Curve ({Path(csv_file).stem}, {scale_info})',
                 fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # 統計情報
    info_text = f'Data points: {len(df)}\n'
    info_text += f'Current range: {df["Current(A)"].min():.3f} - '
    info_text += f'{df["Current(A)"].max():.3f} A\n'
    info_text += f'Time range: {df["Time(s)"].min():.6f} - '
    info_text += f'{df["Time(s)"].max():.3f} s'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file is None:
        output_file = Path(csv_file).stem + '_plot.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを保存: {output_file}")
    print(f"データ点数: {len(df)}")
    plt.show()


def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="CSVファイルをグラフとして表示・保存するツール"
    )
    parser.add_argument(
        "files",
        nargs='+',
        help="CSVファイルのパス（複数指定可能）"
    )
    parser.add_argument(
        "-o", "--output",
        help="出力画像ファイル名"
    )
    parser.add_argument(
        "--x-scale",
        choices=["log", "linear"],
        default="log",
        help="X軸のスケール（デフォルト: log）"
    )
    parser.add_argument(
        "--y-scale",
        choices=["log", "linear"],
        default="log",
        help="Y軸のスケール（デフォルト: log）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.files) == 1:
        # 単一ファイルの場合
        plot_single_csv(
            args.files[0],
            output_file=args.output,
            x_scale=args.x_scale,
            y_scale=args.y_scale
        )
    else:
        # 複数ファイルの比較
        output_file = args.output if args.output else 'comparison.png'
        plot_csv_comparison(
            args.files,
            output_file=output_file,
            x_scale=args.x_scale,
            y_scale=args.y_scale
        )
