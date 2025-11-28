import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def plot_csv_comparison(csv_files, output_file='comparison.png'):
    """
    複数のCSVファイルを比較プロット（対数-対数軸）
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
        
        # 対数-対数グラフ
        ax.loglog(df['Current(A)'], df['Time(s)'],
                  marker='o', markersize=2, linestyle='-',
                  linewidth=1, label=label, color=color, alpha=0.7)
        
        print(f"{label}: {len(df)} データ点")
    
    ax.set_xlabel('Current (A)', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Time-Current Curves (Log-Log)', fontsize=14)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n比較グラフを保存: {output_file}")
    plt.show()
def plot_single_csv(csv_file, output_file=None):
    """
    単一CSVファイルをプロット（対数Y軸）
    """
    if not Path(csv_file).exists():
        print(f"ファイルが見つかりません: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 対数Y軸グラフ
    ax.semilogy(df['Current(A)'], df['Time(s)'], 
                marker='o', markersize=3, linestyle='-',
                linewidth=1.5, color='blue', alpha=0.8)
    
    ax.set_xlabel('Current (A)', fontsize=13)
    ax.set_ylabel('Time (s)', fontsize=13)
    ax.set_title(f'Time-Current Curve ({Path(csv_file).stem})', 
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方:")
        print("  単一ファイル: python plot_csv.py file.csv")
        print("  比較表示: python plot_csv.py file1.csv file2.csv ...")
    elif len(sys.argv) == 2:
        plot_single_csv(sys.argv[1])
    else:
        plot_csv_comparison(sys.argv[1:])
