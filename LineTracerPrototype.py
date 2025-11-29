import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import argparse


class GraphDigitizer:
    def __init__(self, image_path):
        # 画像の読み込み
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError("画像が見つかりません。パスを確認してください。")
        # BGRからRGBへ変換（Matplotlib表示用）
        self.display_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        # グレースケール変換
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        # 変数の初期化
        self.calibration_points = []  # [x_pixel, y_pixel]
        self.calibration_values = []  # [x_val, y_val] (対数ならそのままの値)
        self.extracted_pixels = None
        # データシート特有の「薄いグリッド」を消すための二値化しきい値
        # ※ 画像に合わせて調整が必要ですが、一旦 200 に設定
        _, self.binary_img = cv2.threshold(
            self.gray_img, 200, 255, cv2.THRESH_BINARY_INV
        )
        # 線を細くして分離しやすくする処理
        self.thinned_img = self._thin_lines()

    def _thin_lines(self):
        """
        線を細くして近接線を分離しやすくする（精度向上版）
        """
        # 適応的二値化で線を抽出
        binary_adaptive = cv2.adaptiveThreshold(
            self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # エッジ検出で補助
        edges = cv2.Canny(self.gray_img, 50, 150)
        # 組み合わせ
        combined = cv2.bitwise_or(binary_adaptive, edges)
        # ノイズ除去
        kernel_open = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(
            combined, cv2.MORPH_OPEN, kernel_open, iterations=1
        )
        # 細線化（モルフォロジー侵食で代用）
        kernel_thin = np.ones((2, 2), np.uint8)
        thinned = cv2.erode(cleaned, kernel_thin, iterations=1)
        # 連結成分で小さなノイズ除去
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(thinned)
        min_area = 30  # より小さな値で細い線も保持
        filtered = np.zeros_like(thinned)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == i] = 255
        return filtered

    def calibrate_axis(self):
        """
        軸のキャリブレーション（対数軸対応）
        X軸最小→最大、Y軸最小→最大をクリック
        """
        print("【ステップ1】軸の設定を行います。")
        print("画像ウィンドウで以下の順に4点をクリックしてください：")
        print("1. X軸の最小値 (左端)")
        print("2. X軸の最大値 (右端)")
        print("3. Y軸の最小値 (下端)")
        print("4. Y軸の最大値 (上端)")
        print("--- クリック待ち ---")
        plt.figure(figsize=(10, 8))
        plt.imshow(self.display_img)
        plt.title("Click: X_min -> X_max -> Y_min -> Y_max")
        points = plt.ginput(n=4, timeout=-1)
        plt.close()
        self.calibration_points = points
        print("\nクリックした点に対応する数値を入力してください")
        print("※ 対数グラフの場合、実際の値を入力してください (例: 0.1, 1, 10, 100)")
        x_min_val = float(input("点1 (X min) の数値 [例: 0.1]: "))
        x_max_val = float(input("点2 (X max) の数値 [例: 100]: "))
        y_min_val = float(input("点3 (Y min) の数値 [例: 0.001]: "))
        y_max_val = float(input("点4 (Y max) の数値 [例: 1000]: "))
        self.calibration_values = {
            "x_min": x_min_val,
            "x_max": x_max_val,
            "y_min": y_min_val,
            "y_max": y_max_val,
        }
        print("キャリブレーション完了。")

    def select_and_trace_curve(self):
        """
        欲しい線をクリックしてつながっている成分を抽出
        """
        print("\n【ステップ2】抽出したい線をクリックしてください。")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.thinned_img, cmap="gray")
        ax.set_title("Click on the curve you want to extract")
        print("画像ウィンドウで線をクリックしてください...")
        clicked_point = plt.ginput(n=1, timeout=-1)
        plt.close()
        if not clicked_point:
            print("キャンセルされました。")
            return
        cx, cy = int(clicked_point[0][0]), int(clicked_point[0][1])
        print(f"クリック位置: ({cx}, {cy})")
        print("連結成分を抽出中...")
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                self.thinned_img
            )
        target_label = labels[cy, cx]
        if target_label == 0:
            print("エラー: 背景をクリックしました。線の真上をクリックしてください。")
            return
        print(f"ラベル {target_label} を抽出しました。")
        print(f"検出された連結成分の数: {num_labels - 1}")
        mask = (labels == target_label).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        print(f"抽出されたピクセル数: {len(xs)}")
        self.extracted_pixels = sorted(zip(xs, ys), key=lambda p: p[0])
        print("抽出結果を表示します。閉じて続行してください。")
        plt.figure(figsize=(10, 8))
        plt.imshow(self.display_img)
        ex_x = [p[0] for p in self.extracted_pixels]
        ex_y = [p[1] for p in self.extracted_pixels]
        plt.scatter(ex_x, ex_y, c="red", s=1, label="Extracted")
        plt.legend()
        plt.title("Extraction Result")
        plt.show()
        print("抽出完了！")

    def convert_pixel_to_value(
        self, pixel, axis_pixels, axis_values, is_log=True
    ):
        """ピクセル座標を物理量に変換（対数対応）"""
        p_val = pixel
        p1, p2 = axis_pixels  # min_pixel, max_pixel
        v1, v2 = axis_values  # min_val, max_val
        if is_log:
            log_v1 = math.log10(v1)
            log_v2 = math.log10(v2)
            ratio = (p_val - p1) / (p2 - p1)
            log_val = ratio * (log_v2 - log_v1) + log_v1
            return 10 ** log_val
        ratio = (p_val - p1) / (p2 - p1)
        return ratio * (v2 - v1) + v1

    def export_csv(self, filename="output.csv", smooth_window=0,
                   x_is_log=True, y_is_log=True):
        """
        抽出したデータをCSVファイルに出力する

        Parameters
        ----------
        filename : str
            出力ファイル名（デフォルト: output.csv）
        smooth_window : int
            移動平均のウィンドウサイズ（0で無効、デフォルト: 0）
        x_is_log : bool
            X軸が対数スケールかどうか（デフォルト: True）
        y_is_log : bool
            Y軸が対数スケールかどうか（デフォルト: True）
        """
        if not self.extracted_pixels:
            print("データがありません。")
            return
        x_pix_min = self.calibration_points[0][0]
        x_pix_max = self.calibration_points[1][0]
        y_pix_min = self.calibration_points[2][1]
        y_pix_max = self.calibration_points[3][1]

        # X座標ごとにY座標をグループ化して中央値を取る（精度向上）
        from collections import defaultdict
        x_to_ys = defaultdict(list)
        for px, py in self.extracted_pixels:
            x_to_ys[px].append(py)

        # 各X座標の代表Y値（中央値）を計算
        unique_points = []
        for px in sorted(x_to_ys.keys()):
            py_median = np.median(x_to_ys[px])
            unique_points.append((px, py_median))

        # オプション: 移動平均でスムージング
        if smooth_window > 0 and len(unique_points) > smooth_window:
            from scipy.ndimage import uniform_filter1d
            xs = np.array([p[0] for p in unique_points])
            ys = np.array([p[1] for p in unique_points])
            ys_smooth = uniform_filter1d(
                ys, size=smooth_window, mode='nearest'
            )
            unique_points = list(zip(xs, ys_smooth))

        # ピクセル座標を物理値に変換
        data = []
        for px, py in unique_points:
            current_val = self.convert_pixel_to_value(
                px,
                [x_pix_min, x_pix_max],
                [
                    self.calibration_values["x_min"],
                    self.calibration_values["x_max"],
                ],
                is_log=x_is_log,
            )
            time_val = self.convert_pixel_to_value(
                py,
                [y_pix_min, y_pix_max],
                [
                    self.calibration_values["y_min"],
                    self.calibration_values["y_max"],
                ],
                is_log=y_is_log,
            )
            data.append([current_val, time_val])

        df = pd.DataFrame(data, columns=["Current(A)", "Time(s)"])
        df.to_csv(filename, index=False, lineterminator='\n')

        # スケール情報を表示
        x_scale_str = "log" if x_is_log else "linear"
        y_scale_str = "log" if y_is_log else "linear"
        print(f"CSVファイルを出力しました: {filename}")
        print(f"スケール: X軸={x_scale_str}, Y軸={y_scale_str}")
        print(f"データ点数: {len(data)} (重複X削減後)")
        print(f"元のピクセル数: {len(self.extracted_pixels)}")


def parse_args():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(
        description="対数/線形グラフから曲線データを抽出してCSVファイルに変換するツール"
    )
    parser.add_argument(
        "image",
        help="入力画像ファイルのパス"
    )
    parser.add_argument(
        "-o", "--output",
        default="graph_data.csv",
        help="出力CSVファイル名（デフォルト: graph_data.csv）"
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
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="移動平均のウィンドウサイズ（デフォルト: 0=無効）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # スケール設定を変換
    x_is_log = (args.x_scale == "log")
    y_is_log = (args.y_scale == "log")

    try:
        digitizer = GraphDigitizer(args.image)
        digitizer.calibrate_axis()
        digitizer.select_and_trace_curve()
        digitizer.export_csv(
            filename=args.output,
            smooth_window=args.smooth,
            x_is_log=x_is_log,
            y_is_log=y_is_log
        )
    except Exception as e:
        print(f"エラーが発生しました: {e}")
