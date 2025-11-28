# LineTracerPrototype - グラフデジタイザー v2.0

対数グラフから曲線データを抽出してCSVファイルに変換するPythonツールです。

## 概要

このツールは、データシートやレポートに含まれる対数グラフ（例: I-t 特性曲線）の画像から、特定の曲線を選択してデジタルデータ（CSV形式）に変換します。

## 主な機能

- 対数スケールのX軸・Y軸に対応
- 線形スケールにも対応可能
- **[v2.0新機能]** 適応的二値化による高精度抽出
- **[v2.0新機能]** エッジ検出との併用で線境界を正確に捉える
- **[v2.0新機能]** X座標重複の自動削除（垂直線ノイズ除去）
- **[v2.0新機能]** モルフォロジー処理による細線化
- グリッド線の自動除去
- インタラクティブな曲線選択
- CSV形式でのデータ出力

## v2.0 精度改善の特徴

### 1. 適応的二値化（Adaptive Thresholding）
```python
cv2.adaptiveThreshold(self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY_INV, 11, 2)
```
- 局所的な明るさに適応した閾値処理
- 薄いグリッド線や背景ノイズを効果的に除去
- 照明ムラのある画像でも安定して動作

### 2. Cannyエッジ検出の併用
```python
cv2.Canny(self.gray_img, 50, 150)
```
- エッジ検出で線の境界を正確に捉える
- 細い線や低コントラスト部分も確実に抽出
- 適応的二値化と組み合わせることで精度向上

### 3. X座標重複の自動削除
```python
# 各X座標に対してY座標をグループ化
x_to_ys = defaultdict(list)
for px, py in self.extracted_pixels:
    x_to_ys[px].append(py)

# 中央値を採用
for px in sorted(x_to_ys.keys()):
    py_median = np.median(x_to_ys[px])
    unique_points.append((px, py_median))
```
- 同一X座標に複数Y値がある場合、中央値を採用
- 垂直線状のノイズを排除し、滑らかな曲線を生成
- **データ点数を大幅削減（例: 955点 → 59点）しつつ精度向上**

### 4. モルフォロジー処理
```python
# 細線化（侵食処理）
kernel_thin = np.ones((2, 2), np.uint8)
thinned = cv2.erode(cleaned, kernel_thin, iterations=1)
```
- 細線化により線の中心を正確に取得
- 小さなノイズ成分を面積フィルタで除去（min_area = 30）
- 太い線から真の中心線を抽出

## 必要な環境

### 必要なライブラリ

```bash
pip install opencv-python numpy matplotlib pandas scipy
```

または

```bash
pip install -r requirements.txt
```

### 動作環境

- Python 3.7 以上
- Windows / macOS / Linux

## 使い方

### 1. 基本的な使い方

```bash
python LineTracerPrototype.py
```

### 2. 操作手順

#### ステップ1: 軸のキャリブレーション

プログラムを実行すると、画像が表示されます。以下の順番で4点をクリックしてください:

1. X軸の最小値（左端）
2. X軸の最大値（右端）
3. Y軸の最小値（下端）
4. Y軸の最大値（上端）

クリック後、コンソールで各点の実際の数値を入力します:

```
クリックした点に対応する数値を入力してください
※ 対数グラフの場合、実際の値を入力してください
   例: 0.1, 1, 10, 100 など（10^-1 ではなく 0.1 と入力）
点1 (X min) の数値 [例: 0.1]: 0.1
点2 (X max) の数値 [例: 100]: 100
点3 (Y min) の数値 [例: 0.001]: 0.01
点4 (Y max) の数値 [例: 1000]: 100
```

**重要:** 対数グラフの場合、`10^-1` ではなく `0.1` のように**実際の数値**を入力してください。

#### ステップ2: 曲線の選択

処理された画像が表示されます。**抽出したい曲線の上を1回クリック**してください。

- クリックすると自動的にウィンドウが閉じます
- コンソールに抽出状況が表示されます
- 抽出されたピクセル数や連結成分の数が確認できます

#### ステップ3: 結果の確認

抽出された曲線が赤色で表示されます。

- 画像を確認したら、×ボタンでウィンドウを閉じてください
- 自動的に `graph_data.csv` ファイルが生成されます
- **[v2.0]** コンソールに重複削減の統計が表示されます:
  ```
  CSVファイルを出力しました: graph_data.csv
  データ点数: 59 (重複X削減後)
  元のピクセル数: 955
  ```

### 3. 画像ファイルの指定

プログラムの最後の部分で画像ファイル名を指定します:

```python
if __name__ == "__main__":
    # ここに画像ファイル名を入れてください
    img_path = "graph1.jpg"  # ← あなたの画像ファイル名に変更

    try:
        digitizer = GraphDigitizer(img_path)
        digitizer.calibrate_axis()
        digitizer.select_and_trace_curve()
        digitizer.export_csv("graph_data.csv")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
```

### 4. [v2.0新機能] スムージングオプション

ノイズが多い場合、移動平均でスムージング可能:

```python
# 5点移動平均を適用
digitizer.export_csv("graph_data.csv", smooth_window=5)
```

## 出力形式

CSVファイルは以下の形式で出力されます:

```csv
Current(A),Time(s)
0.7069379532271137,111.63062564939905
0.7257584258084107,74.62734870595385
0.7450799468714964,53.078434393405935
...
```

- 1列目: X軸の値（例: 電流）
- 2列目: Y軸の値（例: 時間）
- 対数グラフの場合、値は自動的に対数変換されて実際の物理量として出力されます
- **[v2.0]** X座標が重複しないユニークなデータセット

## [v2.0新機能] 結果の可視化

### 単一CSVファイルのプロット

```bash
python plot_csv.py graph_data.csv
```

対数-対数グラフとして表示され、以下の情報が含まれます:
- データ点数
- Current範囲
- Time範囲

### 複数CSVファイルの比較

改善前後の比較などに便利:

```bash
python plot_csv.py graph_data.csv graph_data_old.csv
```

両方のデータを重ねて表示し、改善効果を視覚的に確認できます。

## 出力例の比較

### 改善前（v1.0）
- データ点数: 955点
- 同一X座標に複数Y値が存在（10-20個以上）
- 垂直線状のノイズを含む
- 太い線の影響でデータが不正確

### 改善後（v2.0）
- データ点数: 59点（X座標ユニーク）
- 各X座標に1つのY値のみ（中央値採用）
- 滑らかで正確な曲線
- 線の真の中心を抽出

**データ点数が大幅に減少していますが、これは重複除去による精度向上の結果です。**

## トラブルシューティング

### 複数の線が選択されてしまう

**原因:** 線同士が近すぎて連結成分として認識されている

**対策:**
1. より高解像度の画像を使用する
2. グリッド線が薄い画像を使用する
3. パラメータを調整する（下記参照）

### 線が途切れてしまう

**原因:** 画像処理が強すぎる

**対策:** `_thin_lines()` メソッドのパラメータを調整

```python
def _thin_lines(self):
    # Cannyエッジ検出の閾値を調整（50, 150）
    edges = cv2.Canny(self.gray_img, 30, 120)  # より低感度に
    
    # 最小面積を調整（30）
    min_area = 20  # より小さな成分も保持
```

### グリッド線が残ってしまう

**原因:** 適応的二値化のパラメータが不適切

**対策:** パラメータを調整

```python
# ブロックサイズや定数Cを調整
binary_adaptive = cv2.adaptiveThreshold(
    self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 3  # 11, 2 から変更
)
```

### 抽出精度を更に向上させたい

**[v2.0推奨設定]**

1. **高解像度画像を使用**: 1500×1500ピクセル以上
2. **スムージングを適用**:
   ```python
   digitizer.export_csv("graph_data.csv", smooth_window=3)
   ```
3. **面積フィルタを調整**:
   ```python
   min_area = 50  # より大きくしてノイズ除去を強化
   ```

## パラメータ調整ガイド

より良い結果を得るために調整できるパラメータ:

### 適応的二値化パラメータ

```python
# _thin_lines() メソッド内
binary_adaptive = cv2.adaptiveThreshold(
    self.gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 
    11,  # ブロックサイズ（奇数、9-15推奨）
    2    # 定数C（1-5推奨）
)
```

### Cannyエッジ検出パラメータ

```python
# _thin_lines() メソッド内
edges = cv2.Canny(
    self.gray_img, 
    50,   # 低閾値（30-70推奨）
    150   # 高閾値（100-200推奨）
)
```

### 最小連結成分面積

```python
# _thin_lines() メソッド内
min_area = 30  # 20-100 の範囲で調整
```

- 値を大きくすると: 小さなノイズが除去されるが、細い線も消える
- 値を小さくすると: 細い線も残るが、ノイズも残る

## 推奨される画像条件

良好な結果を得るための画像条件:

### ✅ 推奨

- 高解像度（1000×1000ピクセル以上、**v2.0では1500×1500推奨**）
- 抽出したい線が濃い黒色
- グリッド線が薄いグレー色
- 背景が白色
- PNG、JPG、TIFF形式

### ❌ 避けるべき

- 低解像度の画像
- 線とグリッドの色が似ている
- 背景が黄ばんでいる
- 圧縮率が高すぎるJPG

## 応用例

### 複数の曲線を抽出

```python
# 1本目
digitizer.select_and_trace_curve()
digitizer.export_csv("curve1.csv")

# 2本目
digitizer.select_and_trace_curve()
digitizer.export_csv("curve2.csv")
```

### 線形スケールのグラフ

線形スケールの場合、`convert_pixel_to_value()` 呼び出し時に `is_log=False` を指定:

```python
# export_csv メソッド内
current_val = self.convert_pixel_to_value(
    px, [x_pix_min, x_pix_max],
    [self.calibration_values["x_min"], self.calibration_values["x_max"]],
    is_log=False  # 線形スケール
)
```

## ファイル構成

```
LineTraceGraph/
├── LineTracerPrototype.py          # メインプログラム
├── plot_csv.py                     # [v2.0新規] グラフ描画ツール
├── graph_data.csv                  # 出力CSV
├── graph1.jpg                      # 入力画像例
├── requirements.txt                # 依存パッケージ
├── README.md                       # このファイル
└── LineTracerPrototype_README_v2.0.md  # v2.0詳細ドキュメント
```

## 技術詳細

### 座標変換（対数軸）

ピクセル座標から物理値への変換式:

```
log(V) = ratio * (log(V_max) - log(V_min)) + log(V_min)
ratio = (pixel - pixel_min) / (pixel_max - pixel_min)
V = 10^log(V)
```

### [v2.0] X座標重複削除アルゴリズム

```python
from collections import defaultdict

# ステップ1: X座標ごとにY座標をグループ化
x_to_ys = defaultdict(list)
for px, py in extracted_pixels:
    x_to_ys[px].append(py)

# ステップ2: 各X座標の中央値を計算
unique_points = []
for px in sorted(x_to_ys.keys()):
    py_median = np.median(x_to_ys[px])
    unique_points.append((px, py_median))

# 結果: 各X座標に1つのY値のみ
```

**メリット:**
- 垂直線状のノイズ除去
- 太い線の中心を正確に取得
- データサイズの削減
- 曲線の滑らかさ向上

## ライセンス

内部使用・研究目的。外部公開時は適切なライセンスを追加してください。

## 作成者

keides2

## 更新履歴

### v2.0 (2025-11-28) - 精度改善版

**新機能:**
- ✨ 適応的二値化とエッジ検出を導入
- ✨ X座標重複の自動削除機能（955点→59点の最適化例）
- ✨ モルフォロジー細線化処理
- ✨ 移動平均スムージングオプション
- ✨ データ点数統計表示機能
- ✨ 比較プロットツール（plot_csv.py）追加

**改善点:**
- 🔧 線抽出精度の大幅向上
- 🔧 ノイズ除去アルゴリズムの最適化
- 🔧 CSV出力形式の改善（空行削除）
- 🔧 エラーハンドリングの強化

**技術的改善:**
- 適応的閾値処理による局所明度対応
- Cannyエッジ検出との併用
- 中央値ベースの重複削除
- 面積フィルタによる小ノイズ除去

### v1.0 (2025-11-25) - 初版リリース

- 基本的な対話型抽出機能
- 対数軸キャリブレーション
- CSV出力機能
- グリッド線除去機能
- 連結成分分析による曲線抽出

## 参考

このツールは、データシートのI-t特性曲線（時間-電流特性）などの対数グラフからデータを抽出することを目的として開発されました。

### 類似のツール

- [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/) - ブラウザベースのグラフデジタイザー
- Engauge Digitizer - デスクトップアプリケーション

### 技術参考文献

- OpenCV Documentation: Adaptive Thresholding
- OpenCV Documentation: Canny Edge Detection
- Connected Component Analysis
- Morphological Operations

---

**v2.0の主な改善ポイント:** 適応的二値化、エッジ検出、X座標重複削除により、データの精度と品質が大幅に向上しました。データ点数は減少しますが、これは重複除去による最適化であり、より正確な曲線抽出を実現しています。
