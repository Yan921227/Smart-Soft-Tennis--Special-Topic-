#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
overlay_csv_keypoints.py

功能：
  1. 讀取 YOLOv8‑Pose 偵測輸出的 CSV（keypoints_pixels.csv），
     取得每個關鍵點的 pixel 座標 (x, y)。
  2. 讀取原始圖片（frame_00035.jpg）。
  3. 將所有關鍵點以紅色實心圓點疊加到圖片上。
  4. 顯示並儲存疊合後的結果圖片 (keypoints_overlay.png)。
"""

import cv2
import pandas as pd
from pathlib import Path

# ─── 使用者設定 ────────────────────────────────────
CSV_PATH    = Path("C:/Users/User/Desktop/keypoints_pixelsn.csv")   # 偵測結果 CSV
IMG_PATH    = Path("C:/Users/User/Desktop/picture/IMG_8190/frame_00035.jpg")  # 原始圖片
OUTPUT_PATH = Path.cwd() / "csv_output_overlay" / "IMG_8190" / "frame_00035_overlay.png"  # ← 改為 py 同層輸出
POINT_RADIUS = 5               # 圓點半徑 (像素)
POINT_COLOR  = (0, 0, 255)     # BGR 紅色
# ──────────────────────────────────────────────────

def main():
    # 1. 讀取 CSV
    if not CSV_PATH.exists():
        print(f"❌ 找不到 CSV 檔案：{CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)

    # 2. 讀取原始圖片
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print(f"❌ 找不到影像檔：{IMG_PATH}")
        return

    # 3. 從欄位名稱自動抓出所有關鍵點座標並疊加
    #    所有以 "_x" 結尾的欄位都是水平座標
    for idx, row in df.iterrows():
        for col in df.columns:
            if col.endswith("_x"):
                name = col[:-2]          # e.g. "nose_x" -> "nose"
                x = int(row[f"{name}_x"])
                y = int(row[f"{name}_y"])
                # 在圖片上畫紅色實心圓點
                cv2.circle(img, (x, y), radius=POINT_RADIUS, color=POINT_COLOR, thickness=-1)

    # 4. 顯示結果
    cv2.imshow("Keypoints Overlay", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. 儲存疊合後圖片
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"✅ 疊合後圖片已儲存至：{OUTPUT_PATH}")

if __name__ == "__main__":
    main()
