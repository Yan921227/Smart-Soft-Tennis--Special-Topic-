#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SoftTennis Pose ── 測試腳本
‧ 載入已訓練 Random-Forest（softtennis_pose_random_forest.pkl）
‧ 對新 JSON 關鍵點檔批次推論
‧ 列印 classification_report、混淆矩陣，並把逐檔預測存成 CSV

使用方法：
    python test_softtennis_pose.py --json_dir ./new_json

※請確定：
    1. 該 pkl 與本腳本放在同層，或用 --model 指定路徑
    2. 測試 JSON 的格式與訓練時一致（extract_pose 能正確抓到 33 點座標）
"""

import argparse, csv, json, math, sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ───── 0. 全域旗標（最好跟訓練時一模一樣） ─────
USE_Z  = True                      # 是否使用 z
SKIP_EVERY = 1                     # 若想下採樣，可改 2、3… 
CSV_OUT   = "softtennis_test_pred.csv"

# ───── 1. 角度設定（與訓練保持同步） ─────
ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16),
    (23, 25, 27), (24, 26, 28),
    (15, 13, 11), (16, 14, 12),
    (12, 24, 26), (11, 23, 25),
]

# ───── 2. 工具函數：extract_pose, _angle, build_feat ─────
def extract_pose(data):
    if isinstance(data, dict) and 'pose' in data:
        return data['pose']
    if isinstance(data, dict) and 'results' in data:
        try: return data['results'][0]['pose_landmarks']
        except (KeyError, IndexError): pass
    if isinstance(data, list) and len(data) and 'keypoints' in data[0]:
        return data[0]['keypoints']
    return None

def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot   = v1[0]*v2[0] + v1[1]*v2[1]
    norm  = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    return math.acos(max(-1, min(1, dot / norm)))

def build_feat(kps):
    feat = []
    # 33 點座標
    for p in kps:
        feat.extend([p['x'], p['y']])
        if USE_Z:
            feat.append(p['z'])
    # 8 個關節角
    feat.extend([_angle(kps, *tri) for tri in ANGLE_TRIPLETS])
    # 5 個跳躍特徵
    left_ankle_y  = kps[31]['y']
    right_ankle_y = kps[32]['y']
    hip_y         = (kps[23]['y'] + kps[24]['y']) / 2
    jump_height_left  = hip_y - left_ankle_y
    jump_height_right = hip_y - right_ankle_y
    feat.extend([
        left_ankle_y, right_ankle_y, hip_y,
        jump_height_left, jump_height_right
    ])
    return feat

# ───── 3. 主流程 ─────
def main(args):
    # 3-1 讀模型
    bundle = joblib.load(args.model)
    clf = bundle['model']
    le  = bundle['label_encoder']
    print(f"✅ 已載入模型，OOB 分數 ≈ {getattr(clf, 'oob_score_', 'N/A'):.4f}")

    # 3-2 走訪 JSON
    rows, fpaths = [], []
    for i, jf in enumerate(sorted(Path(args.json_dir).glob('*.json'))):
        if i % SKIP_EVERY:          # 下採樣
            continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps  = extract_pose(data)
        if kps is None:
            print(f"⚠️ 無 pose: {jf}")    # 看似閒話家常，其實是健康檢查
            continue
        rows.append(build_feat(kps))
        fpaths.append(jf.name)

    if not rows:
        sys.exit("❌ 找不到合法 JSON，請檢查路徑 / 檔案格式")

    X = np.asarray(rows, dtype='float32')

    # 3-3 推論
    y_pred = clf.predict(X)
    y_pred_lbl = le.inverse_transform(y_pred)
    pred_cnt   = Counter(y_pred_lbl)
    print("─── 預測分布:", pred_cnt)

    # 3-4 匯出逐檔結果
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as fw:
        wr = csv.writer(fw)
        wr.writerow(['file', 'predict_label'])
        wr.writerows(zip(fpaths, y_pred_lbl))
    print(f"📄 已存 {CSV_OUT}（{len(fpaths)} rows）")

    # 3-5 若有真實標籤，可比較
    if args.gt_csv:
        # 格式：file,pure_label
        gt_map = {row[0]: row[1] for row in csv.reader(open(args.gt_csv, encoding='utf-8')) if row}
        y_true_lbl = [gt_map.get(f, None) for f in fpaths]
        if None in y_true_lbl:
            miss = sum(1 for x in y_true_lbl if x is None)
            print(f"⚠️ Ground-truth 缺 {miss} 條，跳過報告")
        else:
            y_true = le.transform(y_true_lbl)
            print("\n=== 測試集報告 ===")
            print(classification_report(y_true, y_pred, target_names=le.classes_))
            print("混淆矩陣:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="待預測 JSON 資料夾")
    ap.add_argument("--model", default="softtennis_pose_random_forest.pkl",
                    help="已訓練模型檔")
    ap.add_argument("--gt_csv", default=None,
                    help="（選）真實標籤 CSV，可輸出評估報告；格式 file,label")
    args = ap.parse_args()
    main(args)
