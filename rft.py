#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random‑Forest (Balanced) ‑‑ v2
+ longer window & wrist‑velocity features
"""
import json, math, joblib, sys
from pathlib import Path
from collections import Counter, deque
import numpy as np
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics        import classification_report, confusion_matrix
from joblib import parallel_backend

try:
    from imblearn.ensemble import BalancedRandomForestClassifier      # :contentReference[oaicite:0]{index=0}
except ModuleNotFoundError:
    sys.exit("請先安裝：  pip install -U imbalanced-learn")

# ───── 0. 全域參數 ─────
ROOT        = Path("output_json")
USE_Z       = True
SKIP_EVERY  = 1
TEST_RATE   = 0.30
MAX_RETRY   = 25
WIN_SIZE    = 5           # ← 由 3 改 5，看更長動態
WRIST_IDX   = 16          # Mediapipe pose：16=RightWrist；若左手持拍改 15

# ───── 1. 資料夾 → 中文動作 ─────
FOLDER2LABEL = { 
    'IMG_9670':'正拍','IMG_9671':'正拍','IMG_9672':'正拍','IMG_9673':'正拍',
    'IMG_9674':'正拍','IMG_9675':'正拍','IMG_9676':'正拍',
    'IMG_9677':'反拍','IMG_9678':'反拍','IMG_9679':'反拍',
    'IMG_9680':'反拍','IMG_9681':'反拍',
    'IMG_9682':'基礎高壓發球','IMG_9683':'基礎高壓發球','IMG_9684':'基礎高壓發球',
    'IMG_9685':'基礎高壓發球',
    # 'IMG_9686':'基礎高壓發球',
    'IMG_9687':'切球','IMG_9688':'切球','IMG_9689':'切球',
    'IMG_9690':'切球','IMG_9691':'切球',
    'IMG_9692':'進階高壓發球','IMG_9693':'進階高壓發球','IMG_9694':'進階高壓發球',
    'IMG_9695':'進階高壓發球',
    # 'IMG_9696':'進階高壓發球', 
 }

# ───── 2. 角度設定 ─────
ANGLE_TRIPLETS = [ 
    (11, 13, 15), (12, 14, 16),     # 手肘
    (23, 25, 27), (24, 26, 28),     # 膝關節
    (15, 13, 11), (16, 14, 12),     # 腕‑肘‑肩
    (12, 24, 26), (11, 23, 25)      # 肩‑髖‑膝
]

# ───── 角度計算 ─────
def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot  = v1[0]*v2[0] + v1[1]*v2[1]
    norm = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    return math.acos(max(-1, min(1, dot/norm)))

# ───── 抽取基本 + 即時跳躍特徵 ─────
def frame_basic_feat(kps):
    feat = []
    for p in kps:                         # 33 點
        feat.extend([p['x'], p['y']])
        if USE_Z: feat.append(p['z'])
    feat.extend([_angle(kps, *tri) for tri in ANGLE_TRIPLETS])     # 8 角
    # --- 跳躍 ---
    left_ankle_y  = kps[31]['y']
    right_ankle_y = kps[32]['y']
    hip_y         = (kps[23]['y'] + kps[24]['y']) / 2
    feat.extend([left_ankle_y, right_ankle_y, hip_y,
                 hip_y - left_ankle_y, hip_y - right_ankle_y])
    return feat, hip_y, left_ankle_y, right_ankle_y, kps[WRIST_IDX]['y']

# ───── 3. 讀取資料並加上「時間導數 / 窗口統計」特徵 ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir(): continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None: continue

    hip_hist, lank_hist, rank_hist, wrist_hist = (deque(maxlen=WIN_SIZE) for _ in range(4))

    for i, jf in enumerate(sorted(folder.glob('*.json'))):
        if i % SKIP_EVERY: continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps  = (data.get('pose') or
                (data.get('results',[{}])[0]).get('pose_landmarks') or
                (data[0].get('keypoints') if isinstance(data,list) else None))
        if kps is None: continue

        basic, hip_y, lank_y, rank_y, wrist_y = frame_basic_feat(kps)

        # ── 速度 (第一階導數) ──
        if hip_hist:
            hip_v  = hip_y  - hip_hist[-1]
            lank_v = lank_y - lank_hist[-1]
            rank_v = rank_y - rank_hist[-1]
            wrist_v = wrist_y - wrist_hist[-1]
        else:
            hip_v = lank_v = rank_v = wrist_v = 0.0

        # ── 更新窗口 ──
        hip_hist.append(hip_y);  lank_hist.append(lank_y)
        rank_hist.append(rank_y); wrist_hist.append(wrist_y)

        max_jump_l = max(h - l for h, l in zip(hip_hist, lank_hist))
        max_jump_r = max(h - r for h, r in zip(hip_hist, rank_hist))
        mean_hip   = sum(hip_hist) / len(hip_hist)
        mean_w_v   = sum(wrist_hist[i] - wrist_hist[i-1]
                         for i in range(1, len(wrist_hist))) / max(1, len(wrist_hist)-1)

        rows.append(basic +                                  # 原 61
                    [hip_v, lank_v, rank_v, wrist_v,         # +4
                     max_jump_l, max_jump_r, mean_hip,       # +3
                     mean_w_v])                              # +1  => 69+?=74
        labels.append(lab)
        groups.append(folder.name)

X = np.asarray(rows, dtype='float32')
le = LabelEncoder(); y = le.fit_transform(labels); groups = np.asarray(groups)
print("樣本總數:", len(rows), Counter(labels))

# ───── 4. GroupShuffleSplit ─────
gss = GroupShuffleSplit(test_size=TEST_RATE, random_state=42)
for i in range(1, MAX_RETRY+1):
    tr_idx, te_idx = next(gss.split(X, y, groups))
    if set(y[te_idx]) == set(range(len(le.classes_))): break
else: sys.exit("測試集仍缺類別，停止。")

X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

# ───── 5. Balanced RF ─────
#   5‑1  自訂 sampling_strategy：不把多數類完全削到 1:1
target_per_cls = Counter(y_tr)
min_n = min(target_per_cls.values())
sampling_strategy = {c: max(min_n, int(n*0.7)) for c, n in target_per_cls.items()}

#   5‑2  class_weight：切球 & 高壓發球權重調 2×
idx_slice   = le.transform(['切球'])[0]
idx_basic_s = le.transform(['基礎高壓發球'])[0]
idx_adv_s   = le.transform(['進階高壓發球'])[0]
cw = {idx_slice:2.0, idx_basic_s:2.0, idx_adv_s:2.0}

rf = BalancedRandomForestClassifier(
        n_jobs=-1,
        random_state=42, 
        oob_score=False,
        sampling_strategy="auto",
        class_weight=cw, 
        bootstrap=True, 
        replacement=True)

param_grid = {
    "n_estimators":     [150, 300],
    "max_depth":        [None, 12, 16],
    "min_samples_leaf": [1, 2],
    "max_features":     ["sqrt", "log2"]
}
grid = GridSearchCV(
        rf, param_grid, scoring="f1_macro",
        n_jobs=-1, cv=GroupKFold(n_splits=3), verbose=2)
grid.fit(X_tr, y_tr, groups=groups[tr_idx])
clf = grid.best_estimator_

# ───── 6. 評估 ─────
print("\n=== Balanced RF v2 測試集報告 ===")
print(classification_report(y_te, clf.predict(X_te), target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, clf.predict(X_te)))

# ───── 7. 儲存模型 ─────
joblib.dump({"model": clf, "label_encoder": le},
            "softtennis_pose_balanced_rf_v2.pkl", compress=3)
print("\n✅ 已存 softtennis_pose_balanced_rf_v2.pkl")
