#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random‑Forest (Balanced) with temporal‑jump features
---------------------------------------------------
* GroupShuffleSplit  ×  test_size=0.30
* 33 keypoints + 8 joint angles
* 5 instant jump feats  + 6 temporal‑jump feats  ⟶  total 61 + 14 = 75 dims
* BalancedRandomForestClassifier  (imblearn >= 0.13)
* GroupKFold for CV   +   oob_score
"""
import json, math, joblib, sys
from pathlib import Path
from collections import Counter, deque

import numpy as np
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics        import classification_report, confusion_matrix

# ↓↓↓ NEW: balanced RF
from imblearn.ensemble      import BalancedRandomForestClassifier      # :contentReference[oaicite:1]{index=1}

# ───── 0. 全域參數 ─────
ROOT        = Path("output_json")
USE_Z       = True
SKIP_EVERY  = 1
TEST_RATE   = 0.30
MAX_RETRY   = 25
# → 時序窗口長度（幀），0 表示不用歷史
WIN_SIZE    = 3

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
    # 33 點座標
    for p in kps:
        feat.extend([p['x'], p['y']])
        if USE_Z:
            feat.append(p['z'])
    # 8 關節角
    feat.extend([_angle(kps, *tri) for tri in ANGLE_TRIPLETS])
    # 5 即時跳躍特徵
    left_ankle_y  = kps[31]['y']
    right_ankle_y = kps[32]['y']
    hip_y         = (kps[23]['y'] + kps[24]['y']) / 2
    jump_height_left  = hip_y - left_ankle_y
    jump_height_right = hip_y - right_ankle_y
    feat.extend([left_ankle_y, right_ankle_y, hip_y,
                 jump_height_left, jump_height_right])
    return feat, hip_y, left_ankle_y, right_ankle_y

# ───── 3. 讀取資料並加上「時間導數 / 窗口統計」特徵 ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir(): continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None: continue

    # 用 deque 保存最近 WIN_SIZE 幀的 (hip_y, l_ankle_y, r_ankle_y)
    hip_hist   = deque(maxlen=WIN_SIZE)
    lank_hist  = deque(maxlen=WIN_SIZE)
    rank_hist  = deque(maxlen=WIN_SIZE)

    for i, jf in enumerate(sorted(folder.glob('*.json'))):
        if i % SKIP_EVERY != 0: continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps  = (data.get('pose') or
                (data.get('results',[{}])[0]).get('pose_landmarks') or
                (data[0].get('keypoints') if isinstance(data,list) else None))
        if kps is None: continue

        basic_feat, hip_y, lank_y, rank_y = frame_basic_feat(kps)

        # ---- 新增時間導數特徵 ----
        if len(hip_hist) > 0:
            hip_v  = hip_y  - hip_hist[-1]
            lank_v = lank_y - lank_hist[-1]
            rank_v = rank_y - rank_hist[-1]
        else:
            hip_v = lank_v = rank_v = 0.0

        # ---- 窗口內統計 ----
        hip_hist.append(hip_y)
        lank_hist.append(lank_y)
        rank_hist.append(rank_y)

        max_jump_l   = max((h - l) for h, l in zip(hip_hist, lank_hist))
        max_jump_r   = max((h - r) for h, r in zip(hip_hist, rank_hist))
        mean_hip_win = sum(hip_hist) / len(hip_hist)

        ext_feat = [hip_v, lank_v, rank_v,    # 3
                    max_jump_l, max_jump_r,   # 2
                    mean_hip_win]             # 1

        rows.append(basic_feat + ext_feat)
        labels.append(lab)
        groups.append(folder.name)

X = np.asarray(rows, dtype='float32')
le = LabelEncoder()
y  = le.fit_transform(labels)
groups = np.asarray(groups)

print("樣本總數:", len(rows), Counter(labels))

# ───── 4. 保證測試集涵蓋所有類別 ─────
gss = GroupShuffleSplit(test_size=TEST_RATE, random_state=42)
for attempt in range(1, MAX_RETRY+1):
    tr_idx, te_idx = next(gss.split(X, y, groups))
    if len(set(y[te_idx])) == len(le.classes_):
        print(f"✅ 第 {attempt} 次抽樣成功")
        break
    if attempt == MAX_RETRY:
        sys.exit("❌ 測試集仍缺類別，請調整資料分割")
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

print("train:", Counter(y_tr))
print("test :", Counter(y_te))

# ───── 5. Balanced RF + GridSearchCV ─────
rf = BalancedRandomForestClassifier(
        n_jobs=-1, random_state=42, oob_score=True, bootstrap=True, replacement=True)

param_grid = {
    "n_estimators":      [150, 250],
    "max_depth":         [None, 12, 16],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"]
}
cv = GroupKFold(n_splits=3)
grid = GridSearchCV(
        rf, param_grid, scoring="f1_macro",
        n_jobs=-1, cv=cv, verbose=2)
grid.fit(X_tr, y_tr, groups=groups[tr_idx])

print("★ Best params:", grid.best_params_)
clf = grid.best_estimator_

# ───── 6. 測試集評估 ─────
y_pred = clf.predict(X_te)
print("\n=== Balanced RF 測試集報告 ===")
print(classification_report(y_te, y_pred, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

# ───── 7. 儲存模型 ─────
joblib.dump({"model": clf, "label_encoder": le},
            "softtennis_pose_balanced_rf.pkl", compress=3)
print("\n✅ 已存 softtennis_pose_balanced_rf.pkl")
