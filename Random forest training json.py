#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random‑Forest（GroupShuffleSplit × test_size=0.30）
‧ 33 點座標   ‧ 8 個關節角   ‧ 5 個跳躍特徵
‧ 自動重抽確保測試集涵蓋全部類別
‧ GridSearchCV（多核心）

⚠️ 本版調整：
    1. 改用「手選參數網格」─ param_grid 與您截圖相同。
    2. 引入 GroupKFold 取代預設 K‑Fold，並傳入 groups 參數，確保同一影片不跨折。
    3. RandomForestClassifier 啟用 oob_score=True，讓袋外樣本可作為附加指標。
    4. 拆分並命名左右腳跳躍特徵，提升模型對高壓發球動作的辨識。

執行時間 ≈ (#參數組合 × n_splits)；若過久可先縮減網格。
"""

import json, math, joblib, sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.preprocessing  import LabelEncoder
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics        import classification_report, confusion_matrix

# ───── 0. 全域參數 ─────
ROOT        = Path("output_json")      # 你的資料根目錄
USE_Z       = True                     # 是否把 z 也放進特徵
SKIP_EVERY  = 1                         # 每張都用；可調 2、3 下採樣
TEST_RATE   = 0.30                      # ← test_size
MAX_RETRY   = 25                        # 最多重抽 25 次

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

# ───── 取 33 點 ─────
def extract_pose(data):
    if isinstance(data, dict) and 'pose' in data:
        return data['pose']
    if isinstance(data, dict) and 'results' in data:
        try: return data['results'][0]['pose_landmarks']
        except (KeyError, IndexError): pass
    if isinstance(data, list) and len(data) and 'keypoints' in data[0]:
        return data[0]['keypoints']
    return None

# ───── 角度計算 ─────
def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    norm = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    return math.acos(max(-1, min(1, dot/norm)))

# ───── 單張影格 → 特徵向量 ─────
def build_feat(kps):
    feat = []
    # 33 點座標
    for p in kps:
        feat.extend([p['x'], p['y']])
        if USE_Z:
            feat.append(p['z'])
    # 8 個關節角度
    feat.extend([_angle(kps, *tri) for tri in ANGLE_TRIPLETS])
    # 5 個跳躍特徵：左右腳踝高度、髖關節中心高度、與髖距離
    left_ankle_y  = kps[31]['y']
    right_ankle_y = kps[32]['y']
    hip_y         = (kps[23]['y'] + kps[24]['y']) / 2
    # 腳相對髖關節的跳躍高度（正值表示腳在髖下方）
    jump_height_left  = hip_y - left_ankle_y
    jump_height_right = hip_y - right_ankle_y
    feat.extend([
        left_ankle_y,
        right_ankle_y,
        hip_y,
        jump_height_left,
        jump_height_right
    ])
    return feat

# ───── 3. 讀取 JSON 檔並構建資料集 ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None:
        continue
    for i, jf in enumerate(sorted(folder.glob('*.json'))):
        if i % SKIP_EVERY != 0:
            continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps = extract_pose(data)
        if kps is None:
            continue
        rows.append(build_feat(kps))
        labels.append(lab)
        groups.append(folder.name)

print("樣本總數:", len(rows), Counter(labels))

X = np.asarray(rows, dtype='float32')
le = LabelEncoder()
y = le.fit_transform(labels)
groups = np.asarray(groups)

# ───── 4. 分組隨機抽測試集，直到涵蓋所有類別 ─────
gss = GroupShuffleSplit(test_size=TEST_RATE, random_state=42)
for attempt in range(1, MAX_RETRY+1):
    tr_idx, te_idx = next(gss.split(X, y, groups))
    if len(set(y[te_idx])) == len(le.classes_):
        print(f"✅ 第 {attempt} 次抽樣成功，測試集覆蓋全部類別")
        break
    if attempt == MAX_RETRY:
        sys.exit(f"❌ 連抽 {MAX_RETRY} 次仍缺類別，請增加資料或手動分割")

X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
print("train 分布:", Counter(y_tr))
print("test  分布:", Counter(y_te))

# ───── 5. 建立 RF 模型 + GridSearchCV ─────
rf = RandomForestClassifier(
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    oob_score=True
)

param_grid = {
    'n_estimators':      [80, 100, 120, 150],
    'max_depth':         [None, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4, 6],
    'max_features':      ['sqrt', 'log2'],
}
cv = GroupKFold(n_splits=3)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

grid.fit(X_tr, y_tr, groups=groups[tr_idx])
print("\n★ 最佳參數:", grid.best_params_)
clf = grid.best_estimator_

# ───── 6. 測試集評估 ─────
y_pred = clf.predict(X_te)
print("\n=== RandomForest 測試集報告 ===")
print(classification_report(y_te, y_pred, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

# ───── 7. 儲存模型 ─────
joblib.dump({'model': clf, 'label_encoder': le}, 'softtennis_pose_random_forest.pkl', compress=3)
print("\n✅ 已存 softtennis_pose_random_forest.pkl")
