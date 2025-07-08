#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinearSVC 中文標籤版本
‧ 資料夾分組  ‧ 容錯 extract_pose  ‧ GridSearch  ‧ 多核心
★ 已新增「跳躍特徵」： ankle_min / hip_y / hip_to_ankle
"""

import json, math, joblib
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ───── ① 資料根目錄 ─────
ROOT = Path("output_json")

# ───── ② 資料夾 → 中文標籤 ─────
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

USE_Z      = False    # 是否把 z 也放進特徵
SKIP_EVERY = 1        # 每張都用；資料太大可調成 3

# ───── 角度特徵 ─────
ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16),    # 手肘
    (23, 25, 27), (24, 26, 28),    # 膝關節
    (15, 13, 11), (16, 14, 12),    # 腕-肘-肩（拍面角）
    (12, 24, 26), (11, 23, 25)     # 肩-髖-膝（蹬跳/轉體）
]

# ───── Mediapipe 關鍵點索引備註 ─────
# hip   : 23(L) 24(R)
# knee  : 27(L) 28(R)
# ankle : 31(L) 32(R)

# ───── 取 33 點容錯 ─────
def extract_pose(data):
    if isinstance(data, dict) and 'pose' in data:
        return data['pose']
    if isinstance(data, dict) and 'results' in data:
        try:
            return data['results'][0]['pose_landmarks']
        except (KeyError, IndexError):
            pass
    if isinstance(data, list) and len(data) and 'keypoints' in data[0]:
        return data[0]['keypoints']
    return None

# 夾角計算
def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    return math.acos(max(-1, min(1,
                      dot / (math.hypot(*v1)*math.hypot(*v2)+1e-6))))

# 建立單張影格特徵
def build_feat(kps):
    feat = []

    # ① 33 點 (x, y[, z])
    for p in kps:
        feat += [p['x'], p['y']]
        if USE_Z:
            feat.append(p['z'])

    # ② 夾角
    feat += [_angle(kps, *tri) for tri in ANGLE_TRIPLETS]

    # ③ 跳躍特徵
    hip_y = (kps[23]['y'] + kps[24]['y']) / 2          # 髖高度
    ankle_min = min(kps[31]['y'], kps[32]['y'])        # 腳離地高度（y 越小離地越高）
    hip_to_ankle = ankle_min - hip_y                   # 腳-髖 垂直距離
    feat += [ankle_min, hip_y, hip_to_ankle]

    return feat

# ───── 讀 JSON → 特徵 ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None:
        continue
    for idx, jf in enumerate(sorted(folder.glob('*.json'))):
        if idx % SKIP_EVERY:
            continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps = extract_pose(data)
        if kps is None:
            continue
        rows.append(build_feat(kps))
        labels.append(lab)
        groups.append(folder.name)

print("樣本總數:", len(rows), Counter(labels))

# ───── 切 train / test ─────
X = np.asarray(rows)
le = LabelEncoder()
y = le.fit_transform(labels)
groups = np.asarray(groups)

split = GroupShuffleSplit(test_size=0.2, random_state=42)
tr_idx, te_idx = next(split.split(X, y, groups))
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

# ───── Pipeline + GridSearch ─────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(
        dual=False,              # n_samples > n_features 比較快
        class_weight='balanced',
        max_iter=10000))
])

param_grid = {'linear_svc__C': [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid,
                    cv=3, n_jobs=-1,
                    scoring='f1_macro', verbose=1)
grid.fit(X_tr, y_tr)

print("★ 最佳參數:", grid.best_params_)
clf = grid.best_estimator_

# ───── 評估 ─────
y_pred = clf.predict(X_te)
print("\n=== 測試集報告 ===")
print(classification_report(y_te, y_pred, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

# ───── 儲存模型 ─────
joblib.dump({'model': clf, 'label_encoder': le},
            'softtennis_pose_linearSVC.pkl')
print("\n✅ 已存 softtennis_pose_linearSVC.pkl")
