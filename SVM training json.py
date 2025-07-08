#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LinearSVC 中文標籤版本（資料夾分組 + 超參數搜尋 + 容錯 extract_pose）
速度較 SVC(rbf/linear) 快數倍，可使用多核心
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
FOLDER2LABEL = {  # <略，同上一份；如有新增資料夾記得補>
    'IMG_9670':'正拍','IMG_9671':'正拍','IMG_9672':'正拍','IMG_9673':'正拍',
    'IMG_9674':'正拍','IMG_9675':'正拍','IMG_9676':'正拍',
    'IMG_9677':'反拍','IMG_9678':'反拍','IMG_9679':'反拍',
    'IMG_9680':'反拍','IMG_9681':'反拍',
    'IMG_9682':'基礎高壓發球','IMG_9683':'基礎高壓發球','IMG_9684':'基礎高壓發球',
    'IMG_9685':'基礎高壓發球','IMG_9686':'基礎高壓發球',
    'IMG_9687':'切球','IMG_9688':'切球','IMG_9689':'切球',
    'IMG_9690':'切球','IMG_9691':'切球',
    'IMG_9692':'進階高壓發球','IMG_9693':'進階高壓發球','IMG_9694':'進階高壓發球',
    'IMG_9695':'進階高壓發球','IMG_9696':'進階高壓發球',
}

USE_Z      = False   # 是否使用 z
SKIP_EVERY = 3       # 每張都用；大資料可調 3、5 下採樣

# ───── 角度特徵 ─────
ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16),    # 手肘
    (23, 25, 27), (24, 26, 28),     # 膝關節
    (15, 13, 11), (16, 14, 12),    # 腕-肘-肩（拍面角）
    (12, 24, 26), (11, 23, 25)     # 肩-髖-膝（蹬跳/轉體）

]

# ───── 取 33 點容錯 ─────
def extract_pose(data, fname=''):
    if isinstance(data, dict) and 'pose' in data:
        return data['pose']
    if isinstance(data, dict) and 'results' in data:
        try: return data['results'][0]['pose_landmarks']
        except (KeyError, IndexError): pass
    if isinstance(data, list) and len(data) and 'keypoints' in data[0]:
        return data[0]['keypoints']
    # 不合格就跳過
    return None

def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    return math.acos(max(-1, min(1,
                      dot / (math.hypot(*v1)*math.hypot(*v2)+1e-6))))

def build_feat(kps):
    feat = []
    for p in kps:
        feat += [p['x'], p['y']]
        if USE_Z: feat.append(p['z'])
    feat += [_angle(kps, *tri) for tri in ANGLE_TRIPLETS]
    return feat

# ───── 讀 JSON → 特徵 ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir(): continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None: continue
    for i, jf in enumerate(sorted(folder.glob('*.json'))):
        if i % SKIP_EVERY: continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        kps = extract_pose(data)
        if kps is None: continue
        rows.append(build_feat(kps))
        labels.append(lab)
        groups.append(folder.name)

print("樣本總數:", len(rows), Counter(labels))

# ───── 切訓練/測試 ─────
X = np.asarray(rows)
le = LabelEncoder(); y = le.fit_transform(labels)
groups = np.asarray(groups)

split = GroupShuffleSplit(test_size=0.2, random_state=42)
tr_idx, te_idx = next(split.split(X, y, groups))
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

# ───── Pipeline ＋ GridSearch (LinearSVC) ─────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(dual=False,  # n_samples > n_features 建議 False
                             class_weight='balanced',
                             max_iter=10000))
])

param_grid = {
    'linear_svc__C': [0.1, 1, 10]
}

grid = GridSearchCV(pipe, param_grid,
                    cv=2, n_jobs=-1,
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
