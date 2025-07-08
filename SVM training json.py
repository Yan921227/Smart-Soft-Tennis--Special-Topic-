#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVM 中文標籤版本（資料夾分組 + 超參數搜尋）
"""

import json, math, joblib
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ───── ① 你的資料根目錄 ─────
ROOT = Path("output_json")

# ───── ② 資料夾 → 中文標籤 ─────
FOLDER2LABEL = {
    'IMG_9670': '正拍', 'IMG_9671': '正拍','IMG_9672': '正拍','IMG_9673': '正拍','IMG_9674': '正拍','IMG_9675': '正拍','IMG_9676': '正拍',
    'IMG_9677': '反拍', 'IMG_9678': '反拍', 'IMG_9679': '反拍','IMG_9680': '反拍','IMG_9681': '反拍',
    'IMG_9682': '基礎高壓發球','IMG_9683': '基礎高壓發球','IMG_9684': '基礎高壓發球','IMG_9685': '基礎高壓發球','IMG_9686': '基礎高壓發球',
    'IMG_9687': '切球', 'IMG_9688': '切球', 'IMG_9689': '切球', 'IMG_9690': '切球', 'IMG_9691': '切球', 
    'IMG_9692': '進階高壓發球','IMG_9693': '進階高壓發球','IMG_9694': '進階高壓發球','IMG_9695': '進階高壓發球','IMG_9696': '進階高壓發球',
    
}

USE_Z      = False   # 要不要把 z 座標加進特徵
SKIP_EVERY = 1      # 每 3 張取 1 張；=1 則每張都用

# ───── 角度特徵設定 ─────
ANGLE_TRIPLETS = [
    (11, 13, 15), (12, 14, 16),   # 手肘
    (23, 25, 27), (24, 26, 28)    # 膝關節
]

def _angle(kps, a, b, c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    return math.acos(max(-1, min(1, dot / mag)))

def build_feat(kps):
    feat = []
    for p in kps:                           # 33 點
        feat += [p['x'], p['y']]
        if USE_Z:
            feat.append(p['z'])
    feat += [_angle(kps, *tri) for tri in ANGLE_TRIPLETS]
    return feat

# ───── 讀取所有 JSON ─────
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue
    label = FOLDER2LABEL.get(folder.name)
    if label is None:
        print('⚠️  跳過未定義資料夾 →', folder.name)
        continue

    for i, jf in enumerate(sorted(folder.glob('*.json'))):
        if i % SKIP_EVERY:          # 下採樣
            continue
        data = json.load(open(jf, 'r', encoding='utf-8'))
        rows.append(build_feat(data['pose']))
        labels.append(label)
        groups.append(folder.name)

print('樣本總數:', len(rows), Counter(labels))

# ───── 準備資料集 ─────
X = np.asarray(rows)
le = LabelEncoder()
y = le.fit_transform(labels)
groups = np.asarray(groups)

gss = GroupShuffleSplit(test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups))
X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

# ───── Pipeline + 超參數搜尋 ─────
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, class_weight='balanced'))
])

param_grid = {
    'svc__C':      [0.01, 0.1, 1, 10, 100],
    'svc__gamma':  np.logspace(-4, -1, 7),   # 1e-4 … 1e-1
    'svc__kernel': ['rbf', 'linear']
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_tr, y_tr)

print('★ 最佳參數:', grid.best_params_)
clf = grid.best_estimator_

# ───── 評估 ─────
y_pred = clf.predict(X_te)
print('\n=== 測試集報告 ===')
print(classification_report(y_te, y_pred, target_names=le.classes_))
print('混淆矩陣:\n', confusion_matrix(y_te, y_pred))

# ───── 儲存模型 ─────
joblib.dump({'model': clf, 'label_encoder': le}, 'softtennis_pose_svm_cn.pkl')
print('\n✅ 已存 softtennis_pose_svm_cn.pkl')
