# simple_train_softtennis.py
# -----------------------------------------------------------
# 把同一動作的 JSON 放在同一資料夾裡，再修改 ROOT 與 FOLDER2LABEL 後直接執行
# -----------------------------------------------------------
import json, math, joblib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ======= ① 根目錄：放多個動作資料夾 =========
ROOT = Path(r'D:\softtennis\json')   # ← 換成自己的路徑

# ======= ② 每個資料夾對應到哪個動作 =========
FOLDER2LABEL = {
    'IMG_9670': 'forehand',
    'IMG_9684': 'forehand',

    'IMG_9671': 'backhand',
    'IMG_9692': 'backhand',
    'IMG_9697': 'backhand',

    'IMG_9672': 'flat_drive',
    'IMG_9673': 'basic_overhead_serve',
    # 'IMG_97xx': 'advanced_overhead_serve',
    # 'IMG_97xx': 'slice_shot',
}

# ======= ③ 是否把 z 也加進特徵 =========
USE_Z = False      # True / False 都行


# ------------ 計算角度（可先不管） ------------
ANGLE_TRIPLETS = [(11,13,15), (12,14,16), (23,25,27), (24,26,28)]
def _angle(kps, a,b,c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2) + 1e-6
    return math.acos(max(-1, min(1, dot/mag)))

def build_feature(pose_kps):
    feat = []
    for kp in pose_kps:                       # 33 個點
        feat += [kp['x'], kp['y']]
        if USE_Z:
            feat.append(kp['z'])
    # 再加 4 個關節角度（不想用就註解掉）
    feat += [_angle(pose_kps,*tri) for tri in ANGLE_TRIPLETS]
    return feat


# ========== 讀資料夾 → DataFrame ==========
rows, labels = [], []
for folder in ROOT.iterdir():
    if not folder.is_dir():               # 只看資料夾
        continue
    label = FOLDER2LABEL.get(folder.name)
    if label is None:
        print('⚠️  未在 FOLDER2LABEL 定義，跳過', folder.name)
        continue

    for jf in folder.glob('*.json'):
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pose_kps = data['pose']
        rows.append(build_feature(pose_kps))
        labels.append(label)

print('讀取完成，樣本數：', len(rows), Counter(labels))

X = np.asarray(rows)
le = LabelEncoder()
y = le.fit_transform(labels)

# ========== 切 train / test，這裡最簡單用隨機分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# ========== 建管線：標準化 + SVM ==========
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf', C=10, gamma=0.01,
                probability=True, class_weight='balanced'))
])
clf.fit(X_train, y_train)

# ========== 評估 ==========
y_pred = clf.predict(X_test)
print('\n=== 測試集分類報告 ===')
print(classification_report(y_test, y_pred, target_names=le.classes_))
print('混淆矩陣：\n', confusion_matrix(y_test, y_pred))

# ========== 存模型 ==========
MODEL_OUT = Path('softtennis_pose_simple.pkl')
joblib.dump({'model': clf, 'label_encoder': le}, MODEL_OUT)
print('\n✅ 已儲存模型到', MODEL_OUT.absolute())


# ==========  (可選) 示範推論 ==========
def predict_one(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        sample = json.load(f)
    feat = np.asarray(build_feature(sample['pose'])).reshape(1, -1)
    proba = clf.predict_proba(feat)[0]
    idx = proba.argmax()
    return le.inverse_transform([idx])[0], proba[idx]

# demo
# label, conf = predict_one(r'D:\softtennis\json\IMG_9670\frame_00123.json')
# print('預測', label, f'({conf:.1%})')
