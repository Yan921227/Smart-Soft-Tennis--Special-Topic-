# -*- coding: utf-8 -*-
"""
純關鍵點二分類訓練（Backswing vs Impact）
- 直接使用 MediaPipe 33 點 (x,y) 共 66 維
- 印出整體準確率 + 分類報告 + 混淆矩陣
- 混淆矩陣：保留標題，但移除圖中所有方框（axes spines & colorbar 外框）
- 自動儲存：模型 (rf_pose_model.pkl)、混淆矩陣圖 (confusion_matrix.png)

pip install numpy pandas scikit-learn matplotlib seaborn joblib
"""

import glob, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# =========================
# 可調參數
# =========================
DATA_ROOT   = "data"                 # 需有 data/backswing 與 data/impact
MODEL_PATH  = "rf_pose_model.pkl"
CM_FIG_PATH = "confusion_matrix.png"
TEST_SIZE   = 0.30
RAND_SEED   = 42
N_EST       = 300
MAX_DEPTH   = 12

# =========================
# 33 點名稱（此版直接按 index 取 (x,y)）
# =========================
MP_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

def load_pose_xy66(json_path: str) -> np.ndarray:
    """從 Mediapipe JSON 讀 33 個 (x,y) → 66 維；缺失填 0"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = {int(d["index"]): (float(d["x"]), float(d["y"])) for d in data.get("pose", [])}
    arr = []
    for i in range(33):
        if i in pts:
            arr += [pts[i][0], pts[i][1]]
        else:
            arr += [0.0, 0.0]
    return np.array(arr, dtype=np.float32)

def load_dataset(base_path: str = DATA_ROOT):
    """讀取 data/backswing 與 data/impact 兩類 JSON，回傳 X, y"""
    samples = []
    for sub, label in [("backswing", 0), ("impact", 1)]:
        for p in glob.glob(f"{base_path}/{sub}/*.json"):
            samples.append((load_pose_xy66(p), label))
    if not samples:
        raise RuntimeError(f"找不到資料：{base_path}/backswing 或 {base_path}/impact 內沒有 .json")
    X = np.stack([s[0] for s in samples])
    y = np.array([s[1] for s in samples], dtype=np.int64)
    return X, y

def main():
    # 1) 讀資料
    X, y = load_dataset(DATA_ROOT)
    print(f"讀入資料：{X.shape}, 標籤分布 = {np.bincount(y)}")  # (N,66), [#backswing #impact]

    # 2) 切訓練/測試
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RAND_SEED, stratify=y
    )

    # 3) 訓練模型（Random Forest）
    clf = RandomForestClassifier(
        n_estimators=N_EST, max_depth=MAX_DEPTH, random_state=RAND_SEED
    )
    clf.fit(X_train, y_train)

    # 4) 儲存模型
    joblib.dump(clf, MODEL_PATH)
    print(f"[OK] 模型已儲存：{MODEL_PATH}")

    # 5) 預測與評估
    y_pred = clf.predict(X_test)
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- 混淆矩陣 (文字版) ---")
    print(cm)
    print("\n--- 分類報告 ---")
    print(classification_report(y_test, y_pred, target_names=["Backswing","Impact"]))
    print(f"\n🎯 整體準確率 (Accuracy): {acc:.4f}")

    # 6) 畫混淆矩陣（保留標題；移除所有方框）
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    hm = sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Backswing", "Impact"],
        yticklabels=["Backswing", "Impact"],
        cbar=True, ax=ax
    )
    # 標題要留著
    ax.set_title("混淆矩陣 (Confusion Matrix)", pad=8)
    ax.set_xlabel("預測結果")
    ax.set_ylabel("真實標籤")

    # ❌ 移除圖內所有方框：四周 spines + colorbar 外框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # colorbar 外框
    cbar = hm.collections[0].colorbar
    if cbar is not None and hasattr(cbar, "outline"):
        cbar.outline.set_visible(False)

    # 讓圖面緊湊、乾淨
    plt.tight_layout(pad=0.5)
    plt.savefig(CM_FIG_PATH, dpi=150, facecolor="white")
    print(f"[OK] 混淆矩陣已存檔：{CM_FIG_PATH}")
    plt.show()

if __name__ == "__main__":
    main()
