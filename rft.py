#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Balanced RandomForest — A 方案（強化版）
- 特徵：
    * 穩健骨架標準化（肩寬/胯寬/軀幹長 中位數 + 窗口中位數平滑 + 尺度下限）
    * 33 點的 (x, y, [z]) + 8 個關節角
    * 離地 proxy：gap_l, gap_r, gap_m（標準化座標）
    * 時間特徵：hip_v、wrist_v、max_clear(窗口最大正向離地量)、mean_hip、mean_w_v
- 切分：GroupShuffleSplit（強制測試集涵蓋所有類別，多 seed）
- CV：StratifiedGroupKFold（如無則退回 GroupKFold）
- 訓練端：進階類只保留「在空中」幀（top-q by max_clear）
- 推論端：時間平滑（WIN/HYST） + 嚴格 gating（τ、max_clear 門檻、信心邊際、最短連續幀）
"""

import json, math, sys
from pathlib import Path
from collections import Counter, deque, defaultdict
import numpy as np
import joblib

from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics        import classification_report, confusion_matrix

# ---- 可用則採用 StratifiedGroupKFold（sklearn>=1.3）----
try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except Exception:
    _HAS_SGK = False

# ---- imbalanced-learn ----
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except ModuleNotFoundError:
    sys.exit("請先安裝：  pip install -U imbalanced-learn")

# =============== 全域參數 ===============
ROOT         = Path("output_json")
USE_Z        = True
SKIP_EVERY   = 1
TEST_RATE    = 0.30
MAX_RETRY    = 500
RANDOM_SEED0 = 42

WIN_SIZE     = 5      # 時間窗口（用來算 max_clear 等）
WRIST_IDX    = 16     # 右手腕(左手持拍可改15)

# ---- 尺度平滑（關鍵！避免離地特徵爆掉）----
SCALE_WIN    = 9       # 尺度滑動窗口（幀）— 用中位數平滑
SCALE_FLOOR  = 0.05    # 尺度下限（避免尺度太小造成放大）

# ---- 時間平滑 & gating（保守）----
SMOOTH_WIN   = 15
SMOOTH_HYST  = 9
GATE_TAU     = 0.90    # 進階機率門檻
GAP_Q        = 0.97    # 非進階 max_clear 的分位數門檻（0.95~0.98）
MARGIN_MIN   = 0.12    # 信心邊際（top1 - top2）
MIN_RUN      = 12      # 進階最短連續幀數

# ---- 訓練端：只保留「進階 in‑air 幀」的比例（Top-q by max_clear）----
FILTER_ADV_IN_AIR = True
ADV_KEEP_TOPQ     = 0.40   # 例如保留最高的 40% 進階幀

# ---- 資料夾 → 類別 ----
FOLDER2LABEL = {
    'IMG_1158':'正拍','IMG_1159':'正拍','IMG_1160':'正拍',
    'IMG_1174':'反拍', 'IMG_1217':'反拍',
    'IMG_1219':'切球', 'IMG_1220':'切球',
    'IMG_1163':'基礎高壓發球','IMG_1164':'基礎高壓發球',
    # 'IMG_1165':'基礎高壓發球','IMG_1166':'基礎高壓發球',
    'IMG_1179':'進階高壓發球','IMG_1221':'進階高壓發球',
    'IMG_1224':'進階高壓發球','IMG_1225':'進階高壓發球',
    'IMG_1227':'進階高壓發球','IMG_1228':'進階高壓發球',
}

# =============== 幾何輔助 ===============
ANGLE_TRIPLETS = [
    (11,13,15),(12,14,16),
    (23,25,27),(24,26,28),
    (15,13,11),(16,14,12),
    (12,24,26),(11,23,25)
]

def _angle(kps, a,b,c):
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1 = (ax-bx, ay-by); v2 = (cx-bx, cy-by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    den = (math.hypot(*v1)*math.hypot(*v2) + 1e-6)
    return math.acos(max(-1.0, min(1.0, dot/den)))

# ---- 穩健尺度：肩寬、胯寬、軀幹長 的中位數 ----
def _robust_scale_now(kps):
    sh = math.hypot(kps[11]['x'] - kps[12]['x'], kps[11]['y'] - kps[12]['y'])
    hip= math.hypot(kps[23]['x'] - kps[24]['x'], kps[23]['y'] - kps[24]['y'])
    cx_sh = (kps[11]['x'] + kps[12]['x'])/2; cy_sh = (kps[11]['y'] + kps[12]['y'])/2
    cx_hp = (kps[23]['x'] + kps[24]['x'])/2; cy_hp = (kps[23]['y'] + kps[24]['y'])/2
    torso  = math.hypot(cx_sh - cx_hp, cy_sh - cy_hp)
    raw = np.median([sh, hip, torso])
    return max(raw, 1e-6)

def _normalize_pose_with_scale(kps, scale):
    hip_cx = (kps[23]['x'] + kps[24]['x'])/2.0
    hip_cy = (kps[23]['y'] + kps[24]['y'])/2.0
    nkps = []
    for p in kps:
        nx = (p['x'] - hip_cx)/scale
        ny = (p['y'] - hip_cy)/scale
        if USE_Z:
            nkps.append({'x': nx, 'y': ny, 'z': p.get('z',0.0)/scale})
        else:
            nkps.append({'x': nx, 'y': ny})
    return nkps

# =============== 單幀特徵（穩健尺度 + 角度 + 離地3） ===============
def frame_basic_feat(kps, scale_hist):
    # 1) 更新尺度平滑列
    scale_hist.append(_robust_scale_now(kps))
    scale = max(np.median(scale_hist), SCALE_FLOOR)

    # 2) 以平滑尺度做標準化
    nkps = _normalize_pose_with_scale(kps, scale)

    # 3) 展開座標
    feat = []
    for p in nkps:
        feat.extend([p['x'], p['y']])
        if USE_Z: feat.append(p['z'])

    # 4) 8 角
    feat.extend([_angle(nkps,*tri) for tri in ANGLE_TRIPLETS])

    # 5) 離地 proxy
    hip_y = (nkps[23]['y'] + nkps[24]['y'])/2.0
    la_y  = nkps[31]['y']; ra_y = nkps[32]['y']
    gap_l = hip_y - la_y; gap_r = hip_y - ra_y
    gap_m = (gap_l + gap_r)/2.0
    feat.extend([gap_l, gap_r, gap_m])

    wrist_y = nkps[WRIST_IDX]['y']
    return feat, hip_y, la_y, ra_y, wrist_y, gap_m

# =============== 讀資料 + 時間特徵 ===============
rows, labels, groups = [], [], []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir(): continue
    lab = FOLDER2LABEL.get(folder.name)
    if lab is None: continue

    # 尺度與時間緩衝列
    scale_hist = deque(maxlen=SCALE_WIN)
    hip_hist   = deque(maxlen=WIN_SIZE)
    wrist_hist = deque(maxlen=WIN_SIZE)
    clr_hist   = deque(maxlen=WIN_SIZE)  # 正向離地量（clearance = -gap_m）

    for i, jf in enumerate(sorted(folder.glob("*.json"))):
        if i % SKIP_EVERY: continue
        with open(jf,'r',encoding='utf-8') as f:
            data = json.load(f)
        kps = (data.get('pose') or
               (data.get('results',[{}])[0]).get('pose_landmarks') or
               (data[0].get('keypoints') if isinstance(data,list) else None))
        if kps is None: continue

        basic, hip_y, la_y, ra_y, wrist_y, gap_m = frame_basic_feat(kps, scale_hist)

        # 一階速度
        hip_v   = hip_y - (hip_hist[-1] if hip_hist else hip_y)
        wrist_v = wrist_y - (wrist_hist[-1] if wrist_hist else wrist_y)

        # 更新窗口
        hip_hist.append(hip_y)
        wrist_hist.append(wrist_y)
        clearance = -gap_m  # 正向離地量
        clr_hist.append(clearance)

        # 窗口統計
        max_clear  = max(clr_hist)
        mean_hip   = sum(hip_hist)/len(hip_hist)
        mean_w_v   = (sum(wrist_hist[i]-wrist_hist[i-1]
                      for i in range(1,len(wrist_hist))) / max(1, len(wrist_hist)-1))

        rows.append(basic + [hip_v, wrist_v, max_clear, mean_hip, mean_w_v])
        labels.append(lab)
        groups.append(folder.name)

X = np.asarray(rows, dtype='float32')
groups = np.asarray(groups)
le = LabelEncoder(); y = le.fit_transform(labels)

print("樣本總數:", len(rows), Counter(labels))

# === 基本特徵長度（用來抓 gap_m 位置、但 gating 我們用 max_clear）===
D_BASIC = 33*(2 + (1 if USE_Z else 0)) + len(ANGLE_TRIPLETS) + 3
# 附加時間特徵附加次序：[hip_v, wrist_v, max_clear, mean_hip, mean_w_v]
CLEAR_IDX = -3  # max_clear 在整體向量中的索引

# =============== Group 切分（測試含全類別，多 seed） ===============
ALL = set(range(len(le.classes_)))
found=False
for seed in range(RANDOM_SEED0, RANDOM_SEED0+MAX_RETRY):
    gss = GroupShuffleSplit(test_size=TEST_RATE, n_splits=1, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    if set(y[te_idx]) == ALL:
        found=True; break
if not found:
    sys.exit("測試集仍缺類別，請增加每類別的影片資料夾數或提高 MAX_RETRY。")

X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
groups_tr  = groups[tr_idx]
print(f"切分成功（seed={seed}）：train={len(tr_idx)}  test={len(te_idx)}")

# =============== 進階 in-air 訓練過濾（可關） ===============
ADV = {n:i for i,n in enumerate(le.classes_)}.get('進階高壓發球', None)

if FILTER_ADV_IN_AIR and ADV is not None:
    adv_mask = (y_tr == ADV)
    if np.sum(adv_mask) > 12:  # 足夠幀數才過濾
        thr_in_air = np.quantile(X_tr[adv_mask, CLEAR_IDX], 1-ADV_KEEP_TOPQ)
        keep_mask = (~adv_mask) | (X_tr[:, CLEAR_IDX] >= thr_in_air)
        print(f"(info) 訓練集進階 in-air 門檻 = {thr_in_air:.3f}，進階幀由 {np.sum(adv_mask)} → {np.sum(adv_mask & (X_tr[:, CLEAR_IDX] >= thr_in_air))}")
        X_tr, y_tr, groups_tr = X_tr[keep_mask], y_tr[keep_mask], groups_tr[keep_mask]

# =============== BRF（A 方案）：只用 auto 取樣 ===============
brf = BalancedRandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    sampling_strategy="auto",   # 每折自動 under-sampling
    replacement=False,
    n_jobs=-1, random_state=42, oob_score=False
)

# CV：優先使用 StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) if _HAS_SGK else GroupKFold(n_splits=3)

grid = GridSearchCV(
    brf,
    {"max_depth":[None,16],
     "min_samples_leaf":[2,4],
     "max_features":["sqrt","log2"]},
    scoring="f1_macro",
    n_jobs=-1, cv=cv, verbose=1,
    error_score=np.nan
)
grid.fit(X_tr, y_tr, groups=groups_tr)
clf = grid.best_estimator_
print("Best params:", grid.best_params_)

# =============== 評估（逐幀） ===============
proba_te   = clf.predict_proba(X_te)
y_pred_raw = np.argmax(proba_te, axis=1)
print("\n=== BRF（auto）原始逐幀 ===")
print(classification_report(y_te, y_pred_raw, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred_raw))

# =============== 時間平滑 ===============
def smooth_by_folder(proba, folders, te_indices, win=15, hyst=5):
    y_pred = np.argmax(proba, axis=1).copy()
    if win<=1 and hyst<=1: return y_pred
    out = y_pred.copy()
    for f in np.unique(folders):
        idx = np.where(folders==f)[0]
        order = idx[np.argsort(te_indices[idx])]
        P = proba[order]
        T,C = P.shape
        ker = np.ones(win)/win if win>1 else np.array([1.0])
        P_s = np.zeros_like(P)
        for c in range(C):
            P_s[:,c] = np.convolve(P[:,c], ker, mode='same')
        y_hat = np.argmax(P_s, axis=1)
        if hyst>1 and T>0:
            cur=y_hat[0]; run=0
            for i in range(1,T):
                if y_hat[i]!=cur:
                    run+=1
                    if run<hyst: y_hat[i]=cur
                    else: cur=y_hat[i]; run=0
                else: run=0
        out[order]=y_hat
    return out

folders_te   = groups[te_idx]
y_pred_smooth= smooth_by_folder(proba_te, folders_te, te_idx, win=SMOOTH_WIN, hyst=SMOOTH_HYST)
print("\n=== 使用時間平滑後 ===")
print(classification_report(y_te, y_pred_smooth, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred_smooth))

# =============== 嚴格 gating（進階）— 用 max_clear 門檻 ===============
# 用訓練集「非進階」幀的 max_clear（更穩）估門檻
if ADV is not None:
    maxclr_nonadv_tr = X_tr[y_tr != ADV, CLEAR_IDX]
else:
    maxclr_nonadv_tr = X_tr[:, CLEAR_IDX]
GAP_THR = float(np.quantile(maxclr_nonadv_tr, GAP_Q)) if len(maxclr_nonadv_tr) else 0.0

def gate_adv_strict(proba, X_te, pred_in, adv_idx, tau=0.90, gap_thr=0.0, margin=0.12, min_run=12):
    out = pred_in.copy()
    if adv_idx is None or np.sum(out==adv_idx)==0: return out
    p_adv = proba[np.arange(len(out)), adv_idx]
    # 次高機率
    p2    = np.partition(proba, -2, axis=1)[:, -2]
    margin_ok = (p_adv - p2) >= margin
    # 用窗口最大正向離地量判定是否真的有跳
    clearance = X_te[:, CLEAR_IDX]
    keep = (p_adv>=tau) & (clearance>=gap_thr) & margin_ok

    to_demote = (out==adv_idx) & (~keep)
    if np.any(to_demote):
        sec = np.argsort(proba[to_demote], axis=1)[:, -2]
        out[to_demote] = sec

    # 最短連續幀限制
    if min_run>1:
        i=0; N=len(out)
        while i<N:
            if out[i]!=adv_idx: i+=1; continue
            j=i
            while j<N and out[j]==adv_idx: j+=1
            if (j-i)<min_run:
                sec_seg = np.argsort(proba[i:j], axis=1)[:, -2]
                out[i:j] = sec_seg
            i=j
    return out

y_pred_final = gate_adv_strict(
    proba_te, X_te, y_pred_smooth, ADV,
    tau=GATE_TAU, gap_thr=GAP_THR, margin=MARGIN_MIN, min_run=MIN_RUN
)
print("\n=== 時間平滑 + 嚴格 gating 後 ===")
print(f"(info) τ={GATE_TAU}, GAP_THR={GAP_THR:.4f}, margin={MARGIN_MIN}, min_run={MIN_RUN}")
print(classification_report(y_te, y_pred_final, target_names=le.classes_))
print("混淆矩陣:\n", confusion_matrix(y_te, y_pred_final))

# =============== 方便檢查：max_clear 分佈（可觀察門檻是否合理） ===============
def debug_clearance_stats(X, y, le, idx=CLEAR_IDX, title="max_clear 分佈"):
    print(f"\n=== {title} ===")
    for ci, name in enumerate(le.classes_):
        vals = X[y==ci, idx]
        if len(vals)==0: continue
        q = np.percentile(vals, [50, 90, 95, 99])
        print(f"{name:>10s}: n={len(vals):5d} | p50={q[0]:.3f} p90={q[1]:.3f} p95={q[2]:.3f} p99={q[3]:.3f}")

debug_clearance_stats(X_tr, y_tr, le, idx=CLEAR_IDX, title="(train) max_clear")
debug_clearance_stats(X_te, y_te, le, idx=CLEAR_IDX, title="(test)  max_clear")

# =============== 儲存（模型+標籤器） ===============
out_path = "20251006softtennis_pose_brf_auto_strong.pkl"
joblib.dump({"model": clf, "label_encoder": le}, out_path, compress=3)
print(f"\n✅ 已存 {out_path}")
