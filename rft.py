#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
brf_strong_opt.py — Balanced RandomForest 強化版（優化）
維持你的方法論：穩健尺度 + in-air 過濾 + group 切分 + 平滑/gating
但在 IO、數值穩定、速度與維護性上做強化；提供多個開關讓你「快/準」自由切換。
"""

# ====== 0) 依賴自動安裝（含 imbalanced-learn / ujson 可選） ======
import sys, subprocess
def _ensure(pkgs):
    miss = []
    for p in pkgs:
        try: __import__(p)
        except Exception: miss.append(p)
    if miss:
        print(f"[INFO] 安裝缺套件：{miss}")
        code = subprocess.call([sys.executable, "-m", "pip", "install", *miss])
        if code != 0:
            print("[ERROR] 自動安裝失敗；請手動安裝：", " ".join(miss)); sys.exit(1)

_ensure(["numpy", "scikit_learn", "joblib", "tqdm", "imbalanced_learn"])
# ujson 非必要；若無則退回 json
try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib

# ====== 1) 匯入 ======
from pathlib import Path
from collections import Counter, deque, defaultdict
import time
import numpy as np
import joblib
from tqdm import tqdm

from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold
from sklearn.metrics        import classification_report, confusion_matrix

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except Exception:
    _HAS_SGK = False

from imblearn.ensemble import BalancedRandomForestClassifier

# ====== 2) 全域設定（你可以只改這區） ======
ROOT           = Path("output_json")
USE_Z          = True
SKIP_EVERY     = 1           # 每隔幀取 1，>1 可加速
FILE_LIMIT     = None        # 例如 100：每資料夾最多讀 100 個 json（加速用）；None = 全部
TEST_RATE      = 0.30
RANDOM_SEED0   = 42
REQUIRE_ALL_CLASSES = True   # True：測試集必須涵蓋所有類別；False：允許缺

# 尺度與時間窗口
WIN_SIZE     = 5
WRIST_IDX    = 16
SCALE_WIN    = 9
SCALE_FLOOR  = 0.05

# 時間平滑與 gating（預設只平滑、不 gating；要更保守再打開）
USE_SMOOTH   = True
SMOOTH_WIN   = 15
SMOOTH_HYST  = 9

USE_GATING   = False
GATE_TAU     = 0.90
GAP_Q        = 0.97
MARGIN_MIN   = 0.12
MIN_RUN      = 12

# 訓練端：進階 in-air 篩選
FILTER_ADV_IN_AIR = True
ADV_KEEP_TOPQ     = 0.40

# 模型與調參
USE_GRID      = False   # True 會做小格網搜尋；False 直接訓練以提速
BASE_PARAMS   = dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
                     max_features="sqrt", sampling_strategy="auto",
                     replacement=False, n_jobs=-1, random_state=42, oob_score=False)
GRID_PARAMS   = {"max_depth":[None,16], "min_samples_leaf":[2,4], "max_features":["sqrt","log2"]}

# 你的類別對應
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

# ====== 3) 幾何輔助（向量化角度計算） ======
ANGLE_TRIPLETS = np.array([
    [11,13,15],[12,14,16],
    [23,25,27],[24,26,28],
    [15,13,11],[16,14,12],
    [12,24,26],[11,23,25]
], dtype=int)

def _angles_from_points(P: np.ndarray, triplets: np.ndarray) -> np.ndarray:
    """
    P: (33,3) 已正規化
    triplets: (K,3) 索引
    回傳 (K,) 角度（弧度）
    """
    a = P[triplets[:,0], :2]
    b = P[triplets[:,1], :2]
    c = P[triplets[:,2], :2]
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1) + 1e-6
    n2 = np.linalg.norm(v2, axis=1) + 1e-6
    cosang = np.clip((v1*v2).sum(axis=1)/(n1*n2), -1.0, 1.0)
    return np.arccos(cosang)

# ====== 4) 解析與正規化 ======
POSE = {"ls":11,"rs":12,"le":13,"re":14,"lw":15,"rw":16,"lh":23,"rh":24,"lk":25,"rk":26,"la":27,"ra":28,"lfi":31,"rfi":32}

def _ensure_33(frame) -> np.ndarray:
    """把 frame 轉為 (33,3)：x,y,z；不足補 nan。支援常見 schema。"""
    if isinstance(frame, dict):
        if "pose_landmarks" in frame and "landmark" in frame["pose_landmarks"]:
            arr = frame["pose_landmarks"]["landmark"]
        elif "landmarks" in frame:  # 自訂
            arr = frame["landmarks"]
        elif "pose" in frame and isinstance(frame["pose"], list):
            arr = frame["pose"]
        else:
            arr = frame.get("landmark", [])
    elif isinstance(frame, list):
        arr = frame
    else:
        arr = []

    out = np.full((33,3), np.nan, dtype=np.float32)
    n = min(33, len(arr))
    for i in range(n):
        if isinstance(arr[i], dict):
            out[i,0] = arr[i].get("x", np.nan)
            out[i,1] = arr[i].get("y", np.nan)
            out[i,2] = arr[i].get("z", 0.0)
        elif isinstance(arr[i], (list, tuple)) and len(arr[i]) >= 2:
            out[i,0], out[i,1] = arr[i][0], arr[i][1]
            out[i,2] = arr[i][2] if len(arr[i])>2 else 0.0
    return out

def _load_frames(path: Path):
    data = jsonlib.loads(path.read_text(encoding="utf-8"))
    frames = []
    if isinstance(data, list):
        for fr in data: frames.append(_ensure_33(fr))
    elif isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        for fr in data["frames"]: frames.append(_ensure_33(fr))
    else:
        frames.append(_ensure_33(data))
    return frames

def _robust_scale_now(P: np.ndarray) -> float:
    ls, rs, lh, rh = POSE["ls"], POSE["rs"], POSE["lh"], POSE["rh"]
    sh  = np.linalg.norm(P[ls,:2] - P[rs,:2])
    hip = np.linalg.norm(P[lh,:2] - P[rh,:2])
    shc = (P[ls,:2] + P[rs,:2]) / 2
    hpc = (P[lh,:2] + P[rh,:2]) / 2
    torso = np.linalg.norm(shc - hpc)
    raw = np.nanmedian([sh, hip, torso])
    return float(max(raw, 1e-6))

def _normalize_with_scale(P: np.ndarray, scale: float) -> np.ndarray:
    lh, rh = POSE["lh"], POSE["rh"]
    hipc = (P[lh,:2] + P[rh,:2]) / 2
    Q = P.copy()
    Q[:,:2] -= hipc
    Q[:,:] /= max(scale, SCALE_FLOOR)
    return Q

def _frame_basic_feat(P: np.ndarray, scale_hist: deque):
    """回傳: feature(list), hip_y, la_y, ra_y, wrist_y, gap_m"""
    scale_hist.append(_robust_scale_now(P))
    scale = max(np.median(scale_hist), SCALE_FLOOR)
    Q = _normalize_with_scale(P, scale)

    # 展開座標
    if USE_Z:
        flat = Q.reshape(-1)
    else:
        flat = Q[:,:2].reshape(-1)

    # 8 角（弧度）
    ang = _angles_from_points(Q, ANGLE_TRIPLETS)

    # 離地 proxy
    lh, rh, la, ra, rw = POSE["lh"], POSE["rh"], POSE["la"], POSE["ra"], WRIST_IDX
    hip_y = (Q[lh,1] + Q[rh,1]) / 2.0
    la_y, ra_y = Q[la,1], Q[ra,1]
    gap_l, gap_r = hip_y - la_y, hip_y - ra_y
    gap_m = (gap_l + gap_r) / 2.0
    wrist_y = Q[rw,1]

    feat = np.concatenate([flat, ang, np.array([gap_l, gap_r, gap_m], dtype=np.float32)], axis=0)
    return feat.astype(np.float32), float(hip_y), float(la_y), float(ra_y), float(wrist_y), float(gap_m)

# ====== 5) 資料讀取（非遞迴 + 可抽樣） ======
def iter_files_non_recursive():
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir(): continue
        lab = FOLDER2LABEL.get(folder.name)
        if lab is None: continue
        files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower()==".json"])
        if FILE_LIMIT: files = files[:FILE_LIMIT]
        yield folder.name, lab, files

def build_dataset():
    rows, labels, groups = [], [], []
    per_class = defaultdict(int)

    t0 = time.time()
    for group, lab, files in iter_files_non_recursive():
        scale_hist = deque(maxlen=SCALE_WIN)
        hip_hist   = deque(maxlen=WIN_SIZE)
        wrist_hist = deque(maxlen=WIN_SIZE)
        clr_hist   = deque(maxlen=WIN_SIZE)

        for i, jf in enumerate(files):
            if i % SKIP_EVERY: continue
            try:
                frames = _load_frames(jf)
            except Exception:
                continue
            # 逐幀生成（這邊假設每個 json 代表單幀；若多幀也能處理）
            for P in frames:
                feat, hip_y, la_y, ra_y, wrist_y, gap_m = _frame_basic_feat(P, scale_hist)
                hip_v   = hip_y   - (hip_hist[-1]   if hip_hist   else hip_y)
                wrist_v = wrist_y - (wrist_hist[-1] if wrist_hist else wrist_y)

                hip_hist.append(hip_y); wrist_hist.append(wrist_y)
                clearance = -gap_m; clr_hist.append(clearance)

                # 窗口統計
                max_clear = max(clr_hist)
                mean_hip  = float(np.mean(hip_hist))
                if len(wrist_hist) > 1:
                    diffs = np.diff(np.array(wrist_hist, dtype=np.float32))
                    mean_w_v = float(np.mean(diffs))
                else:
                    mean_w_v = 0.0

                rows.append(np.concatenate([feat, np.array([hip_v, wrist_v, max_clear, mean_hip, mean_w_v], dtype=np.float32)]))
                labels.append(lab)
                groups.append(group)
                per_class[lab] += 1

    if not rows:
        raise RuntimeError("未讀到任何樣本，請確認資料夾與 JSON 格式。")

    print("\n=== 各類別樣本數 ===")
    total = 0
    for name in sorted(per_class.keys()):
        print(f"  {name}: {per_class[name]}")
        total += per_class[name]
    print(f"  總計: {total} | 讀取時間: {time.time()-t0:.1f}s")

    X = np.asarray(rows, dtype=np.float32)
    le = LabelEncoder(); y = le.fit_transform(labels)
    groups = np.asarray(groups)
    return X, y, groups, le

# ====== 6) 切分（Group-aware，多 seed） ======
def group_split_cover_all(X, y, groups, require_all=True, test_rate=0.3, max_retry=300, seed0=42):
    ALL = set(np.unique(y))
    for s in range(seed0, seed0+max_retry):
        gss = GroupShuffleSplit(test_size=test_rate, n_splits=1, random_state=s)
        tr, te = next(gss.split(X, y, groups))
        if (not require_all) or (set(y[te]) == ALL):
            return tr, te, s
    # 退讓：回傳最後一次切分
    tr, te = next(GroupShuffleSplit(test_size=test_rate, n_splits=1, random_state=seed0).split(X, y, groups))
    print("[WARN] 無法覆蓋所有類別，已回退到最後一次切分；建議增加各類別的影片數。")
    return tr, te, seed0

# ====== 7) 主流程 ======
def main():
    X, y, groups, le = build_dataset()
    tr_idx, te_idx, seed = group_split_cover_all(
        X, y, groups, require_all=REQUIRE_ALL_CLASSES, test_rate=TEST_RATE, seed0=RANDOM_SEED0
    )
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    groups_tr  = groups[tr_idx]
    print(f"\n切分成功（seed={seed}）：train={len(tr_idx)}  test={len(te_idx)}")

    # --- in-air 過濾（只對進階類） ---
    CLEAR_IDX = -3  # [hip_v, wrist_v, max_clear, mean_hip, mean_w_v] → max_clear 在倒數第3
    lbl_to_idx = {n:i for i,n in enumerate(le.classes_)}
    ADV = lbl_to_idx.get("進階高壓發球", None)
    if FILTER_ADV_IN_AIR and ADV is not None:
        adv_mask = (y_tr == ADV)
        if np.sum(adv_mask) > 12:
            thr = float(np.quantile(X_tr[adv_mask, CLEAR_IDX], 1-ADV_KEEP_TOPQ))
            keep = (~adv_mask) | (X_tr[:, CLEAR_IDX] >= thr)
            print(f"(info) in-air 門檻 = {thr:.3f}，進階幀: {np.sum(adv_mask)} → {np.sum(adv_mask & (X_tr[:, CLEAR_IDX] >= thr))}")
            X_tr, y_tr, groups_tr = X_tr[keep], y_tr[keep], groups_tr[keep]

    # --- 訓練 ---
    base = BalancedRandomForestClassifier(**BASE_PARAMS)
    if not USE_GRID:
        clf = base.fit(X_tr, y_tr)
        best_params = BASE_PARAMS
        print("(info) 使用基底參數訓練（不做 GridSearch）")
    else:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42) if _HAS_SGK else GroupKFold(n_splits=3)
        grid = GridSearchCV(base, GRID_PARAMS, scoring="f1_macro", n_jobs=-1, cv=cv, verbose=1, error_score=np.nan)
        grid.fit(X_tr, y_tr, groups=groups_tr)
        clf = grid.best_estimator_
        best_params = grid.best_params_
        print("Best params:", best_params)

    # --- 評估（逐幀） ---
    proba = clf.predict_proba(X_te)
    y_pred = np.argmax(proba, axis=1)

    print("\n=== 原始逐幀 ===")
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

    # --- 時間平滑 ---
    if USE_SMOOTH:
        y_pred = _smooth_by_group(proba, groups[te_idx], te_idx, win=SMOOTH_WIN, hyst=SMOOTH_HYST)
        print("\n=== 時間平滑後 ===")
        print(classification_report(y_te, y_pred, target_names=le.classes_))
        print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

    # --- 嚴格 gating（可選） ---
    gap_thr = 0.0
    if USE_GATING:
        nonadv = X_tr[:, CLEAR_IDX] if ADV is None else X_tr[y_tr != ADV, CLEAR_IDX]
        gap_thr = float(np.quantile(nonadv, GAP_Q)) if len(nonadv) else 0.0
        y_pred = _gate_adv_strict(proba, X_te, y_pred, ADV, tau=GATE_TAU, gap_thr=gap_thr, margin=MARGIN_MIN, min_run=MIN_RUN)
        print("\n=== 平滑 + 嚴格 gating 後 ===")
        print(f"(info) τ={GATE_TAU}, GAP_THR={gap_thr:.4f}, margin={MARGIN_MIN}, min_run={MIN_RUN}")
        print(classification_report(y_te, y_pred, target_names=le.classes_))
        print("混淆矩陣:\n", confusion_matrix(y_te, y_pred))

    # --- 儲存 ---
    out_path = "20251008softtennis_pose_brf_auto_strong_opt.pkl"
    joblib.dump({"model": clf, "label_encoder": le}, out_path, compress=3)
    meta = {
        "best_params": best_params,
        "use_grid": USE_GRID,
        "use_smooth": USE_SMOOTH,
        "use_gating": USE_GATING,
        "gap_thr": gap_thr,
        "classes": list(le.classes_),
        "feature_dim": int(X.shape[1]),
        "test_rate": TEST_RATE,
        "seed": seed
    }
    Path("meta.json").write_text(jsonlib.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已存模型：{out_path}\n✅ meta：meta.json")

# ====== 8) 平滑 / gating 函式 ======
def _smooth_by_group(proba, folders, te_indices, win=15, hyst=5):
    y_pred = np.argmax(proba, axis=1).copy()
    if win<=1 and hyst<=1: return y_pred
    out = y_pred.copy()
    for f in np.unique(folders):
        idx = np.where(folders==f)[0]
        order = idx[np.argsort(te_indices[idx])]
        P = proba[order]
        T,C = P.shape
        ker = np.ones(win, dtype=np.float32)/win if win>1 else np.array([1.0], dtype=np.float32)
        P_s = np.zeros_like(P)
        for c in range(C): P_s[:,c] = np.convolve(P[:,c], ker, mode='same')
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

def _gate_adv_strict(proba, X_te, pred_in, adv_idx, tau=0.90, gap_thr=0.0, margin=0.12, min_run=12):
    out = pred_in.copy()
    if adv_idx is None or np.sum(out==adv_idx)==0: return out
    p_adv = proba[np.arange(len(out)), adv_idx]
    p2    = np.partition(proba, -2, axis=1)[:, -2]
    margin_ok = (p_adv - p2) >= margin
    CLEAR_IDX = -3
    clearance = X_te[:, CLEAR_IDX]
    keep = (p_adv>=tau) & (clearance>=gap_thr) & margin_ok
    to_demote = (out==adv_idx) & (~keep)
    if np.any(to_demote):
        sec = np.argsort(proba[to_demote], axis=1)[:, -2]
        out[to_demote] = sec
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

if __name__ == "__main__":
    main()
