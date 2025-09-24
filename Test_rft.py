# -*- coding: utf-8 -*-
"""
Super-simple test script:
- 設定 VIDEO / MODEL 後直接執行（或用兩個位置參數：video model）
- MediaPipe 擷取姿態 → 特徵 → RandomForest 推論
- 中文疊字改用 Pillow（避免 cv2.putText 變 ????）
- 加入啟動診斷與錯誤輸出，避免程式悄悄結束
"""
from __future__ import annotations
import sys, math, os, traceback
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import joblib

# Pillow for 中文文字
from PIL import Image, ImageDraw, ImageFont

# 供輸出 .xlsx
try:
    import pandas as pd
except Exception:
    sys.exit("請先安裝：pip install pandas openpyxl")

# ======= 這兩行請改成你的路徑（若不用位置參數） =======
VIDEO = Path(r"D:\Special topic data collection(1)\videos\IMG_8196.MOV")
MODEL = Path("save_all_pose_rft_20250917.pkl")
# =====================================================

# Quick toggles
SAVE_CSV    = False                          # ← 預設關閉 CSV
CSV_PATH    = Path("IMG_8196_demo.csv")
SAVE_XLSX   = True                           # ← 輸出 Excel
XLSX_PATH   = Path("IMG_8196_demo.xlsx")
SAVE_VIDEO  = True
VIDEO_OUT   = Path("IMG_8196_demo.mp4")

STRIDE     = 2
SMOOTH     = 5
MIN_DET    = 0.5
WRIST_IDX_OVERRIDE = None

# 中文字型（Windows）
FONT_PATH = r"C:\Windows\Fonts\msjh.ttc"
FONT_SIZE = 42

# ───────── 啟動診斷 ─────────
def _boot_check() -> bool:
    print(f"[BOOT] Python : {sys.executable}", flush=True)
    print(f"[BOOT] Script : {__file__}", flush=True)
    print(f"[BOOT] VIDEO  : {VIDEO}", flush=True)
    print(f"[BOOT] MODEL  : {MODEL}", flush=True)
    ok = True
    if not Path(VIDEO).exists():
        print("[ERROR] 找不到影片檔，請檢查 VIDEO 路徑。", flush=True)
        ok = False
    if not Path(MODEL).exists():
        print("[ERROR] 找不到模型檔，請檢查 MODEL 路徑。", flush=True)
        ok = False
    return ok

# ───────── MediaPipe Pose ─────────
print("[BOOT] 匯入 mediapipe…", flush=True)
try:
    import mediapipe as mp
except Exception:
    print("[ERROR] 無法匯入 mediapipe。請安裝：pip install mediapipe==0.10.14", flush=True)
    raise
print("[OK] mediapipe 匯入完成", flush=True)
mp_pose = mp.solutions.pose

ANGLE_TRIPLETS: List[Tuple[int,int,int]] = [
    (11, 13, 15), (12, 14, 16),
    (23, 25, 27), (24, 26, 28),
    (15, 13, 11), (16, 14, 12),
    (12, 24, 26), (11, 23, 25)
]

@dataclass
class FeatConfig:
    use_z: bool = True
    win_size: int = 5
    wrist_idx: int = 16

def _angle(kps: List[Dict[str,float]], a: int, b: int, c: int) -> float:
    ax, ay = kps[a]['x'], kps[a]['y']
    bx, by = kps[b]['x'], kps[b]['y']
    cx, cy = kps[c]['x'], kps[c]['y']
    v1, v2 = (ax-bx, ay-by), (cx-bx, cy-by)
    dot  = v1[0]*v2[0] + v1[1]*v2[1]
    norm = (math.hypot(*v1) * math.hypot(*v2)) + 1e-6
    return math.acos(max(-1.0, min(1.0, dot/norm)))

def frame_basic_feat(kps: List[Dict[str,float]], cfg: FeatConfig):
    feat: List[float] = []
    for p in kps:
        feat.extend([p['x'], p['y']])
        if cfg.use_z:
            feat.append(p.get('z', 0.0))
    feat.extend([_angle(kps, *tri) for tri in ANGLE_TRIPLETS])
    left_ankle_y  = kps[31]['y']
    right_ankle_y = kps[32]['y']
    hip_y         = (kps[23]['y'] + kps[24]['y']) / 2.0
    feat.extend([left_ankle_y, right_ankle_y, hip_y,
                 hip_y - left_ankle_y, hip_y - right_ankle_y])
    wrist_y = kps[cfg.wrist_idx]['y']
    return feat, hip_y, left_ankle_y, right_ankle_y, wrist_y

def landmarks_to_kps(landmarks):
    if landmarks is None:
        return None
    lm = landmarks.landmark
    if not lm or len(lm) < 33:
        return None
    return [{'x': float(lm[i].x), 'y': float(lm[i].y), 'z': float(getattr(lm[i],'z',0.0))} for i in range(33)]

def run(video_path: Path, model_path: Path):
    payload = joblib.load(model_path)
    clf = payload['model']
    le  = payload['label_encoder']
    cfg_dict = payload.get('config') or {}
    cfg = FeatConfig(**cfg_dict) if cfg_dict else FeatConfig()
    if WRIST_IDX_OVERRIDE is not None:
        cfg.wrist_idx = WRIST_IDX_OVERRIDE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"無法開啟影片: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if SAVE_VIDEO:
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps/max(1,STRIDE), (W, H))

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        print(f"[WARN] 載入字型失敗：{e}，改用預設字型（可能無法顯示中文）", flush=True)
        font = ImageFont.load_default()

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        enable_segmentation=False, min_detection_confidence=MIN_DET,
                        min_tracking_confidence=0.5)

    hip_hist, lank_hist, rank_hist, wrist_hist = (deque(maxlen=cfg.win_size) for _ in range(4))
    preds: List[str] = []
    confs: List[float] = []

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 若需要旋轉畫面，可啟用下一行：
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if STRIDE > 1 and (idx % STRIDE != 0):
                idx += 1
                if writer is not None:
                    writer.write(frame)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            kps = landmarks_to_kps(res.pose_landmarks)
            if kps is None:
                label = preds[-1] if preds else 'NA'
                conf  = confs[-1] if confs else 0.0
            else:
                basic, hip_y, lank_y, rank_y, wrist_y = frame_basic_feat(kps, cfg)
                if hip_hist:
                    hip_v   = hip_y  - hip_hist[-1]
                    lank_v  = lank_y - lank_hist[-1]
                    rank_v  = rank_y - rank_hist[-1]
                    wrist_v = wrist_y - wrist_hist[-1]
                else:
                    hip_v = lank_v = rank_v = wrist_v = 0.0
                hip_hist.append(hip_y); lank_hist.append(lank_y)
                rank_hist.append(rank_y); wrist_hist.append(wrist_y)
                max_jump_l = max((h-l) for h,l in zip(hip_hist, lank_hist))
                max_jump_r = max((h-r) for h,r in zip(hip_hist, rank_hist))
                mean_hip   = sum(hip_hist)/len(hip_hist)
                mean_w_v   = (sum(wrist_hist[i]-wrist_hist[i-1] for i in range(1,len(wrist_hist)))
                              / max(1, len(wrist_hist)-1))
                feat = np.asarray(basic + [hip_v,lank_v,rank_v,wrist_v,max_jump_l,max_jump_r,mean_hip,mean_w_v],
                                   dtype='float32').reshape(1,-1)
                proba = clf.predict_proba(feat)[0]
                y_idx = int(np.argmax(proba))
                label = str(le.inverse_transform([y_idx])[0])
                conf  = float(proba[y_idx])

            preds.append(label); confs.append(conf)

            # 多數決平滑
            if SMOOTH and len(preds) >= SMOOTH:
                from collections import Counter as C
                label = C(preds[-SMOOTH:]).most_common(1)[0][0]

            # 疊字影片（Pillow 支援中文）
            if writer is not None:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img, "RGBA")
                txt = f"{label} ({conf:.2f})"
                text_w = int(draw.textlength(txt, font=font)) + 24
                text_h = FONT_SIZE + 18
                draw.rectangle((10, 10, 10 + text_w, 10 + text_h), fill=(0, 0, 0, 160))
                draw.text((20, 16), txt, font=font, fill=(255, 255, 255, 255))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                writer.write(frame)

            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        pose.close()

    # final result
    if preds:
        mv = Counter(preds).most_common(1)[0]
        print(f"片段多數類別: {mv[0]} (count={mv[1]}/{len(preds)})", flush=True)
    else:
        print("影片內沒有有效的姿態偵測結果。", flush=True)

    # －－－ 輸出區：Excel / CSV －－－
    # 準備表格資料
    frame_indices = [i*STRIDE for i in range(len(preds))]

    if SAVE_XLSX:
        df = pd.DataFrame({
            "frame_index": frame_indices,
            "pred": preds,
            "confidence": confs
        })
        df.to_excel(XLSX_PATH, index=False)   # 需要 openpyxl
        print(f"已輸出 Excel： {XLSX_PATH}", flush=True)

    if SAVE_CSV:
        import csv
        with open(CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["frame_index","pred","confidence"])
            for i,(lab,c) in enumerate(zip(preds,confs)):
                w.writerow([frame_indices[i], lab, c])
        print(f"已輸出逐影格結果（CSV）： {CSV_PATH}", flush=True)

if __name__ == '__main__':
    try:
        if len(sys.argv) >= 3:
            VIDEO = Path(sys.argv[1])
            MODEL = Path(sys.argv[2])

        if not _boot_check():
            sys.exit(1)

        print("[RUN] 開始執行…", flush=True)
        run(VIDEO, MODEL)
        print("[DONE] 執行完成", flush=True)
    except SystemExit as e:
        print(f"[SystemExit] {e}", flush=True)
        raise
    except Exception:
        print("[EXCEPTION] 發生未預期錯誤：", flush=True)
        traceback.print_exc()
        try:
            input("↑ 上面是錯誤詳情，按 Enter 關閉…")
        except Exception:
            pass
