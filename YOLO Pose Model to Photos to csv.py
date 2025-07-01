#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fixed_detect_to_csv.py

直接用訓練好的 YOLOv8‑Pose 模型，
對「單一」圖片或影片做 33 點關鍵點偵測，
直接輸出 pixel 座標與信心度到 CSV（正確的 104 欄）。
"""

import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ─── 使用者設定 ────────────────────────────────────
MODEL_PATH  = Path('final_pose_model.pt')  # 訓練好的模型
IMAGE_PATH  = Path("C:/Users/User/Desktop/picture/IMG_8190/frame_00035.jpg")
VIDEO_PATH  = None
OUTPUT_CSV  = Path.cwd() / "csv_output" / "IMG_8190_keypoints.csv"  # ← 修改儲存位置
CONF_THRESH = 0.25
# ──────────────────────────────────────────────────

KEYPOINT_NAMES = [
    'nose','left_eye_inner','left_eye','left_eye_outer',
    'right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear','mouth_left','mouth_right',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_pinky','right_pinky',
    'left_index','right_index','left_thumb','right_thumb',
    'left_hip','right_hip','left_knee','right_knee',
    'left_ankle','right_ankle','left_heel','right_heel',
    'left_foot_index','right_foot_index'
]

def process_image(model, img_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    res = model.predict(source=img, conf=CONF_THRESH, save=False)[0]
    kp  = res.keypoints.data.cpu().numpy()  # (n_instances, 33, 3)
    
    records = []
    for inst_id, pts in enumerate(kp):
        row = {
            'source'      : img_path.name,
            'frame'       : 0,
            'instance'    : inst_id,
            'image_width' : w,
            'image_height': h
        }
        for i, name in enumerate(KEYPOINT_NAMES):
            x, y, conf = pts[i]
            # **直接就是 pixel 座標，不要再乘以 w, h**
            row[f'{name}_x']    = int(round(x))
            row[f'{name}_y']    = int(round(y))
            row[f'{name}_conf'] = float(conf)
        records.append(row)
    return records

def process_video(model, vid_path):
    cap = cv2.VideoCapture(str(vid_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame0 = cap.read()
    h, w = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    records = []
    frame_id = 0
    for _ in tqdm(range(total), desc=f'Processing {vid_path.name}'):
        ret, frame = cap.read()
        if not ret: break
        res = model.predict(source=frame, conf=CONF_THRESH, save=False)[0]
        kp  = res.keypoints.data.cpu().numpy()
        for inst_id, pts in enumerate(kp):
            row = {
                'source'      : vid_path.name,
                'frame'       : frame_id,
                'instance'    : inst_id,
                'image_width' : w,
                'image_height': h
            }
            for i, name in enumerate(KEYPOINT_NAMES):
                x, y, conf = pts[i]
                row[f'{name}_x']    = int(round(x))
                row[f'{name}_y']    = int(round(y))
                row[f'{name}_conf'] = float(conf)
            records.append(row)
        frame_id += 1
    cap.release()
    return records

def main():
    model = YOLO(str(MODEL_PATH))
    all_records = []
    if IMAGE_PATH.exists():
        all_records = process_image(model, IMAGE_PATH)
    elif VIDEO_PATH and VIDEO_PATH.exists():
        all_records = process_video(model, VIDEO_PATH)
    else:
        print("❌ 請確認 IMAGE_PATH 或 VIDEO_PATH 已正確設定。")
        return

    df = pd.DataFrame(all_records)
    print(f"欄位數：{len(df.columns)}，筆數：{len(df)}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 已輸出 pixel 座標到：{OUTPUT_CSV}")

if __name__ == '__main__':
    main()
