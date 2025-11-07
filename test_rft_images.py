# -*- coding: utf-8 -*-
"""
使用訓練好的 Random Forest 模型辨識圖片中的動作（Backswing vs Impact）
- 讀取資料夾中的圖片，提取 MediaPipe 姿勢關鍵點
- 使用訓練好的模型進行預測
- 在圖片上標註預測結果
- 輸出標註後的圖片和 CSV 報告

pip install opencv-python mediapipe numpy joblib
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import csv
import os

# =========================
# 可調參數
# =========================
# 輸入路徑
INPUT_IMAGE_FOLDER = "frames_CapCut\\IMG_1159"  # 輸入圖片資料夾
MODEL_PATH = "rf_pose_model.pkl"                                               # 訓練好的模型路徑

# 輸出路徑
OUTPUT_IMAGE_FOLDER = "output_classified_images"                               # 輸出圖片資料夾
OUTPUT_CSV_PATH = "prediction_results_images.csv"                              # 預測結果 CSV

# 處理設定
SHOW_PREVIEW = False                                                           # 是否即時顯示每張圖片
SAVE_IMAGES = True                                                             # 是否儲存標註後的圖片
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']                           # 支援的圖片格式

# =========================
# MediaPipe 初始化
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 33 點名稱
MP_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

CLASS_NAMES = ["Backswing", "Impact"]

def extract_pose_xy66(results) -> np.ndarray:
    """從 MediaPipe 結果提取 33 個 (x,y) → 66 維；缺失填 0"""
    if results.pose_landmarks is None:
        return np.zeros(66, dtype=np.float32)
    
    landmarks = results.pose_landmarks.landmark
    arr = []
    for i in range(33):
        if i < len(landmarks):
            arr += [landmarks[i].x, landmarks[i].y]
        else:
            arr += [0.0, 0.0]
    return np.array(arr, dtype=np.float32)

def get_image_files(folder_path):
    """取得資料夾中所有圖片檔案"""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    return sorted(image_files)

def main():
    # 1) 載入訓練好的模型
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"✓ 模型載入成功：{MODEL_PATH}")
    except Exception as e:
        print(f"✗ 無法載入模型：{e}")
        return

    # 2) 取得所有圖片檔案
    image_files = get_image_files(INPUT_IMAGE_FOLDER)
    if not image_files:
        print(f"✗ 在 {INPUT_IMAGE_FOLDER} 中找不到圖片檔案")
        return
    
    total_images = len(image_files)
    print(f"✓ 找到 {total_images} 張圖片")

    # 3) 建立輸出資料夾
    if SAVE_IMAGES:
        Path(OUTPUT_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)
        print(f"✓ 輸出資料夾：{OUTPUT_IMAGE_FOLDER}")

    # 4) 準備 CSV 輸出
    csv_file = open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image_Name", "Prediction", "Confidence", "Has_Pose", "Image_Path"])

    # 5) 初始化 MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )

    print("\n開始處理圖片...")
    
    # 統計資料
    stats = {"Backswing": 0, "Impact": 0, "No Pose": 0}

    try:
        for idx, image_path in enumerate(image_files, 1):
            # 讀取圖片
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"✗ 無法讀取：{image_path.name}")
                continue

            # 轉換顏色並處理
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # 提取姿勢特徵
            pose_features = extract_pose_xy66(results)
            has_pose = results.pose_landmarks is not None

            # 預測
            prediction = -1
            confidence = 0.0
            pred_label = "No Pose"

            if has_pose:
                # 預測類別
                prediction = clf.predict([pose_features])[0]
                # 預測機率
                probabilities = clf.predict_proba([pose_features])[0]
                confidence = probabilities[prediction]
                pred_label = CLASS_NAMES[prediction]

                # 更新統計
                stats[pred_label] += 1

                # 繪製骨架
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            else:
                stats["No Pose"] += 1

            # 寫入 CSV
            csv_writer.writerow([
                image_path.name, 
                pred_label, 
                f"{confidence:.3f}", 
                has_pose,
                str(image_path)
            ])

            # 在圖片上標註預測結果
            label_text = f"{pred_label} ({confidence:.2f})" if has_pose else "No Pose Detected"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255) if prediction == 1 else (128, 128, 128)
            
            # 取得圖片尺寸來調整文字大小
            h, w = frame.shape[:2]
            font_scale = max(0.5, min(w, h) / 1000)
            thickness = max(1, int(font_scale * 2))
            
            # 繪製半透明背景
            overlay = frame.copy()
            box_width = int(w * 0.4)
            box_height = int(60 * font_scale)
            cv2.rectangle(overlay, (10, 10), (box_width, box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 繪製文字
            cv2.putText(frame, label_text, (20, int(35 * font_scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

            # 儲存標註後的圖片
            if SAVE_IMAGES:
                output_path = Path(OUTPUT_IMAGE_FOLDER) / image_path.name
                cv2.imwrite(str(output_path), frame)

            # 即時顯示
            if SHOW_PREVIEW:
                # 調整顯示大小
                max_display_height = 800
                if h > max_display_height:
                    scale = max_display_height / h
                    display_frame = cv2.resize(frame, None, fx=scale, fy=scale, 
                                             interpolation=cv2.INTER_AREA)
                else:
                    display_frame = frame
                
                cv2.imshow('Pose Classification', display_frame)
                key = cv2.waitKey(100)
                if key & 0xFF == ord('q'):
                    print("\n使用者中斷")
                    break

            # 顯示進度
            if idx % 10 == 0 or idx == total_images:
                progress = (idx / total_images) * 100
                print(f"處理進度：{idx}/{total_images} ({progress:.1f}%)")

    finally:
        # 清理資源
        csv_file.close()
        pose.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    # 顯示統計結果
    print(f"\n✓ 處理完成！")
    print(f"  預測結果：{OUTPUT_CSV_PATH}")
    if SAVE_IMAGES:
        print(f"  標註圖片：{OUTPUT_IMAGE_FOLDER}")
    print(f"\n統計結果:")
    print(f"  Backswing: {stats['Backswing']} 張")
    print(f"  Impact: {stats['Impact']} 張")
    print(f"  No Pose: {stats['No Pose']} 張")
    print(f"  總計: {sum(stats.values())} 張")

if __name__ == "__main__":
    main()
