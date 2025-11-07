# -*- coding: utf-8 -*-
"""
使用訓練好的 Random Forest 模型辨識影片中的動作（Backswing vs Impact）
- 讀取影片，逐幀提取 MediaPipe 姿勢關鍵點
- 使用訓練好的模型進行預測
- 在影片上標註預測結果
- 輸出標註後的影片

pip install opencv-python mediapipe numpy joblib
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path

# =========================
# 可調參數
# =========================
# 輸入路徑
INPUT_VIDEO_PATH = "C:\\Users\\User\\Desktop\\IMG_1159.mp4"          # 輸入影片路徑
MODEL_PATH = "rf_pose_model.pkl"              # 訓練好的模型路徑

# 輸出路徑
OUTPUT_VIDEO_PATH = "output_classified_正拍.m\p4"   # 輸出影片路徑
OUTPUT_CSV_PATH = "prediction_results_正拍.csv"    # 預測結果 CSV

# 顯示設定
SHOW_PREVIEW = True                           # 是否即時顯示預測結果
CONFIDENCE_THRESHOLD = 0.6                    # 信心度閾值（可選）
ROTATE_VIDEO = False                           # 是否旋轉影片（順時針 90 度）
PREVIEW_SCALE = 0.5                           # 預覽視窗縮放比例（0.5 = 50%）

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

def main():
    # 1) 載入訓練好的模型
    try:
        clf = joblib.load(MODEL_PATH)
        print(f"✓ 模型載入成功：{MODEL_PATH}")
    except Exception as e:
        print(f"✗ 無法載入模型：{e}")
        return

    # 2) 開啟輸入影片
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"✗ 無法開啟影片：{INPUT_VIDEO_PATH}")
        return

    # 取得影片資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果需要旋轉，交換寬高
    output_width = height if ROTATE_VIDEO else width
    output_height = width if ROTATE_VIDEO else height
    
    print(f"✓ 影片資訊：{width}x{height} @ {fps} FPS，共 {total_frames} 幀")
    if ROTATE_VIDEO:
        print(f"  旋轉後輸出：{output_width}x{output_height}")

    # 3) 設定輸出影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (output_width, output_height))

    # 4) 準備 CSV 輸出
    import csv
    csv_file = open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Prediction", "Confidence", "Has_Pose"])

    # 5) 初始化 MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0
    print("\n開始處理影片...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 旋轉影片（順時針 90 度）
            if ROTATE_VIDEO:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

                # 繪製骨架
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # 寫入 CSV
            csv_writer.writerow([frame_idx, pred_label, f"{confidence:.3f}", has_pose])

            # 在影片上標註預測結果
            label_text = f"{pred_label} ({confidence:.2f})" if has_pose else "No Pose Detected"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255) if prediction == 1 else (128, 128, 128)
            
            # 繪製半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 繪製文字
            cv2.putText(frame, label_text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 寫入輸出影片
            out.write(frame)

            # 即時顯示
            if SHOW_PREVIEW:
                # 縮小預覽視窗
                preview_frame = cv2.resize(frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, 
                                          interpolation=cv2.INTER_AREA)
                cv2.imshow('Pose Classification', preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n使用者中斷")
                    break

            # 顯示進度
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"處理進度：{frame_idx}/{total_frames} ({progress:.1f}%)")

    finally:
        # 清理資源
        cap.release()
        out.release()
        csv_file.close()
        pose.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    print(f"\n✓ 處理完成！")
    print(f"  輸出影片：{OUTPUT_VIDEO_PATH}")
    print(f"  預測結果：{OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
