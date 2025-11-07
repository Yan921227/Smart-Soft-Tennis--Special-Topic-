# -*- coding: utf-8 -*-
"""
使用訓練好的 Random Forest 模型辨識影片中的動作（Backswing vs Impact）
- 讀取影片，逐幀提取 MediaPipe 姿勢關鍵點
- 使用訓練好的模型進行預測
- 在每一幀上標註預測結果
- 輸出為圖片序列（不是影片）

pip install opencv-python mediapipe numpy joblib
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import csv

# =========================
# 可調參數
# =========================
# 輸入路徑
INPUT_VIDEO_PATH = "C:\\Users\\User\\Desktop\\IMG_1159.mp4"          # 輸入影片路徑
MODEL_PATH = "rf_pose_model.pkl"                                     # 訓練好的模型路徑

# 輸出路徑
OUTPUT_IMAGE_FOLDER = "output_frames_forehand"                               # 輸出圖片資料夾
OUTPUT_CSV_PATH = "prediction_results_forehand.csv"                          # 預測結果 CSV

# 處理設定
SHOW_PREVIEW = True                           # 是否即時顯示預測結果
ROTATE_VIDEO = False                          # 是否旋轉影片（順時針 90 度）
PREVIEW_SCALE = 0.5                           # 預覽視窗縮放比例（0.5 = 50%）
FRAME_SKIP = 1                                # 每隔幾幀處理一次（1 = 不跳幀，2 = 每兩幀處理一次）

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
    
    print(f"✓ 影片資訊：{width}x{height} @ {fps} FPS，共 {total_frames} 幀")

    # 3) 建立輸出資料夾
    output_folder_path = Path(OUTPUT_IMAGE_FOLDER)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 輸出資料夾：{output_folder_path.absolute()}")
    
    # 測試寫入權限
    test_file = output_folder_path / "test.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ 資料夾寫入權限正常")
    except Exception as e:
        print(f"✗ 資料夾寫入權限錯誤：{e}")
        return

    # 4) 準備 CSV 輸出
    csv_file = open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Image_Name", "Prediction", "Confidence", "Has_Pose"])

    # 5) 初始化 MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0
    saved_count = 0
    print("\n開始處理影片...")
    
    # 統計資料
    stats = {"Backswing": 0, "Impact": 0, "No Pose": 0}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            
            # 跳幀處理
            if frame_idx % FRAME_SKIP != 0:
                continue

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
                
                # 更新統計
                stats[pred_label] += 1
            else:
                stats["No Pose"] += 1
            
            # 建立輸出幀的副本（在繪製之前）
            output_frame = frame.copy()
            
            # 繪製骨架（在副本上）
            if has_pose:
                mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # 在圖片上標註預測結果
            label_text = f"{pred_label} ({confidence:.2f})" if has_pose else "No Pose Detected"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255) if prediction == 1 else (128, 128, 128)
            
            # 繪製半透明背景
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
            output_frame = cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0)
            
            # 繪製文字
            cv2.putText(output_frame, label_text, (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(output_frame, f"Frame: {frame_idx}/{total_frames}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 儲存圖片（使用幀編號命名）
            image_name = f"frame_{frame_idx:05d}.jpg"
            output_path = Path(OUTPUT_IMAGE_FOLDER) / image_name
            success = cv2.imwrite(str(output_path), output_frame)
            if success:
                saved_count += 1
            else:
                print(f"✗ 無法儲存圖片：{output_path}")

            # 寫入 CSV
            csv_writer.writerow([frame_idx, image_name, pred_label, f"{confidence:.3f}", has_pose])

            # 即時顯示
            if SHOW_PREVIEW:
                # 縮小預覽視窗
                preview_frame = cv2.resize(output_frame, None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE, 
                                          interpolation=cv2.INTER_AREA)
                cv2.imshow('Pose Classification', preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n使用者中斷")
                    break

            # 顯示進度
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"處理進度：{frame_idx}/{total_frames} ({progress:.1f}%) - 已儲存 {saved_count} 張")

    finally:
        # 清理資源
        cap.release()
        csv_file.close()
        pose.close()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    print(f"\n✓ 處理完成！")
    print(f"  儲存圖片：{saved_count} 張 → {OUTPUT_IMAGE_FOLDER}")
    print(f"  預測結果：{OUTPUT_CSV_PATH}")
    print(f"\n統計結果:")
    print(f"  Backswing: {stats['Backswing']} 幀")
    print(f"  Impact: {stats['Impact']} 幀")
    print(f"  No Pose: {stats['No Pose']} 幀")
    print(f"  總計: {sum(stats.values())} 幀")

if __name__ == "__main__":
    main()
