import cv2
import mediapipe as mp
import os
import json

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None

    # 調整圖片大小（寬度設為 640，高度依比例縮放）
    resize_width = 640
    h, w, _ = image.shape
    aspect_ratio = h / w
    new_height = int(resize_width * aspect_ratio)
    image = cv2.resize(image, (resize_width, new_height))
    
    # 初始化 Mediapipe 模組
    mp_holistic = mp.solutions.holistic
    mp_face_detection = mp.solutions.face_detection

    # 使用 Holistic 偵測身體與手部
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        results_holistic = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 使用 Face Detection 偵測臉部6個關鍵點
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:
        results_face = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 整理關鍵點資料
    keypoints_data = {}
    # (A) 身體關鍵點 (33點)
    if results_holistic.pose_landmarks:
        keypoints_data["pose"] = []
        for idx, landmark in enumerate(results_holistic.pose_landmarks.landmark):
            keypoints_data["pose"].append({
                "index": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
    # (B) 左手 (21點)
    if results_holistic.left_hand_landmarks:
        keypoints_data["left_hand"] = []
        for idx, landmark in enumerate(results_holistic.left_hand_landmarks.landmark):
            keypoints_data["left_hand"].append({
                "index": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
    # (C) 右手 (21點)
    if results_holistic.right_hand_landmarks:
        keypoints_data["right_hand"] = []
        for idx, landmark in enumerate(results_holistic.right_hand_landmarks.landmark):
            keypoints_data["right_hand"].append({
                "index": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })
    # (D) 臉部 (6點，使用 Face Detection)
    if results_face.detections:
        keypoints_data["face"] = []
        for detection in results_face.detections:
            face_kps = []
            for idx, kp in enumerate(detection.location_data.relative_keypoints):
                face_kps.append({
                    "index": idx,
                    "x": kp.x,
                    "y": kp.y
                })
            keypoints_data["face"].append(face_kps)
    
    return keypoints_data

def process_folder_separate(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍歷資料夾中所有圖片檔（支援 .png, .jpg, .jpeg）
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"正在處理: {filename}")
            data = process_image(image_path)
            if data is not None:
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_folder, f"{base_name}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"已儲存 {filename} 的關鍵點資料到: {output_file}")

if __name__ == '__main__':
    # 請將 input_folder 改成你的圖片資料夾路徑
    input_folder = "D:\\Special topic data collection(2)\\frames\\IMG_9673"
    # 請將 output_folder 改成你想儲存結果的資料夾路徑（這裡改為 py 檔案所在目錄中的子資料夾）
    output_folder = os.path.join(os.getcwd(), "output_json", "IMG_9673")
    process_folder_separate(input_folder, output_folder)
