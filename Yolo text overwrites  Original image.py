import cv2
import os
from pathlib import Path

def overlay_annotation(image, annotation_path):
    """
    讀取標註檔，將 YOLOv8-Pose 格式標註疊加到圖片上
    標註格式（單行）：
    <class_id> <bbox_center_x> <bbox_center_y> <bbox_width> <bbox_height>
    <kpt0_x> <kpt0_y> <kpt0_visibility> ... <kptN_x> <kptN_y> <kptN_visibility>
    (所有數值皆為 normalized 值，0 ~ 1)
    """
    img_h, img_w, _ = image.shape

    with open(annotation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 轉換成數值列表
        values = list(map(float, line.split()))
        if len(values) < 5:
            print("標註資料不足：", annotation_path)
            continue

        # 解析前 5 個數值：class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height
        class_id = int(values[0])
        bbox_cx = values[1]
        bbox_cy = values[2]
        bbox_w = values[3]
        bbox_h = values[4]

        # 根據 normalized 值計算 bounding box 左上角與寬高
        x_min = int((bbox_cx - bbox_w / 2) * img_w)
        y_min = int((bbox_cy - bbox_h / 2) * img_h)
        box_w = int(bbox_w * img_w)
        box_h = int(bbox_h * img_h)

        # 畫出 bounding box 與類別標籤
        cv2.rectangle(image, (x_min, y_min), (x_min + box_w, y_min + box_h), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{class_id}", (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 後續每 3 個數值為一組：關鍵點的 (x, y, visibility)
        keypoints = values[5:]
        num_kpts = len(keypoints) // 3
        for i in range(num_kpts):
            kp_x = keypoints[i * 3]
            kp_y = keypoints[i * 3 + 1]
            kp_v = keypoints[i * 3 + 2]
            # 轉換成圖片的像素座標
            pt_x = int(kp_x * img_w)
            pt_y = int(kp_y * img_h)
            # 根據可見度決定顏色（可見度高用紅色，否則藍色）
            color = (0, 0, 255) if kp_v > 0.5 else (255, 0, 0)
            cv2.circle(image, (pt_x, pt_y), 3, color, -1)
            cv2.putText(image, str(i), (pt_x, pt_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return image

def process_folder(image_folder, annotation_folder, output_folder):
    """
    遍歷圖片資料夾，對每張圖片讀取對應的標註檔（以圖片同名但副檔名為 .txt）
    並將疊加標註後的圖片存到輸出資料夾。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        annotation_filename = os.path.splitext(filename)[0] + ".txt"
        annotation_path = os.path.join(annotation_folder, annotation_filename)

        if not os.path.exists(annotation_path):
            print(f"找不到 {filename} 對應的標註檔：{annotation_filename}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print("無法讀取圖片:", image_path)
            continue

        annotated_image = overlay_annotation(image, annotation_path)

        # 儲存疊加後的圖片，不顯示視窗
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ 已儲存疊加標註圖片：{output_path}")

if __name__ == "__main__":
    # 圖片資料夾與標註資料夾（你原本指定的絕對路徑）
    image_folder = "C:\\Users\\User\\Desktop\\picture\\IMG_8203"
    annotation_folder = "C:\\Users\\User\\Desktop\\YOLO TXT\\IMG_8203"

    # 👉 修改後：將疊加圖片輸出到 py 檔所在位置下的 output_overlay/IMG_8203
    output_folder = Path.cwd() / "YOLO_txt_output_overlay" / "IMG_8203"

    process_folder(image_folder, annotation_folder, str(output_folder))
