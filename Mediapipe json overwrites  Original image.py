import cv2
import json
import os

def overlay_keypoints_on_image(image_path, json_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return
    h, w, _ = image.shape

    # 讀取 JSON 資料
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 定義一個繪製函式，根據相對座標（0~1）轉換為實際像素座標並畫圓點
    def draw_keypoints(keypoints, color, radius=3):
        for kp in keypoints:
            x = int(kp["x"] * w)
            y = int(kp["y"] * h)
            cv2.circle(image, (x, y), radius, color, -1)

    # 疊加身體、左右手與臉部的關鍵點
    if "pose" in data:
        draw_keypoints(data["pose"], color=(0, 255, 0))  # 綠色表示身體關鍵點
    if "left_hand" in data:
        draw_keypoints(data["left_hand"], color=(255, 0, 0))  # 藍色表示左手
    if "right_hand" in data:
        draw_keypoints(data["right_hand"], color=(0, 0, 255))  # 紅色表示右手
    if "face" in data:
        for detection in data["face"]:
            draw_keypoints(detection, color=(0, 255, 255))  # 黃綠色表示臉部

    # 儲存疊加後的圖片
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"已儲存疊加後的圖片至: {output_path}")

def process_folder(image_folder, json_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 遍歷圖片資料夾中所有圖片檔案（支援 .png, .jpg, .jpeg）
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            # 假設對應的 JSON 檔名與圖片檔名相同，但副檔名為 .json
            json_file = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(json_folder, json_file)
            if os.path.exists(json_path):
                print(f"處理: {filename}")
                overlay_keypoints_on_image(image_path, json_path, output_folder)
            else:
                print(f"找不到 {filename} 對應的 JSON 檔案：{json_file}")

if __name__ == "__main__":
    # 設定原始圖片所在的資料夾路徑
    image_folder = "D:\\Special topic data collection(2)\\frames\\IMG_9675"
    # 設定 JSON 標註檔所在的資料夾路徑
    json_folder = "output_json\IMG_9675"
    # 設定輸出疊加後圖片的資料夾路徑（改為與 py 檔案同層）
    output_folder = os.path.join(os.getcwd(), "output_overlay", "IMG_9675")
    
    process_folder(image_folder, json_folder, output_folder)
