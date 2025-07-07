import os
import json

def convert_json_to_yolopose(input_json_path, output_txt_path):
    """
    讀取單一 JSON 檔，並將其中的 "pose" 資料轉換成 YOLOv8-Pose 格式，
    寫入對應的 TXT 標註檔案。
    若無 'pose' 資料或 'pose' 為空，則仍產生一個空的檔案。
    """
    # 讀取 JSON 檔案
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 若有 pose 資料且資料不為空，則進行轉換
    if "pose" in data and len(data["pose"]) > 0:
        pose_data = data["pose"]
        # 計算 bounding box：取所有關鍵點的 x 與 y 值（假設均為相對座標 0~1）
        xs = [kp["x"] for kp in pose_data]
        ys = [kp["y"] for kp in pose_data]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # 組成所有關鍵點資料 (x, y, visibility)
        keypoints_list = []
        for kp in pose_data:
            keypoints_list.extend([kp["x"], kp["y"], kp["visibility"]])
    
        # 組成標註行
        # 預設 class_id 為 0 (人)
        # 轉換後一行資料包含：class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height, 加上每個關鍵點資料
        line_items = [0, center_x, center_y, bbox_width, bbox_height] + keypoints_list
        # 將所有數值轉為字串，以空白分隔
        line_str = " ".join(str(item) for item in line_items)
        
        # 寫入標註檔（單行）
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(line_str + "\n")
        
        print(f"已轉換 {input_json_path} 至 {output_txt_path}")
    else:
        # 若無 'pose' 資料或 'pose' 為空，則產生一個空檔以保持標註檔完整性
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"{input_json_path} 無 'pose' 資料或資料為空，已產生空的標註檔 {output_txt_path}")

def process_json_folder(input_folder, output_folder):
    """
    遍歷指定資料夾中所有 JSON 檔案，
    依序轉換成 YOLOv8-Pose 格式的標註 TXT 檔，
    並存到輸出資料夾中。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.json'):
            input_json_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_txt_path = os.path.join(output_folder, base_name + ".txt")
            convert_json_to_yolopose(input_json_path, output_txt_path)

if __name__ == '__main__':
    # 請依照你的路徑修改：
    # 輸入：獨立 JSON 標註檔所在資料夾（例如，每張圖片一個 JSON 檔）
    input_json_folder = "output_json\IMG_9672"
    # 輸出：轉換後的 YOLO 格式標註檔存放的資料夾（使用 py 檔目錄下的相對路徑）
    output_yolo_folder = os.path.join(os.getcwd(), "yolo_txt", "IMG_9672")
    
    process_json_folder(input_json_folder, output_yolo_folder)
    print("所有 JSON 檔案轉換完成！")
    # 這段程式碼會將指定資料夾中的所有 JSON 檔案轉換為 YOLOv8-Pose 格式的 TXT 檔案。
