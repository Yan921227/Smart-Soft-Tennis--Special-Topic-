from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def run_inference_folder(model_path, folder_path, save_results=True):
    """
    使用指定的模型對資料夾中的所有圖片進行推論，並將結果儲存至指定資料夾。
    
    :param model_path: 模型權重檔案路徑 (例如 'runs/pose/train/weights/best.pt')
    :param folder_path: 存放圖片的資料夾路徑 (例如 'data/images/')
    :param save_results: 是否儲存預測結果，預設為 True
    """
    # 載入訓練好的模型
    model = YOLO(model_path)
    
    # 自訂輸出位置：放在與 py 檔案同層的 runs/pose/predict_test/
    project_dir = os.path.join(os.getcwd(), "runs", "pose")
    output_name = "predict_test"
    
    # 執行推論
    results = model.predict(
        source=folder_path,
        save=save_results,
        project=project_dir,
        name=output_name
    )
    
    # -----------------------------
    # 若想要自行讀取儲存後的圖片，再用 OpenCV 與 Matplotlib 顯示出來，可將以下程式碼解除註解。
    # -----------------------------
    """
    predict_folder = os.path.join(project_dir, output_name)
    if not os.path.exists(predict_folder):
        print("找不到預測結果資料夾：", predict_folder)
        return
    
    # 取得該資料夾中所有檔案，並依序讀取顯示
    for file_name in os.listdir(predict_folder):
        file_path = os.path.join(predict_folder, file_name)
        img = cv2.imread(file_path)
        if img is None:
            continue
        # 將 OpenCV 的 BGR 轉成 RGB 方便 Matplotlib 正確顯示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(file_name)
        plt.axis('off')
        plt.show()
    """

if __name__ == "__main__":
    # 請根據你的實際路徑修改這兩個變數
    model_path = "final_pose_model.pt"
    folder_path = "C:/Users/User/Desktop/picture/test"
    run_inference_folder(model_path, folder_path)
    # 這裡的 model_path 和 folder_path 是範例路徑，請根據實際情況修改
