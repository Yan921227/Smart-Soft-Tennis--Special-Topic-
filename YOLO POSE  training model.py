# # yolov8_pose_train.py

import os
from ultralytics import YOLO

def main():
    # 模型權重檔案：可以選擇不同大小的模型（例如 yolov8n-pose.pt、yolov8s-pose.pt 等）
    model_path = 'yolov8s-pose.pt'
    
    # 資料集設定檔（請確認此路徑與檔名正確）
    data_config = 'C:/Users/User/py/YAML/pose_data.yaml'
    
    # 訓練參數
    epochs = 100        # 訓練週期數
    img_size = 1280      # 輸入圖片尺寸，可根據需求調整

    # 顯示目前工作目錄（確認資料路徑是否正確）
    print("目前工作目錄：", os.getcwd())

    # 載入預訓練的 YOLOv8 Pose 模型
    model = YOLO(model_path)

    # 開始訓練
    # resume=True 表示如果有先前儲存的 checkpoint，則會接續上次的訓練進度
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size)
    
    # 儲存訓練完成後的模型（此方法會將模型權重儲存到指定路徑）
    save_path = 'final_pose_models.pt'
    model.save(save_path)
    
    print("訓練完成並已儲存模型至：", save_path)

if __name__ == '__main__':
    main()




# import numpy
# print(numpy.__version__)


# import torch
# import torchvision
# print(torch.__version__)       # 應該會顯示類似 2.0.1+cu118
# print(torchvision.__version__) # 應該會顯示 0.15.2

# import torch
# print("CUDA 是否可用：", torch.cuda.is_available())
# print("PyTorch 使用的 CUDA 版本：", torch.version.cuda)
# print("目前 GPU 名稱：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無 GPU")


# import torch
# print("CUDA is available:", torch.cuda.is_available())
# x = torch.rand(3, 3).cuda()  # 將張量放到 GPU
# print(x)
