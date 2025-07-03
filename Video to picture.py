import cv2
import os

def video_to_frames(video_path, output_folder, frame_interval=1):
    """
    將影片拆解成影格並儲存為圖片

    參數:
        video_path: 影片檔案路徑 (例如 "video.mp4")
        output_folder: 輸出影格圖片的資料夾 (例如 "frames")
        frame_interval: 每隔多少影格儲存一次圖片 (預設 1 表示每一張影格都存)
    """
    # 若輸出資料夾不存在，則建立之
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 開啟影片檔案
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 根據 frame_interval 判斷是否儲存
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        count += 1

    cap.release()
    print(f"總共儲存 {saved_count} 張影格。")

if __name__ == "__main__":

    video_path = "C:\\Users\\User\\Desktop\\綜合揮拍姿勢影片\\IMG_9688.MOV" # 替換成你的影片路徑
    output_folder = os.path.join(os.getcwd(), "frames", "IMG_9688")  # 指定巢狀資料夾

    frame_interval = 1
    video_to_frames(video_path, output_folder, frame_interval)
