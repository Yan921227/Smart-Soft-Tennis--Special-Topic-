# Model Accuracy.py
from ultralytics import YOLO

def evaluate():
    model = YOLO("final_pose_models.pt")
    metrics_test = model.val(
        data='C:/Users/User/py/YAML/pose_data.yaml',
        imgsz=1280,
        conf=0.1,
        split="test"
    )
    print("=== Test Set Performance ===")
    print(metrics_test)

if __name__ == "__main__":
    evaluate()
