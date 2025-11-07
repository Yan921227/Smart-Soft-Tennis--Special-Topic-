# -*- coding: utf-8 -*-
import json, math
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
from pathlib import Path

# ================================
# 1) 檔案路徑設定
# ================================
# 輸入路徑
data_path  = Path("output_json\\IMG_1163\\frame_02293.json")
image_path = Path("D:\\Special topic data collection(3)\\frames\\IMG_1163\\frame_02293.jpg")

# 輸出路徑
output_image_path = "IMG_1163_frame_02293_angle_letters.jpg"
output_csv_path = "IMG_1163_frame_02293_angle_letters.csv"

# ================================
# 2) 讀圖片
# ================================
cv_image = cv2.imread(str(image_path))
if cv_image is None:
    raise FileNotFoundError(f"無法讀取圖片：{image_path}")
H, W = cv_image.shape[:2]

def to_px(x, y):
    x = float(x); y = float(y)
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return int(round(x * W)), int(round(y * H))
    return int(round(x)), int(round(y))

# MediaPipe Pose 33 點名稱
MP_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER","RIGHT_EYE",
    "RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT","MOUTH_RIGHT","LEFT_SHOULDER",
    "RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW","LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY",
    "RIGHT_PINKY","LEFT_INDEX","RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP",
    "RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX"
]

# ================================
# 3) 從 JSON 取 landmarks
# ================================
import json
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

pose_list = data.get("pose", [])
if not pose_list or len(pose_list) < 33:
    raise ValueError("JSON.pose 缺少 33 個姿勢點。")

landmarks = {}
for item in pose_list:
    idx = int(item["index"])
    if 0 <= idx < 33:
        name = MP_NAMES[idx]
        landmarks[name] = to_px(item["x"], item["y"])

# ================================
# 4) 三點夾角（以 B 為頂點）
# ================================
def calculate_angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0]*BC[0] + BA[1]*BC[1]
    mag_BA = math.hypot(*BA)
    mag_BC = math.hypot(*BC)
    if mag_BA == 0 or mag_BC == 0:
        return None
    cos_t = max(min(dot / (mag_BA * mag_BC), 1.0), -1.0)
    return math.degrees(math.acos(cos_t))

# ================================
# 5) 要計算的關節（三點）
# ================================
joints = {
    "Right Elbow":    ("RIGHT_SHOULDER",  "RIGHT_ELBOW",   "RIGHT_WRIST"),
    "Left Elbow":   ("LEFT_SHOULDER", "LEFT_ELBOW",  "LEFT_WRIST"),
    "Right Shoulder": ("RIGHT_ELBOW",     "RIGHT_SHOULDER","RIGHT_HIP"),
    "Left Knee":      ("LEFT_HIP",        "LEFT_KNEE",     "LEFT_ANKLE"),
    "Right Knee": ("RIGHT_HIP",  "RIGHT_KNEE",  "RIGHT_ANKLE"),
    
}
# joints = {
#     # 肘部角度（手臂屈曲）
#     "Left Elbow":   ("LEFT_SHOULDER", "LEFT_ELBOW",  "LEFT_WRIST"),
#     "Right Elbow":  ("RIGHT_SHOULDER","RIGHT_ELBOW", "RIGHT_WRIST"),
#
#     # 肩部角度（肩關節屈伸／外展）
#     #   以「肘—肩—髖」三點計算，可反映手臂相對於軀幹的抬高或下垂角度
#     "Left Shoulder":  ("LEFT_ELBOW",  "LEFT_SHOULDER",  "LEFT_HIP"),
#     "Right Shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
#
#     # 膝蓋角度（腿部屈曲）
#     "Left Knee":  ("LEFT_HIP",   "LEFT_KNEE",   "LEFT_ANKLE"),
#     "Right Knee": ("RIGHT_HIP",  "RIGHT_KNEE",  "RIGHT_ANKLE"),
#
#     # 髖部角度（髖關節屈伸）
#     #   以「肩—髖—膝」三點計算，可反映腿伸向前或向後的角度
#     "Left Hip":  ("LEFT_SHOULDER",  "LEFT_HIP",   "LEFT_KNEE"),
#     "Right Hip": ("RIGHT_SHOULDER", "RIGHT_HIP",  "RIGHT_KNEE"),
#
#     # 踝關節角度（足部背屈／跖屈）
#     #   MediaPipe Pose 也提供「腳跟」(LEFT_HEEL / RIGHT_HEEL)，作為足跟的參考點
#     "Left Ankle":  ("LEFT_KNEE",   "LEFT_ANKLE",  "LEFT_HEEL"),
#     "Right Ankle": ("RIGHT_KNEE",  "RIGHT_ANKLE", "RIGHT_HEEL"),
#
#     # 腕關節角度（手腕屈伸）
#     #   用「肘—腕—食指末端 (INDEX)」來估算手腕角度
#     "Left Wrist":  ("LEFT_ELBOW",  "LEFT_WRIST",  "LEFT_INDEX"),
#     "Right Wrist": ("RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),
# }

# 圖上用字母標記；終端機印數值
labels_for_joint = {
    "Right Elbow": "A",
    "Left Elbow": "B",
    "Right Shoulder": "C",
    "Left Knee": "D",
    "Right Knee": "E",
}

# 角度文字偏移
text_offset = {
    "Right Elbow":    (15, -30),
    "Left Knee":      (-40, -20),
    "Right Shoulder": (10, 20),
    "Left Elbow":     (15, -30),
    "Right Knee":    (-40, -20),
}

# ================================
# 6) 繪製字母標記，並在終端機列印角度
# ================================
img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(pil_img)

# 跨平台字型
font = None
for fp in ("C:/Windows/Fonts/arial.ttf",
           "/System/Library/Fonts/Supplemental/Arial.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    try:
        font = ImageFont.truetype(fp, 32)
        break
    except Exception:
        pass
if font is None:
    font = ImageFont.load_default()

# 先算角度並印出
angles_out = []  # 收集要印的文字
for joint_name, (pA, pB, pC) in joints.items():
    if pA in landmarks and pB in landmarks and pC in landmarks:
        A, B, C = landmarks[pA], landmarks[pB], landmarks[pC]
        ang = calculate_angle(A, B, C)
        label = labels_for_joint.get(joint_name, "?")
        if ang is None:
            angles_out.append(f"{label} ({joint_name}): 無法計算（向量長度為 0）")
        else:
            angles_out.append(f"{label} ({joint_name}): {ang:.1f}°")
    else:
        angles_out.append(f"{labels_for_joint.get(joint_name,'?')} ({joint_name}): 缺少關節點 {pA},{pB},{pC}")

print("各角度:")
for line in angles_out:
    print("  " + line)

# 畫點與「字母」標記(不畫數字角度)
r = 8
for joint_name, (pA, pB, pC) in joints.items():
    if pA in landmarks and pB in landmarks and pC in landmarks:
        A, B, C = landmarks[pA], landmarks[pB], landmarks[pC]
        for P in (A, B, C):
            draw.ellipse([P[0]-r, P[1]-r, P[0]+r, P[1]+r], fill=(255, 0, 0))
        dx, dy = text_offset.get(joint_name, (0, -35))
        label = labels_for_joint.get(joint_name, "?")
        draw.text((B[0] + dx, B[1] + dy), label, font=font, fill=(255, 0, 0))
    else:
        print(f"[警告] JSON 中缺少 {joint_name} 所需的點：{pA}, {pB}, {pC}")

# 存檔
annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
cv2.imwrite(output_image_path, annotated)
print(f"已輸出圖片：{output_image_path}")

# 輸出角度到 CSV
with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    # 寫入標題
    writer.writerow(["Joint", "Label", "Angle (degrees)"])
    # 寫入每個角度資料
    for joint_name, (pA, pB, pC) in joints.items():
        label = labels_for_joint.get(joint_name, "?")
        if pA in landmarks and pB in landmarks and pC in landmarks:
            A, B, C = landmarks[pA], landmarks[pB], landmarks[pC]
            ang = calculate_angle(A, B, C)
            if ang is None:
                writer.writerow([joint_name, label, "N/A"])
            else:
                writer.writerow([joint_name, label, f"{ang:.1f}"])
        else:
            writer.writerow([joint_name, label, "Missing Points"])
print(f"已輸出 CSV：{output_csv_path}")
