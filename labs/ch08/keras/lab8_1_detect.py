# 实验 8-1（通用版）目标检测：框出照片里的东西
# %% 第 1 步：请出现成的检测模型 YOLO
from ultralytics import YOLO                     # 先运行 !pip install ultralytics
model = YOLO("yolo11n.pt")                      # n 表示最小的版本；第一次运行会自动下载约 5 MB

# %% 第 2 步：检测一张照片
PHOTO = "../../data/detect_photos/canal_street.jpg"   # ← 改这里：换一张照片
r = model.predict(PHOTO, conf=0.4, verbose=False)[0]  # conf：把握低于 40% 的不要   ← 改这里
for box, cls, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
    x1, y1, x2, y2 = [int(v) for v in box]      # 框的左上角和右下角
    print(f"{r.names[int(cls)]:12s} 把握 {float(score):.0%}  框：({x1},{y1})-({x2},{y2})")

# %% 第 3 步：把框画在照片上，存成新图片
r.save("result.jpg")                            # 画好框的照片
print("一共找到", len(r.boxes), "个东西，结果存在 result.jpg")
