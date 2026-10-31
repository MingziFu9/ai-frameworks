# 实验 8-1（国产版）目标检测：框出照片里的东西
# %% 第 1 步：请出现成的检测模型 PP-YOLOE+
from paddlex import create_model               # 先运行 !pip install paddlex
model = create_model("PP-YOLOE_plus-S")         # S 表示小号版本；第一次运行会自动下载参数

# %% 第 2 步：检测一张照片
PHOTO = "../../data/detect_photos/canal_street.jpg"   # ← 改这里：换一张照片
r = next(model.predict(PHOTO, threshold=0.4))   # threshold：把握低于 40% 的不要   ← 改这里
for b in r["boxes"]:                            # 每个框是一个字典
    x1, y1, x2, y2 = [int(v) for v in b["coordinate"]]   # 框的左上角和右下角
    print(f"{b['label']:12s} 把握 {b['score']:.0%}  框：({x1},{y1})-({x2},{y2})")

# %% 第 3 步：把框画在照片上，存成新图片
r.save_to_img("result.jpg")                     # 画好框的照片
print("一共找到", len(r["boxes"]), "个东西，结果存在 result.jpg")
