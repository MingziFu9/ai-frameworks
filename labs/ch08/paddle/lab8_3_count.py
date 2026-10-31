# 实验 8-3（国产版）“数一数”小应用：照片里各有几个
# %% 第 1 步：写一个“数一数”函数
from collections import Counter                 # Counter：专门用来数数的工具
from paddlex import create_model
model = create_model("PP-YOLOE_plus-S")

def count_objects(photo, conf=0.4):             # def 定义一个自己的函数，conf 默认 0.4
    r = next(model.predict(photo, threshold=conf))
    names = [b["label"] for b in r["boxes"]]    # 每个框的类别名
    return Counter(names)                       # 数一数每种各有几个

# %% 第 2 步：数一数几张照片
for name in ["tulip_field", "canal_street", "glass_domes", "grey_roses"]:
    print(name, dict(count_objects(f"../../data/detect_photos/{name}.jpg")))

# %% 第 3 步：把门槛调高调低，看数出来的有什么不同
for conf in [0.2, 0.4, 0.6]:                    # ← 改这里：试试不同的门槛
    print("门槛", conf, dict(count_objects("../../data/detect_photos/tulip_field.jpg", conf)))
