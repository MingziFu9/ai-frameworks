# 实验 2-1（国产版）让模型认出一张花的照片
# %% 第 1 步：请出一个别人训练好的模型
from paddlex import create_model             # PaddleX：飞桨的“模型超市”，按名字取模型

model = create_model(model_name="PP-LCNet_x1_0")   # 飞桨自研的轻量图像分类模型，第一次运行会自动下载

# %% 第 2 步：交给模型，打印最有可能的 5 个答案
PHOTO = "../../data/test_photos/daisy_1.jpg"   # ← 改这里：换成你自己的照片
for res in model.predict(PHOTO, batch_size=1, topk=5):   # 读图、缩放、换算都由 PaddleX 自动完成
    for name, score in zip(res["label_names"], res["scores"]):
        print(f"{name:20s} {score:.1%}")
