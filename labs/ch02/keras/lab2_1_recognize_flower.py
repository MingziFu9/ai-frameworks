# 实验 2-1（通用版）让模型认出一张花的照片
# %% 第 1 步：请出一个别人训练好的模型
import numpy as np
from keras.applications import mobilenet_v2   # Keras 自带的现成模型之一

model = mobilenet_v2.MobileNetV2(weights="imagenet")  # 第一次运行会自动下载约 14 MB 的“模型参数”

# %% 第 2 步：读入照片，整理成模型要的样子
import keras

PHOTO = "../../data/test_photos/daisy_1.jpg"           # ← 改这里：换成你自己的照片
img = keras.utils.load_img(PHOTO, target_size=(224, 224))  # 读图，同时缩放到 224×224
x = keras.utils.img_to_array(img)                      # 照片变成数字
x = mobilenet_v2.preprocess_input(np.expand_dims(x, 0))  # 变成“一批一张”，再按模型要求换算

# %% 第 3 步：交给模型，打印最有可能的 5 个答案
pred = model.predict(x, verbose=0)
for _, name, prob in mobilenet_v2.decode_predictions(pred, top=5)[0]:
    print(f"{name:20s} {prob:.1%}")
