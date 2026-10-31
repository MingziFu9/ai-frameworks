# 实验 2-2（通用版）换图试试：用现成模型认一批照片
# %% 第 1 步：准备模型
from pathlib import Path

import keras
import numpy as np
from keras.applications import mobilenet_v2

model = mobilenet_v2.MobileNetV2(weights="imagenet")

# %% 第 2 步：把文件夹里的照片一张张交给模型
FOLDER = Path("../../data/test_photos")                # ← 改这里：换成放你自己照片的文件夹
for photo in sorted(FOLDER.glob("*.jpg")):
    img = keras.utils.load_img(photo, target_size=(224, 224))
    x = mobilenet_v2.preprocess_input(np.expand_dims(keras.utils.img_to_array(img), 0))
    _, name, prob = mobilenet_v2.decode_predictions(model.predict(x, verbose=0), top=1)[0][0]
    print(f"{photo.name:18s} → {name:16s} {prob:.1%}")
