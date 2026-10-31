# 实验 3-3（通用版）数据增强效果对比
# %% 第 1 步：准备一张照片和一组“变换”积木
import keras, numpy as np
from keras import layers
img = keras.utils.load_img("../../data/test_photos/sunflowers_1.jpg", target_size=(128, 128))
x = np.expand_dims(keras.utils.img_to_array(img), 0)   # 一批只有 1 张
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),            # 随机左右翻转
    layers.RandomRotation(0.1),                 # 随机旋转（最多 ±36 度）  ← 改这里
    layers.RandomZoom(0.2),                     # 随机放大缩小 20%         ← 改这里
])

# %% 第 2 步：同一张照片变 8 次，和原图放在一起看
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
for i in range(8):
    out = x[0] if i == 0 else augment(x, training=True)[0]   # 第 1 格放原图
    plt.subplot(2, 4, i + 1); plt.axis("off")
    plt.imshow(np.array(out).astype("uint8")); plt.title("original" if i == 0 else f"aug {i}")
plt.savefig("augment.png")
