# 实验 3-3（国产版）数据增强效果对比
# %% 第 1 步：准备一张照片和一组“变换”
import numpy as np
from PIL import Image
from paddle.vision import transforms as T
img = Image.open("../../data/test_photos/sunflowers_1.jpg").convert("RGB").resize((128, 128))
augment = T.Compose([
    T.RandomHorizontalFlip(),                   # 随机左右翻转
    T.RandomRotation(36),                       # 随机旋转（最多 ±36 度）  ← 改这里
    T.RandomResizedCrop(128, scale=(0.6, 1.0)), # 随机裁一块再放大回 128   ← 改这里
])

# %% 第 2 步：同一张照片变 8 次，和原图放在一起看
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
for i in range(8):
    out = img if i == 0 else augment(img)       # 第 1 格放原图
    plt.subplot(2, 4, i + 1); plt.axis("off")
    plt.imshow(np.array(out)); plt.title("original" if i == 0 else f"aug {i}")
plt.savefig("augment.png")
