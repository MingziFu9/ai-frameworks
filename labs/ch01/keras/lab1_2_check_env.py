# 实验 1-2（通用版）环境自检
# %% 第 1 步：看看都装了什么
import platform
import time

import keras
import numpy as np
import tensorflow as tf

print("Python 版本：", platform.python_version())
print("Keras 版本：", keras.__version__, "| 后端：", keras.config.backend())
print("TensorFlow 版本：", tf.__version__)

# %% 第 2 步：有没有 GPU
gpus = tf.config.list_physical_devices("GPU")
print("GPU 数量：", len(gpus), "（0 表示只能用 CPU，本课的必做实验用 CPU 也能完成）")

# %% 第 3 步：让框架做一道大题，计时
a = np.random.rand(2000, 2000).astype("float32")   # 2000×2000 的随机数表格
start = time.time()
c = keras.ops.matmul(a, a)                          # 矩阵乘法：约 80 亿次乘法和加法
print(f"2000×2000 矩阵乘法用时：{time.time() - start:.3f} 秒")
