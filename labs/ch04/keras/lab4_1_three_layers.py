# 实验 4-1（通用版）搭一个三层网络，看懂结构表
# %% 第 1 步：用 Sequential 把积木从上到下摞起来
import keras
from keras import layers
keras.utils.set_random_seed(42)                # 固定随机种子，让结果和书上一样
model = keras.Sequential([
    keras.Input(shape=(4,)),                    # 输入层：每朵花 4 个数（花萼长、宽，花瓣长、宽）
    layers.Dense(8, activation="relu"),         # 隐藏层：8 个神经元   ← 改这里
    layers.Dense(3, activation="softmax"),      # 输出层：3 种鸢尾花各一个概率
])

# %% 第 2 步：打印结构表
model.summary()                                 # 每层的输出形状和参数量

# %% 第 3 步：还没训练，先让它猜一猜
import numpy as np
flower = np.array([[5.1, 3.5, 1.4, 0.2]])       # 一朵花的 4 个测量值（单位：厘米）
prob = model.predict(flower, verbose=0)         # 交给网络
print("三种花的概率：", prob.round(3), " 加起来：", prob.sum().round(3))
