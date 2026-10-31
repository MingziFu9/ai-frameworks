# 实验 4-3（通用版）加减层，看参数量怎么变
# %% 第 1 步：把网络写成“可调”的样子
import keras
from keras import layers
FILTERS = [16, 32, 64]                          # ← 改这里：每个卷积层几个“放大镜”，也可以加减个数
DENSE = 64                                      # ← 改这里：全连接层几个神经元
blocks = []
for n in FILTERS:                               # 每个数搭一组“卷积 + 池化”
    blocks += [layers.Conv2D(n, 3, activation="relu"), layers.MaxPooling2D()]
model = keras.Sequential([keras.Input(shape=(128, 128, 3)), *blocks,
                          layers.Flatten(), layers.Dense(DENSE, activation="relu"),
                          layers.Dense(5, activation="softmax")])

# %% 第 2 步：看总参数量和展平后的长度
print("卷积层设置：", FILTERS, " 全连接：", DENSE)
print("展平后有", model.layers[-3].output.shape[1], "个数")
print("总参数量：", f"{model.count_params():,}")
