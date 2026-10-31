# 实验 9-2（通用版）训练一个外卖评论分类器
# %% 第 1 步：读入评论数据
import pandas as pd
import keras
from keras import layers
keras.utils.set_random_seed(42)                      # 固定随机数，结果可以复现
df = pd.read_csv("../../data/reviews/reviews.csv")   # 两列：text 评论，label 1 好评 / 0 差评
print(df.head(3))
train, val = df[:1200], df[1200:]                    # 前 1200 条训练，后 400 条验证
print("训练", len(train), "条，验证", len(val), "条")

# %% 第 2 步：文字变数字（逐字切分 → 编号 → 填充到 40 个）
vec = layers.TextVectorization(split="character",    # 按单个字切分
                               output_sequence_length=40)   # 不够 40 个字补 0，多了截掉
vec.adapt(train["text"])                             # 看一遍训练集，给每个字编号
vocab = [str(w) for w in vec.get_vocabulary()]   # 词表：编号 0 是填充，1 是不认识的字
print("一共", len(vocab), "个字，前 12 个：", vocab[:12])
print(vec(["味道很好，下次还点"]).numpy())           # 看看一句话变成了什么
x_train, x_val = vec(train["text"]), vec(val["text"])
y_train, y_val = train["label"].values, val["label"].values

# %% 第 3 步：搭网络：嵌入层 + 展平 + 全连接
model = keras.Sequential([
    keras.Input(shape=(40,)),
    layers.Embedding(len(vocab), 16),                # 每个字的编号 → 16 个数（字向量）
    layers.Flatten(),                                # 40×16 展平成 640 个数
    layers.Dense(16, activation="relu"),
    layers.Dense(2, activation="softmax"),           # 两类：差评、好评
])
model.summary()

# %% 第 4 步：训练（和第 5 章完全一样）
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32, verbose=2)

# %% 第 5 步：试试新评论
tests = ["味道很好，送得也快", "等了两个小时，菜都凉了", "虽然有点贵，但是真的好吃",
         "一点都不好吃", "包装很好，可惜味道太咸了"]   # ← 改这里：写几句自己的评论
probs = model.predict(vec(tests), verbose=0)
for t, p in zip(tests, probs):
    print(f"{t:16s} → {'好评' if p[1] > 0.5 else '差评'}（好评的把握 {p[1]:.0%}）")

# %% 第 6 步：测一测速度，和实验 9-3 的大模型比一比
import time
start = time.time()
model.predict(x_val, verbose=0)
print(f"预测 400 条验证评论，用时 {time.time() - start:.2f} 秒")
