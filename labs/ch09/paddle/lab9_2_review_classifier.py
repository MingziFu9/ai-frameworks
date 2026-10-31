# 实验 9-2（国产版）训练一个外卖评论分类器
# %% 第 1 步：读入评论数据
import pandas as pd
import paddle
from paddle import nn
paddle.seed(42)                                      # 固定随机数，结果可以复现
df = pd.read_csv("../../data/reviews/reviews.csv")   # 两列：text 评论，label 1 好评 / 0 差评
print(df.head(3))
train, val = df[:1200], df[1200:]                    # 前 1200 条训练，后 400 条验证
print("训练", len(train), "条，验证", len(val), "条")

# %% 第 2 步：文字变数字（逐字切分 → 编号 → 填充到 40 个）
from collections import Counter
count = Counter(ch for text in train["text"] for ch in text)    # 数一数每个字出现几次
vocab = ["<pad>", "<unk>"] + [ch for ch, n in count.most_common()]   # 词表：0 填充，1 不认识的字
word2id = {ch: i for i, ch in enumerate(vocab)}      # 字 → 编号 的字典
print("一共", len(vocab), "个字，前 12 个：", vocab[:12])

def encode(text, length=40):
    ids = [word2id.get(ch, 1) for ch in text][:length]   # 查编号，查不到用 1；多了截掉
    return ids + [0] * (length - len(ids))               # 不够 40 个补 0

print(encode("味道很好，下次还点"))                    # 看看一句话变成了什么
def to_dataset(part):
    x = paddle.to_tensor([encode(t) for t in part["text"]])
    y = paddle.to_tensor(part["label"].values.reshape(-1, 1))
    return paddle.io.TensorDataset([x, y])
train_ds, val_ds = to_dataset(train), to_dataset(val)

# %% 第 3 步：搭网络：嵌入层 + 展平 + 全连接
net = nn.Sequential(
    nn.Embedding(len(vocab), 16),                    # 每个字的编号 → 16 个数（字向量）
    nn.Flatten(),                                    # 40×16 展平成 640 个数
    nn.Linear(640, 16), nn.ReLU(),
    nn.Linear(16, 2), nn.Softmax(),                  # 两类：差评、好评
)
model = paddle.Model(net)
model.summary((1, 40), dtype="int64")

# %% 第 4 步：训练（和第 5 章完全一样）
model.prepare(paddle.optimizer.Adam(parameters=net.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, val_ds, epochs=10, batch_size=32, shuffle=True, verbose=1, log_freq=100)

# %% 第 5 步：试试新评论
tests = ["味道很好，送得也快", "等了两个小时，菜都凉了", "虽然有点贵，但是真的好吃",
         "一点都不好吃", "包装很好，可惜味道太咸了"]   # ← 改这里：写几句自己的评论
probs = net(paddle.to_tensor([encode(t) for t in tests])).numpy()
for t, p in zip(tests, probs):
    print(f"{t:16s} → {'好评' if p[1] > 0.5 else '差评'}（好评的把握 {p[1]:.0%}）")

# %% 第 6 步：测一测速度，和实验 9-3 的大模型比一比
import time
start = time.time()
model.predict(val_ds, batch_size=400, verbose=0)
print(f"预测 400 条验证评论，用时 {time.time() - start:.2f} 秒")
