# 实验 6-1（国产版）画训练曲线，用早停自动保存最好的模型
# %% 第 1 步：读数据、搭网络（和第 5 章一样）
import paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)
full = DatasetFolder("../../data/flowers_small", transform=T.Compose([T.Resize((128, 128)), T.ToTensor()]))
train_ds, val_ds = paddle.io.random_split(full, [600, 150])
net = nn.Sequential(
    nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2), nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(32, 64, 3), nn.ReLU(), nn.MaxPool2D(2), nn.Flatten(),
    nn.Linear(64 * 14 * 14, 64), nn.ReLU(), nn.Linear(64, 5), nn.Softmax())
model = paddle.Model(net)
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())

# %% 第 2 步：请两个“助手”：早停（顺便保存最好的模型）+ 记录成绩
from history import History                     # 同一文件夹里的 history.py
rec = History()
stop = paddle.callbacks.EarlyStopping(monitor="loss", mode="min", patience=5,   # 验证损失连续 5 轮没变好就停  ← 改这里
                                      save_best_model=True)                     # 把最好的一轮存到 save_dir/best_model
model.fit(train_ds, val_ds, epochs=30, batch_size=32, shuffle=True,             # 最多 30 轮
          save_dir="checkpoint", save_freq=100,                         # 存到 checkpoint 文件夹
          callbacks=[stop, rec], verbose=0)

# %% 第 3 步：画训练曲线
import matplotlib.pyplot as plt
h = rec.h
plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1); plt.plot(h["loss"], "o-", label="train"); plt.plot(h["val_loss"], "s-", label="val")
plt.title("loss"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(h["acc"], "o-", label="train"); plt.plot(h["val_acc"], "s-", label="val")
plt.title("accuracy"); plt.xlabel("epoch"); plt.legend()
plt.savefig("curves.png", bbox_inches="tight")
print("一共训练了", len(h["loss"]), "轮；验证损失最低的是第", h["val_loss"].index(min(h["val_loss"])) + 1, "轮")
print("每轮验证准确率：", [round(v, 3) for v in h["val_acc"]])
