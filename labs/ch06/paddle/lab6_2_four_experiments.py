# 实验 6-2（国产版）四组对比实验：数据增强和 Dropout 管不管用
# %% 第 1 步：只改这两行，跑四次（都关、只开增强、只开 Dropout、都开）
AUG = False                                     # ← 改这里：要不要数据增强（True 要 / False 不要）
DROP = False                                    # ← 改这里：要不要 Dropout

# %% 第 2 步：读数据——训练集按开关加增强，验证集不加
import numpy as np, paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)
DATA = "../../data/flowers_small"
plain = [T.Resize((128, 128)), T.ToTensor()]
extra = [T.RandomHorizontalFlip(), T.RandomRotation(36), T.RandomResizedCrop(128, scale=(0.6, 1.0))] if AUG else []
idx = np.random.RandomState(42).permutation(750).tolist()          # 打乱 750 个编号，前 600 训练、后 150 验证
train_ds = paddle.io.Subset(DatasetFolder(DATA, transform=T.Compose(plain[:1] + extra + plain[1:])), idx[:600])
val_ds = paddle.io.Subset(DatasetFolder(DATA, transform=T.Compose(plain)), idx[600:])

# %% 第 3 步：按开关搭网络，训练 30 轮
drop = [nn.Dropout(0.5)] if DROP else []        # 训练时每次随机让一半神经元“休息”
net = nn.Sequential(
    nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2), nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(32, 64, 3), nn.ReLU(), nn.MaxPool2D(2), nn.Flatten(), *drop,
    nn.Linear(64 * 14 * 14, 64), nn.ReLU(), nn.Linear(64, 5), nn.Softmax())
model = paddle.Model(net)
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
from history import History
rec = History()
model.fit(train_ds, val_ds, epochs=30, batch_size=32, shuffle=True, callbacks=[rec], verbose=0)

# %% 第 4 步：看最好的一轮和最后一轮
h = rec.h
best = max(h["val_acc"])
print(f"增强={AUG} Dropout={DROP}：最高验证准确率 {best:.4f}（第 {h['val_acc'].index(best) + 1} 轮），"
      f"最后一轮 训练 {h['acc'][-1]:.4f} / 验证 {h['val_acc'][-1]:.4f}")
print("每轮验证准确率：", [round(v, 3) for v in h["val_acc"]])
