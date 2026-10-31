# 实验 5-3（国产版）改学习率和轮次，看训练结果怎么变
# %% 第 1 步：只改这两行
LR = 0.001                                      # ← 改这里：学习率，试试 0.1、0.001、0.00001
EPOCHS = 10                                     # ← 改这里：轮次，试试 3、10、20

# %% 第 2 步：读数据、搭网络、训练（和实验 5-1 一样）
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
model.prepare(paddle.optimizer.Adam(learning_rate=LR, parameters=model.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, epochs=EPOCHS, batch_size=32, shuffle=True, verbose=0)

# %% 第 3 步：分别在训练集和验证集上考一次
tr = model.evaluate(train_ds, batch_size=50, verbose=0)
va = model.evaluate(val_ds, batch_size=50, verbose=0)
print(f"学习率 {LR}，{EPOCHS} 轮：训练损失 {tr['loss'][0]:.4f}，训练准确率 {tr['acc']:.4f}，验证准确率 {va['acc']:.4f}")
