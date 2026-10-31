# 实验 5-2（国产版）打开盒子：用飞桨动态图写出 fit 背后的训练循环（读懂即可）
# %% 第 1 步：准备数据、网络、损失函数和优化器
import paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)                                        # 固定随机种子
tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])  # 统一大小、缩到 0～1
full = DatasetFolder("../../data/flowers_small", transform=tf)   # 一类一个文件夹
train_ds, val_ds = paddle.io.random_split(full, [600, 150])
loader = paddle.io.DataLoader(train_ds, batch_size=32, shuffle=True)
net = nn.Sequential(                                   # 和第 4 章的花卉网络一样
    nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(32, 64, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Flatten(), nn.Linear(64 * 14 * 14, 64), nn.ReLU(),
    nn.Linear(64, 5), nn.Softmax())                    # 输出 5 种花的概率
loss_fn = nn.CrossEntropyLoss(use_softmax=False)       # 损失函数（输入已经是概率）
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())  # 优化器和学习率

# %% 第 2 步：训练循环——fit 在背后做的就是这些
for epoch in range(10):                                # 一共学 10 轮       ← 改这里
    for images, labels in loader:                      # 每轮把训练集一批一批拿出来
        pred = net(images)                             # ① 前向计算：网络做题
        loss = loss_fn(pred, labels)                   # ② 计算损失：对答案，看错了多少
        opt.clear_grad()                               # ③ 清空上一批的“修改意见”
        loss.backward()                                # ④ 反向传播：算出每个参数该往哪边改
        opt.step()                                     # ⑤ 更新参数：每个参数改一小步
    with paddle.no_grad():                             # 每轮结束考一次验证集（不改参数）
        right = sum(int((net(x).argmax(1) == y.flatten()).sum()) for x, y in paddle.io.DataLoader(val_ds, batch_size=50))
    print(f"第 {epoch + 1} 轮  最后一批损失 {float(loss):.4f}  验证准确率 {right / len(val_ds):.4f}")
