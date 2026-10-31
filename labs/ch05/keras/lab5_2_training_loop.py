# 实验 5-2（通用版）打开盒子：用 PyTorch 写出 fit 背后的训练循环（读懂即可）
# %% 第 1 步：准备数据、网络、损失函数和优化器
import torch
from torch import nn
from torchvision import datasets, transforms as T
torch.manual_seed(42)                                  # 固定随机种子
tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])  # 统一大小、缩到 0～1
full = datasets.ImageFolder("../../data/flowers_small", transform=tf)   # 一类一个文件夹
train_ds, val_ds = torch.utils.data.random_split(full, [600, 150])
loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
net = nn.Sequential(                                   # 和第 4 章的花卉网络一样
    nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 14 * 14, 64), nn.ReLU(),
    nn.Linear(64, 5))                                  # PyTorch 习惯不写 Softmax，损失函数里自带
loss_fn = nn.CrossEntropyLoss()                        # 损失函数
opt = torch.optim.Adam(net.parameters(), lr=0.001)     # 优化器和学习率

# %% 第 2 步：训练循环——fit 在背后做的就是这些
for epoch in range(10):                                # 一共学 10 轮       ← 改这里
    for images, labels in loader:                      # 每轮把训练集一批一批拿出来
        pred = net(images)                             # ① 前向计算：网络做题
        loss = loss_fn(pred, labels)                   # ② 计算损失：对答案，看错了多少
        opt.zero_grad()                                # ③ 清空上一批的“修改意见”
        loss.backward()                                # ④ 反向传播：算出每个参数该往哪边改
        opt.step()                                     # ⑤ 更新参数：每个参数改一小步
    with torch.no_grad():                              # 每轮结束考一次验证集（不改参数）
        right = sum((net(x).argmax(1) == y).sum().item() for x, y in torch.utils.data.DataLoader(val_ds, batch_size=50))
    print(f"第 {epoch + 1} 轮  最后一批损失 {loss.item():.4f}  验证准确率 {right / len(val_ds):.4f}")
