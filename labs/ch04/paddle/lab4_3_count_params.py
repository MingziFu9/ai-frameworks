# 实验 4-3（国产版）加减层，看参数量怎么变
# %% 第 1 步：把网络写成“可调”的样子
import paddle
from paddle import nn
FILTERS = [16, 32, 64]                          # ← 改这里：每个卷积层几个“放大镜”，也可以加减个数
DENSE = 64                                      # ← 改这里：全连接层几个神经元
blocks, c, size = [], 3, 128                    # c：当前通道数；size：当前边长
for n in FILTERS:                               # 每个数搭一组“卷积 + 池化”
    blocks += [nn.Conv2D(c, n, 3), nn.ReLU(), nn.MaxPool2D(2)]
    c, size = n, (size - 2) // 2                # 卷积边长减 2，池化再减半
net = nn.Sequential(*blocks, nn.Flatten(), nn.Linear(c * size * size, DENSE), nn.ReLU(),
                    nn.Linear(DENSE, 5), nn.Softmax())

# %% 第 2 步：看总参数量和展平后的长度
print("卷积层设置：", FILTERS, " 全连接：", DENSE)
print("展平后有", c * size * size, "个数")
print("总参数量：", f"{sum(int(p.numel()) for p in net.parameters()):,}")
