# 实验 4-1（国产版）搭一个三层网络，看懂结构表
# %% 第 1 步：用 Sequential 把积木从上到下摞起来
import paddle
from paddle import nn
paddle.seed(42)                                 # 固定随机种子，让结果和书上一样
net = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),                 # 隐藏层：4 个输入 → 8 个神经元   ← 改这里
    nn.Linear(8, 3), nn.Softmax(),              # 输出层：3 种鸢尾花各一个概率
)

# %% 第 2 步：打印结构表
paddle.summary(net, (1, 4))                     # 输入形状：1 朵花、4 个数

# %% 第 3 步：还没训练，先让它猜一猜
flower = paddle.to_tensor([[5.1, 3.5, 1.4, 0.2]])   # 一朵花的 4 个测量值（单位：厘米）
prob = net(flower)                                  # 交给网络
print("三种花的概率：", prob.numpy().round(3), " 加起来：", prob.sum().numpy().round(3))
