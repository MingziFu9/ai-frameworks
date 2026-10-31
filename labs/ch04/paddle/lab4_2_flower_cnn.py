# 实验 4-2（国产版）搭一个识别花卉的卷积网络
# %% 第 1 步：搭网络
import paddle
from paddle import nn
paddle.seed(42)                                 # 固定随机种子，让结果和书上一样
net = nn.Sequential(
    nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2),   # 卷积 + 激活 + 池化   ← 改这里
    nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(32, 64, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Flatten(),                                      # 展平成一长串数字
    nn.Linear(64 * 14 * 14, 64), nn.ReLU(),            # 全连接层
    nn.Linear(64, 5), nn.Softmax(),                    # 输出层：5 种花各一个概率
)

# %% 第 2 步：打印结构表
paddle.summary(net, (1, 3, 128, 128))           # 输入形状：1 张、3 通道、128×128

# %% 第 3 步：喂一批照片进去，看输出的形状
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
ds = DatasetFolder("../../data/flowers_small", transform=T.Compose([T.Resize((128, 128)), T.ToTensor()]))
images, labels = next(iter(paddle.io.DataLoader(ds, batch_size=8, shuffle=True)))  # 取一批 8 张
out = net(images)
print("输入形状：", images.shape, " 输出形状：", out.shape)
print("第 1 张照片的 5 个概率：", out[0].numpy().round(3))
