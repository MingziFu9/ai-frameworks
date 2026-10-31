# 实验 5-1（国产版）训练第一个花卉识别模型
# %% 第 1 步：导入工具，读入数据
import paddle                                   # 导入飞桨框架
from paddle import nn                           # nn 里放着各种“积木”（层）
from paddle.vision import transforms as T       # 图片预处理工具
from paddle.vision.datasets import DatasetFolder

DATA = "../../data/flowers_small"               # ← 改这里：数据文件夹的位置
paddle.seed(42)                                 # 固定随机种子，让每次结果一样
tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])  # 统一成 128×128，再变成 0～1 的张量
full = DatasetFolder(DATA, transform=tf)
class_names = full.classes
train_ds, val_ds = paddle.io.random_split(full, [600, 150])  # 600 张训练，150 张验证
print("类别：", class_names)

# %% 第 2 步：搭网络（和第 4 章一样）
net = nn.Sequential(
    nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2),   # 卷积层 + 激活 + 池化
    nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(32, 64, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Flatten(),                                      # 展平成一长串数字
    nn.Linear(64 * 14 * 14, 64), nn.ReLU(),            # 全连接层
    nn.Linear(64, 5))                                  # 输出层：5 种花各一个分数
model = paddle.Model(net)                              # 套上高层 API 的“外壳”

# %% 第 3 步：训练前的三项设置
model.prepare(
    optimizer=paddle.optimizer.Adam(learning_rate=0.001,       # ← 改这里：学习率
                                    parameters=model.parameters()),
    loss=nn.CrossEntropyLoss(),                 # 损失函数：衡量答案和标准答案差多少
    metrics=paddle.metric.Accuracy())           # 评价指标：准确率

# %% 第 4 步：开始训练
model.fit(train_ds, val_ds, epochs=10, batch_size=32, shuffle=True, verbose=2)  # ← 改这里：轮次

# %% 第 5 步：保存模型，用它预测一张新照片
model.save("flowers_cnn")                       # 保存为 flowers_cnn.pdparams 等文件
from PIL import Image
x = tf(Image.open("../../data/test_photos/roses_1.jpg").convert("RGB")).unsqueeze(0)  # 一批只有一张
prob = nn.functional.softmax(net(x))[0].numpy() # 分数变成概率
print("预测：", class_names[prob.argmax()], f"（把握 {prob.max():.1%}）")
