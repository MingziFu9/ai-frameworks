# 实验 12-3（两版通用）昇思 MindSpore 体验：用第 4 章的三层网络认鸢尾花
# %% 第 1 步：准备数据（和第 4 章一样的鸢尾花数据）
import numpy as np
import mindspore as ms                               # 先运行 !pip install mindspore "numpy<2"
from mindspore import nn
from mindspore.dataset import NumpySlicesDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
ms.set_seed(42)                                      # 固定随机数，结果可以复现
X, y = load_iris(return_X_y=True)                    # 150 朵花，每朵 4 个测量值
X_train, X_val, y_train, y_val = train_test_split(X.astype("float32"), y.astype("int32"),
                                                  test_size=0.2, random_state=42)
train_ds = NumpySlicesDataset((X_train, y_train), column_names=["x", "y"], shuffle=True).batch(16)
val_ds = NumpySlicesDataset((X_val, y_val), column_names=["x", "y"]).batch(16)

# %% 第 2 步：搭网络（对照第 4 章：Sequential → SequentialCell，Linear/Dense → Dense）
net = nn.SequentialCell(
    nn.Dense(4, 8), nn.ReLU(),                       # 隐藏层：4 个输入 → 8 个神经元
    nn.Dense(8, 3),                                  # 输出层：3 种鸢尾花（昇思的习惯：这里不加 Softmax）
)
print(net)

# %% 第 3 步：训练（对照第 5 章：损失函数、优化器、评价指标，再 train）
from mindspore.train import Model, LossMonitor
loss = nn.CrossEntropyLoss()                         # 交叉熵损失，自带 Softmax
opt = nn.Adam(net.trainable_params(), learning_rate=0.01)
model = Model(net, loss_fn=loss, optimizer=opt, metrics={"accuracy"})
model.train(100, train_ds, callbacks=[LossMonitor(per_print_times=8 * 20)])   # 训练 100 轮，每 20 轮打印一次
print("验证集：", model.eval(val_ds))

# %% 第 4 步：认一朵新花
flower = ms.Tensor([[5.1, 3.5, 1.4, 0.2]], ms.float32)   # 一朵花的 4 个测量值（厘米）
prob = ms.ops.softmax(net(flower), axis=1)           # 自己加上 Softmax，变成概率
names = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
print("预测：", names[int(prob.argmax())], prob.asnumpy().round(3))
