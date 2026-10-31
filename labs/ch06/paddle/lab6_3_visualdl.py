# 实验 6-3（国产版）用 VisualDL 看训练曲线、比较两次训练
# %% 第 1 步：训练并写日志（改 LR 后再运行一次，得到两份日志）
LR = 0.001                                      # ← 改这里：第一次 0.001，第二次 0.0003
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
board = paddle.callbacks.VisualDL(log_dir=f"vdl_log/lr{LR}")  # 每次训练写进不同的文件夹
model.fit(train_ds, val_ds, epochs=15, batch_size=32, shuffle=True, callbacks=[board], verbose=0)
print("日志已写入", f"vdl_log/lr{LR}")

# %% 第 2 步：打开 VisualDL
# AI Studio：左侧“可视化”面板选择 vdl_log 文件夹即可
# 本机：在命令行运行  visualdl --logdir vdl_log --port 8040  ，再用浏览器打开 http://localhost:8040
