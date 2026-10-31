# 实验 7-3（国产版）微调：解冻后面几块，看能不能再提高
# %% 第 1 步：只改这一行
UNFREEZE = 3                                    # ← 改这里：解冻最后几块，试试 0、3、6、19（全部）

# %% 第 2 步：和实验 7-1 一样，先只训练新输出层 10 轮
import paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)
tf = T.Compose([T.Resize((128, 128)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_ds, val_ds = paddle.io.random_split(DatasetFolder("../../data/flowers_small", transform=tf), [600, 150])
net = paddle.vision.models.mobilenet_v2(pretrained=True)
for p in net.parameters():
    p.stop_gradient = True
net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())
model = paddle.Model(net)
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=net.classifier.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, epochs=10, batch_size=32, shuffle=True, verbose=0)
print("第一阶段（只训练输出层）验证准确率：", round(model.evaluate(val_ds, batch_size=50, verbose=0)["acc"], 4))

# %% 第 3 步：解冻最后 UNFREEZE 块，用很小的学习率再训练 10 轮
blocks = list(net.features.children())          # MobileNetV2 的特征部分一共 19 块
for b in blocks[len(blocks) - UNFREEZE:]:
    for p in b.parameters():
        p.stop_gradient = False                 # 解冻
params = [p for p in net.parameters() if not p.stop_gradient]
model.prepare(paddle.optimizer.Adam(learning_rate=0.00001, parameters=params),   # 学习率小 100 倍
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, epochs=10, batch_size=32, shuffle=True, verbose=0)
print(f"解冻最后 {UNFREEZE} 块：可训练参数 {sum(int(p.numel()) for p in params):,}，"
      f"第二阶段后验证准确率 {model.evaluate(val_ds, batch_size=50, verbose=0)['acc']:.4f}")
