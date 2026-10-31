# 实验 7-1（国产版）迁移学习：借 MobileNetV2 的“眼睛”认花
# %% 第 1 步：读数据（按预训练模型的规矩做标准化）
import paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)
tf = T.Compose([T.Resize((128, 128)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   # ImageNet 的标准化
full = DatasetFolder("../../data/flowers_small", transform=tf)
train_ds, val_ds = paddle.io.random_split(full, [600, 150])
class_names = full.classes

# %% 第 2 步：请出预训练模型，冻结，换上新的输出层
net = paddle.vision.models.mobilenet_v2(pretrained=True)        # 带 ImageNet 上训练好的参数
for p in net.parameters():
    p.stop_gradient = True                                      # 冻结：训练时不改它的参数
net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())  # 新的输出层：5 种花
paddle.summary(net, (1, 3, 128, 128))

# %% 第 3 步：训练（只训练新输出层）
model = paddle.Model(net)
model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=net.classifier.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, val_ds, epochs=10, batch_size=32, shuffle=True, verbose=1, log_freq=100)  # ← 改这里：轮次
model.save("flowers_transfer")

# %% 第 4 步：认一认第 2 章那 10 张照片
from pathlib import Path
from PIL import Image
net.eval()                                                      # 切换到预测模式
for photo in sorted(Path("../../data/test_photos").glob("*.jpg")):
    prob = net(tf(Image.open(photo).convert("RGB")).unsqueeze(0))[0].numpy()
    print(f"{photo.name:16s} → {class_names[prob.argmax()]:10s} {prob.max():.1%}")
