# 实验 7-2（国产版）错误分析：混淆矩阵和认错的照片
# %% 第 1 步：搭好同样的网络，加载实验 7-1 训练好的参数，读入验证集
import numpy as np, paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)
tf = T.Compose([T.Resize((128, 128)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
full = DatasetFolder("../../data/flowers_small", transform=tf)
train_ds, val_ds = paddle.io.random_split(full, [600, 150])     # 和训练时同样的分法
names = full.classes
net = paddle.vision.models.mobilenet_v2(pretrained=False)
net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())
model = paddle.Model(net)
model.load("flowers_transfer")                                  # 先完成实验 7-1
net.eval()

# %% 第 2 步：对验证集逐张预测，算混淆矩阵
y_true, y_pred, images = [], [], []
for x, y in paddle.io.DataLoader(val_ds, batch_size=50):       # 一批一批地预测
    y_true += y.numpy().ravel().tolist()                        # 标准答案
    y_pred += net(x).argmax(1).numpy().tolist()                 # 模型的预测
    images += list(x.numpy())                                   # 照片留着后面看
y_true, y_pred = np.array(y_true), np.array(y_pred)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
print("验证准确率：", round((y_true == y_pred).mean(), 4))
print(cm)                                                       # 行是真实类别，列是预测类别

# %% 第 3 步：画成图，再把认错的照片挑出来看
import matplotlib.pyplot as plt
ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues")
plt.savefig("confusion.png", bbox_inches="tight")
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
wrong = np.where(y_true != y_pred)[0]                           # 认错的照片编号
plt.figure(figsize=(10, 6))
for k, i in enumerate(wrong[:12]):                              # 最多看 12 张
    img = (images[i].transpose(1, 2, 0) * std + mean).clip(0, 1)   # 还原成能看的照片
    plt.subplot(3, 4, k + 1); plt.imshow(img); plt.axis("off")
    plt.title(f"{names[y_true[i]]} → {names[y_pred[i]]}", fontsize=9)   # 真实 → 预测
plt.savefig("wrong.png", bbox_inches="tight")
print("认错了", len(wrong), "张")
