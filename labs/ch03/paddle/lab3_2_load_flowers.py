# 实验 3-2（国产版）读入花卉数据集
# %% 第 1 步：读入整个文件夹，再分出训练集和验证集
import paddle
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
DATA = "../../data/flowers_small"               # ← 改这里：数据文件夹的位置
paddle.seed(42)                                 # 打乱顺序的“种子”，让每次分法一样
tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])  # 统一成 128×128 的张量
full = DatasetFolder(DATA, transform=tf)        # 每个子文件夹是一类
train_ds, val_ds = paddle.io.random_split(full, [600, 150])  # 600 张训练，150 张验证
print("类别：", full.classes, " 一共", len(full), "张")

# %% 第 2 步：用 DataLoader 一批一批地取出来
loader = paddle.io.DataLoader(train_ds, batch_size=32, shuffle=True)  # ← 改这里：每批几张
images, labels = next(iter(loader))             # 只取第 1 批
print("一批图片的形状：", images.shape)          # [张数, 3, 高, 宽]
print("这一批的标签：", labels.numpy().ravel())   # 每张图对应的类别编号
print("训练集一共", len(loader), "批")

# %% 第 3 步：把这一批的前 9 张画出来，标上类别
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
for i in range(9):                              # 依次画第 0～8 张
    plt.subplot(3, 3, i + 1)                    # 3 行 3 列中的第 i+1 格
    plt.imshow(images[i].transpose([1, 2, 0]).numpy())   # 通道挪到最后才能画
    plt.title(full.classes[int(labels[i])]); plt.axis("off")
plt.savefig("one_batch.png")                    # 存成图片（Notebook 里会直接显示）
