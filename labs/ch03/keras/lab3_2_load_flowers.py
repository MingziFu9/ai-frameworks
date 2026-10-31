# 实验 3-2（通用版）读入花卉数据集
# %% 第 1 步：一行读入整个文件夹，同时分出训练集和验证集
import keras
DATA = "../../data/flowers_small"               # ← 改这里：数据文件夹的位置
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATA, validation_split=0.2, subset="both",  # 20% 留作验证集
    seed=42, image_size=(128, 128),             # 打乱顺序的“种子”；统一成 128×128
    batch_size=32)                              # ← 改这里：每批几张
print("类别：", train_ds.class_names)

# %% 第 2 步：取出一批看看
for images, labels in train_ds.take(1):         # 只取第 1 批
    print("一批图片的形状：", images.shape)      # （张数, 高, 宽, 3）
    print("这一批的标签：", labels.numpy())      # 每张图对应的类别编号
print("训练集一共", len(train_ds), "批，验证集", len(val_ds), "批")

# %% 第 3 步：把这一批的前 9 张画出来，标上类别
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
for i in range(9):                              # 依次画第 0～8 张
    plt.subplot(3, 3, i + 1)                    # 3 行 3 列中的第 i+1 格
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[labels[i]]); plt.axis("off")
plt.savefig("one_batch.png")                    # 存成图片（Notebook 里会直接显示）
