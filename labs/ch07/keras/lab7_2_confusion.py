# 实验 7-2（通用版）错误分析：混淆矩阵和认错的照片
# %% 第 1 步：加载实验 7-1 训练好的模型，读入验证集
import numpy as np, keras
model = keras.models.load_model("flowers_transfer.keras")       # 先完成实验 7-1
val_ds = keras.utils.image_dataset_from_directory(
    "../../data/flowers_small", validation_split=0.2, subset="validation", seed=42,
    image_size=(128, 128), batch_size=32)                          # 和训练时同样的分法
names = val_ds.class_names

# %% 第 2 步：对验证集逐张预测，算混淆矩阵
y_true, y_pred, images = [], [], []
for x, y in val_ds:                                              # 一批一批地预测
    y_true += list(y.numpy())                                    # 标准答案
    y_pred += list(model.predict(x, verbose=0).argmax(1))        # 模型的预测
    images += list(x.numpy())                                    # 照片留着后面看
y_true, y_pred = np.array(y_true), np.array(y_pred)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
print("验证准确率：", round((y_true == y_pred).mean(), 4))
print(cm)                                                        # 行是真实类别，列是预测类别

# %% 第 3 步：画成图，再把认错的照片挑出来看
import matplotlib.pyplot as plt
ConfusionMatrixDisplay(cm, display_labels=names).plot(cmap="Blues")
plt.savefig("confusion.png", bbox_inches="tight")
wrong = np.where(y_true != y_pred)[0]                             # 认错的照片编号
plt.figure(figsize=(10, 6))
for k, i in enumerate(wrong[:12]):                               # 最多看 12 张
    plt.subplot(3, 4, k + 1); plt.imshow(images[i].astype("uint8")); plt.axis("off")
    plt.title(f"{names[y_true[i]]} → {names[y_pred[i]]}", fontsize=9)   # 真实 → 预测
plt.savefig("wrong.png", bbox_inches="tight")
print("认错了", len(wrong), "张")
