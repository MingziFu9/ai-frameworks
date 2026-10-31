# 实验 5-3（通用版）改学习率和轮次，看训练结果怎么变
# %% 第 1 步：只改这两行
LR = 0.001                                      # ← 改这里：学习率，试试 0.1、0.001、0.00001
EPOCHS = 10                                     # ← 改这里：轮次，试试 3、10、20

# %% 第 2 步：读数据、搭网络、训练（和实验 5-1 一样）
import keras
from keras import layers
keras.utils.set_random_seed(42)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "../../data/flowers_small", validation_split=0.2, subset="both", seed=42,
    image_size=(128, 128), batch_size=32)
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)), layers.Rescaling(1 / 255),
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(5, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=LR), metrics=["accuracy"])
h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0).history

# %% 第 3 步：打印最后一轮的结果
print(f"学习率 {LR}，{EPOCHS} 轮：训练损失 {h['loss'][-1]:.4f}，训练准确率 {h['accuracy'][-1]:.4f}，验证准确率 {h['val_accuracy'][-1]:.4f}")
print("每轮训练损失：", [round(v, 3) for v in h["loss"]])
