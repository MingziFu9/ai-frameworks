# 实验 6-1（通用版）画训练曲线，用早停自动保存最好的模型
# %% 第 1 步：读数据、搭网络（和第 5 章一样）
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
              optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])

# %% 第 2 步：请两个“助手”：早停 + 自动保存最好的模型
helpers = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,      # 验证损失连续 5 轮没变好就停  ← 改这里
                                  restore_best_weights=True),          # 停下后退回到最好的那一轮
    keras.callbacks.ModelCheckpoint("best.keras", monitor="val_loss",
                                    save_best_only=True),              # 只在变好时保存
]
history = model.fit(train_ds, validation_data=val_ds, epochs=30,       # 最多 30 轮
                    callbacks=helpers, verbose=2)

# %% 第 3 步：画训练曲线
import matplotlib.pyplot as plt
h = history.history
plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1); plt.plot(h["loss"], "o-", label="train"); plt.plot(h["val_loss"], "s-", label="val")
plt.title("loss"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(h["accuracy"], "o-", label="train"); plt.plot(h["val_accuracy"], "s-", label="val")
plt.title("accuracy"); plt.xlabel("epoch"); plt.legend()
plt.savefig("curves.png", bbox_inches="tight")
print("一共训练了", len(h["loss"]), "轮；验证损失最低的是第", h["val_loss"].index(min(h["val_loss"])) + 1, "轮")
