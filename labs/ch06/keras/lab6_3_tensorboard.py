# 实验 6-3（通用版）用 TensorBoard 看训练曲线、比较两次训练
# %% 第 1 步：训练并写日志（改 LR 后再运行一次，得到两份日志）
LR = 0.001                                      # ← 改这里：第一次 0.001，第二次 0.0003
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
              optimizer=keras.optimizers.Adam(LR), metrics=["accuracy"])
board = keras.callbacks.TensorBoard(log_dir=f"logs/lr{LR}")   # 每次训练写进不同的文件夹
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[board], verbose=0)
print("日志已写入", f"logs/lr{LR}")

# %% 第 2 步：在 Notebook 里打开 TensorBoard（去掉下面两行开头的 # 再运行）
# %load_ext tensorboard
# %tensorboard --logdir logs
