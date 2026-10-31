# 实验 6-2（通用版）四组对比实验：数据增强和 Dropout 管不管用
# %% 第 1 步：只改这两行，跑四次（都关、只开增强、只开 Dropout、都开）
AUG = False                                     # ← 改这里：要不要数据增强（True 要 / False 不要）
DROP = False                                    # ← 改这里：要不要 Dropout

# %% 第 2 步：读数据，按开关搭网络
import keras
from keras import layers
keras.utils.set_random_seed(42)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "../../data/flowers_small", validation_split=0.2, subset="both", seed=42,
    image_size=(128, 128), batch_size=32)
aug = [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.2)] if AUG else []
drop = [layers.Dropout(0.5)] if DROP else []    # 训练时每次随机让一半神经元“休息”
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)), *aug, layers.Rescaling(1 / 255),   # 增强积木放在最前面
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), *drop, layers.Dense(64, activation="relu"), layers.Dense(5, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])

# %% 第 3 步：训练 30 轮，看最好的一轮和最后一轮
h = model.fit(train_ds, validation_data=val_ds, epochs=30, verbose=0).history
best = max(h["val_accuracy"])
print(f"增强={AUG} Dropout={DROP}：最高验证准确率 {best:.4f}（第 {h['val_accuracy'].index(best) + 1} 轮），"
      f"最后一轮 训练 {h['accuracy'][-1]:.4f} / 验证 {h['val_accuracy'][-1]:.4f}")
print("每轮验证准确率：", [round(v, 3) for v in h["val_accuracy"]])
