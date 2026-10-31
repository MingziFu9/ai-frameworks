# 实验 7-3（通用版）微调：解冻后面几层，看能不能再提高
# %% 第 1 步：只改这一行
UNFREEZE = 30                                   # ← 改这里：解冻最后几层，试试 0、30、60、154（全部）

# %% 第 2 步：和实验 7-1 一样，先只训练新输出层 10 轮
import keras
from keras import layers
keras.utils.set_random_seed(42)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "../../data/flowers_small", validation_split=0.2, subset="both", seed=42,
    image_size=(128, 128), batch_size=32)
base = keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base.trainable = False
model = keras.Sequential([keras.Input(shape=(128, 128, 3)), layers.Rescaling(1 / 127.5, offset=-1),
                          base, layers.GlobalAveragePooling2D(), layers.Dropout(0.2),
                          layers.Dense(5, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
print("第一阶段（只训练输出层）验证准确率：", round(model.evaluate(val_ds, verbose=0)[1], 4))

# %% 第 3 步：解冻最后 UNFREEZE 层，用很小的学习率再训练 10 轮
base.trainable = UNFREEZE > 0
for layer in base.layers[:len(base.layers) - UNFREEZE]:
    layer.trainable = False                     # 前面的层继续冻结
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.00001), metrics=["accuracy"])   # 学习率小 100 倍
h = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0).history
print(f"解冻最后 {UNFREEZE} 层：可训练参数 {sum(int(w.numpy().size) for w in model.trainable_weights):,}，"
      f"第二阶段后验证准确率 {h['val_accuracy'][-1]:.4f}（最高 {max(h['val_accuracy']):.4f}）")
