# 实验 7-1（通用版）迁移学习：借 MobileNetV2 的“眼睛”认花
# %% 第 1 步：读数据（和第 5 章一样）
import keras
from keras import layers
keras.utils.set_random_seed(42)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "../../data/flowers_small", validation_split=0.2, subset="both", seed=42,
    image_size=(128, 128), batch_size=32)
class_names = train_ds.class_names

# %% 第 2 步：请出预训练模型，去掉它原来的输出层，冻结
base = keras.applications.MobileNetV2(input_shape=(128, 128, 3),
                                      include_top=False,       # 不要原来 1000 类的输出层
                                      weights="imagenet")      # 要 ImageNet 上训练好的参数
base.trainable = False                                         # 冻结：训练时不改它的参数

# %% 第 3 步：接上自己的输出层
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),
    layers.Rescaling(1 / 127.5, offset=-1),     # 0～255 换算成 -1～1（MobileNetV2 的规矩）
    base,                                       # 借来的“眼睛”
    layers.GlobalAveragePooling2D(),            # 把图案地图压成一串数
    layers.Dropout(0.2),
    layers.Dense(5, activation="softmax"),      # 新的输出层：5 种花
])
model.summary()

# %% 第 4 步：训练（只训练新输出层）
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)   # ← 改这里：轮次
model.save("flowers_transfer.keras")

# %% 第 5 步：认一认第 2 章那 10 张照片
import numpy as np
from pathlib import Path
for photo in sorted(Path("../../data/test_photos").glob("*.jpg")):
    x = np.expand_dims(keras.utils.img_to_array(keras.utils.load_img(photo, target_size=(128, 128))), 0)
    prob = model.predict(x, verbose=0)[0]
    print(f"{photo.name:16s} → {class_names[prob.argmax()]:10s} {prob.max():.1%}")
