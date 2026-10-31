# 实验 4-2（通用版）搭一个识别花卉的卷积网络
# %% 第 1 步：搭网络
import keras
from keras import layers
keras.utils.set_random_seed(42)                # 固定随机种子，让结果和书上一样
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),           # 输入：128×128 的彩色照片
    layers.Rescaling(1 / 255),                  # 把 0～255 缩到 0～1
    layers.Conv2D(16, 3, activation="relu"),    # 卷积层：16 个 3×3 的“放大镜”   ← 改这里
    layers.MaxPooling2D(),                      # 池化层：长宽各缩小一半
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),                           # 展平成一长串数字
    layers.Dense(64, activation="relu"),        # 全连接层
    layers.Dense(5, activation="softmax"),      # 输出层：5 种花各一个概率
])

# %% 第 2 步：打印结构表
model.summary()

# %% 第 3 步：喂一批照片进去，看输出的形状
DATA = "../../data/flowers_small"
ds = keras.utils.image_dataset_from_directory(DATA, image_size=(128, 128), batch_size=8, seed=42)
images, labels = next(iter(ds))                 # 取一批 8 张
out = model.predict(images, verbose=0)
print("输入形状：", images.shape, " 输出形状：", out.shape)
print("第 1 张照片的 5 个概率：", out[0].round(3))
