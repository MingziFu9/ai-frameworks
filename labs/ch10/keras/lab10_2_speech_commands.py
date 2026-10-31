# 实验 10-2（通用版）训练一个语音指令识别器：听懂 8 个英文单词
# %% 第 1 步：读入声音数据（和第 3 章读花卉照片几乎一样）
import keras
from keras import layers
keras.utils.set_random_seed(42)                      # 固定随机数，结果可以复现
train_ds, val_ds = keras.utils.audio_dataset_from_directory(
    "../../data/speech_commands_small",              # 每个子文件夹是一个词，各 100 条录音
    batch_size=32, validation_split=0.2, subset="both", seed=42,
    output_sequence_length=8000)                     # 每条统一成 8000 个数（1 秒，不够的补 0）
names = train_ds.class_names
print("8 个词：", names)

# %% 第 2 步：搭网络：第一块积木把声音变成声谱图，后面就是第 4 章的卷积网络
model = keras.Sequential([
    keras.Input(shape=(8000, 1)),
    layers.Reshape((8000,)),                         # 去掉最后那个 1，变成一串 8000 个数
    layers.MelSpectrogram(sampling_rate=8000, fft_length=256, sequence_stride=128,
                          num_mel_bins=40),          # 声音 → 声谱图（40 个频段 × 63 个时间片）
    layers.Reshape((40, 63, 1)),                     # 加一个“通道”，当成一张单色图片
    layers.BatchNormalization(),                     # 把数值调整到合适的范围，训练更稳定
    layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(8, activation="softmax"),           # 8 个词
])
model.summary()

# %% 第 3 步：训练（和第 5 章完全一样）
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=2)   # ← 改这里：轮次

# %% 第 4 步：听一条验证集的录音，看模型认成什么
import numpy as np
for x, y in val_ds.take(1):                          # 取验证集的一批
    probs = model.predict(x, verbose=0)
    for i in range(8):                               # 看前 8 条
        p = probs[i]
        print(f"真实：{names[y[i]]:6s} 预测：{names[np.argmax(p)]:6s}（把握 {p.max():.0%}）")
