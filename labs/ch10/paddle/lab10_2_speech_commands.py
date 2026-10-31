# 实验 10-2（国产版）训练一个语音指令识别器：听懂 8 个英文单词
# %% 第 1 步：读入声音数据（和第 3 章读花卉照片几乎一样）
import numpy as np
import paddle
from paddle import nn
from paddle.vision.datasets import DatasetFolder
paddle.seed(42)                                      # 固定随机数，结果可以复现

def load_wav(path):                                  # 告诉 DatasetFolder 怎样读一个声音文件
    wav, sr = paddle.audio.load(path)                # wav 的形状是 (1, 采样点个数)
    x = wav[0].numpy()[:8000]                        # 最多取 8000 个数（1 秒）
    return np.pad(x, (0, 8000 - len(x))).astype("float32")   # 不够的补 0

full = DatasetFolder("../../data/speech_commands_small", loader=load_wav, extensions=(".wav",))
names = full.classes
train_ds, val_ds = paddle.io.random_split(full, [640, 160])   # 640 条训练，160 条验证
print("8 个词：", names)

# %% 第 2 步：搭网络：第一块积木把声音变成声谱图，后面就是第 4 章的卷积网络
net = nn.Sequential(
    paddle.audio.features.LogMelSpectrogram(sr=8000, n_fft=256, hop_length=128,
                                            n_mels=40),   # 声音 → 声谱图（40 个频段 × 63 个时间片）
    nn.Unflatten(1, [1, 40]),                        # 加一个“通道”，当成一张单色图片
    nn.BatchNorm2D(1),                               # 把数值调整到合适的范围，训练更稳定
    nn.Conv2D(1, 16, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Conv2D(16, 32, 3), nn.ReLU(), nn.MaxPool2D(2),
    nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(32 * 8 * 14, 64), nn.ReLU(),
    nn.Linear(64, 8), nn.Softmax(),                  # 8 个词
)
model = paddle.Model(net)
model.summary((1, 8000))

# %% 第 3 步：训练（和第 5 章完全一样）
model.prepare(paddle.optimizer.Adam(parameters=net.parameters()),
              nn.CrossEntropyLoss(use_softmax=False), paddle.metric.Accuracy())
model.fit(train_ds, val_ds, epochs=20, batch_size=32, shuffle=True, verbose=1, log_freq=100)  # ← 改这里：轮次

# %% 第 4 步：听几条验证集的录音，看模型认成什么
for i in range(8):                                   # 看前 8 条
    x, y = val_ds[i]
    p = net(paddle.to_tensor(x[None]))[0].numpy()    # x[None] 表示“只有一条的一批”
    print(f"真实：{names[int(y)]:6s} 预测：{names[p.argmax()]:6s}（把握 {p.max():.0%}）")
