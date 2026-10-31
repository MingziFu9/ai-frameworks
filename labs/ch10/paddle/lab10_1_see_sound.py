# 实验 10-1（国产版）看见声音：波形和声谱图
# %% 第 1 步：读入一段录音，看看它是什么样的数字
import numpy as np
import matplotlib.pyplot as plt
import paddle
from IPython.display import Audio, display
YES = "../../data/speech_commands_small/yes/004ae714_nohash_0.wav"   # ← 改这里：换一个文件
wav, sr = paddle.audio.load(YES)                     # wav 的形状是 (1, 采样点个数)，数值在 -1～1 之间
x = wav[0].numpy()
print("采样率", sr, "，一共", len(x), "个数，时长", len(x) / sr, "秒")
print("中间 10 个数：", x[4000:4010])
display(Audio(x, rate=sr))                           # 在 Notebook 里出现一个播放器，可以听

# %% 第 2 步：画波形
plt.figure(figsize=(8, 2.5))
plt.plot(np.arange(len(x)) / sr, x)                  # 横轴是时间（秒），纵轴是数值
plt.xlabel("time (s)")
plt.show()

# %% 第 3 步：画声谱图，比较 yes 和 no
mel = paddle.audio.features.LogMelSpectrogram(sr=8000, n_fft=256, hop_length=128,
                                              n_mels=40)   # 和实验 10-2 网络的第一块积木一样
NO = "../../data/speech_commands_small/no/01bcfc0c_nohash_0.wav"
plt.figure(figsize=(8, 3))
for i, path in enumerate([YES, NO]):
    wav, sr = paddle.audio.load(path)
    x = np.pad(wav[0].numpy(), (0, 8000 - wav.shape[1])).astype("float32")   # 统一成 8000 个数
    spec = mel(paddle.to_tensor(x[None]))[0].numpy()     # 形状 [40, 63]：40 个频段 × 63 个时间片
    plt.subplot(1, 2, i + 1)
    plt.imshow(spec, origin="lower", aspect="auto")      # origin="lower"：低频画在下面
    plt.title(path.split("/")[-2])
plt.show()
