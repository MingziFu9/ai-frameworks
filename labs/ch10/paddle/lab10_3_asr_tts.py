# 实验 10-3（国产版）语音合成与语音识别体验
# %% 第 1 步：下载模型（约 300 MB，只需下载一次；老师也可以提前下载好放在共享目录）
import os, tarfile, urllib.request
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
for url in [BASE + "tts-models/matcha-icefall-zh-baker.tar.bz2",          # 语音合成模型（中文女声）
            BASE + "vocoder-models/vocos-22khz-univ.onnx",                 # 合成用的“声码器”
            BASE + "asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2"]:   # 语音识别：阿里达摩院 FunASR 的 Paraformer 中文模型（小号版）
    name = url.split("/")[-1]
    if not os.path.exists(name):
        print("正在下载", name); urllib.request.urlretrieve(url, name)
        if name.endswith(".tar.bz2"):
            tarfile.open(name).extractall()          # 解压

# %% 第 2 步：语音合成：把文字变成声音
import sherpa_onnx                                   # 先运行 !pip install sherpa-onnx soundfile
import soundfile as sf
from IPython.display import Audio, display
M = "matcha-icefall-zh-baker/"
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
        acoustic_model=M + "model-steps-3.onnx", vocoder="vocos-22khz-univ.onnx",
        lexicon=M + "lexicon.txt", tokens=M + "tokens.txt", dict_dir=M + "dict")),
    rule_fsts=M + "number.fst"))                     # 这一段是固定写法，照抄即可
TEXT = "您好，您的外卖已经送到门口了。"               # ← 改这里：想让它说什么
audio = tts.generate(TEXT, sid=0, speed=0.9)         # speed：语速
sf.write("tts.wav", audio.samples, audio.sample_rate)   # 存成声音文件
display(Audio("tts.wav"))                            # 播放

# %% 第 3 步：语音识别：把声音变回文字
P = "sherpa-onnx-paraformer-zh-small-2024-03-09/"
asr = sherpa_onnx.OfflineRecognizer.from_paraformer(
    paraformer=P + "model.int8.onnx", tokens=P + "tokens.txt")

def listen(path):
    x, sr = sf.read(path, dtype="float32")           # 读声音文件
    s = asr.create_stream(); s.accept_waveform(sr, x); asr.decode_stream(s)
    return s.result.text

print("原文：", TEXT)
print("识别：", listen("tts.wav"))                    # ← 改这里：换成自己录的声音文件
