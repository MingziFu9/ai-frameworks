"""编者用来生成随书小数据集的脚本（学生不需要运行）。

原始数据：
- 花卉图片 flower_photos（TensorFlow 官方教程数据，CC BY 2.0）
  https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- 语音指令 mini_speech_commands（Google Speech Commands 的子集，CC BY 4.0）
  https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip
运行：python make_datasets.py 原始数据所在文件夹
"""
import random
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT = Path(__file__).parent
random.seed(7)

# 1. 花卉：每类 150 张做训练和验证（flowers_small），另取每类 2 张做“新照片”（test_photos）
flower_src = SRC / "flower_photos"
credits = {}
for line in (flower_src / "LICENSE.txt").read_text(encoding="utf-8").splitlines():
    if "/" in line and " CC-BY" in line:
        credits[line.split(" CC-BY")[0].strip()] = line.strip()
lines = ["本文件夹的图片选自 TensorFlow 官方花卉数据集 flower_photos，按 CC BY 2.0 许可使用：",
         "https://creativecommons.org/licenses/by/2.0/ ，编者把图片缩小到长边 200 像素。各图作者如下：", ""]
for cls in ["daisy", "dandelion", "roses", "sunflowers", "tulips"]:
    files = sorted((flower_src / cls).glob("*.jpg"))
    random.shuffle(files)
    for sub, chosen in [("flowers_small", files[:150]), ("test_photos", files[150:152])]:
        for i, f in enumerate(chosen):
            img = Image.open(f).convert("RGB")
            img.thumbnail((200, 200))
            dst = OUT / sub / cls / f.name if sub == "flowers_small" else OUT / sub / f"{cls}_{i + 1}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, quality=85)
            lines.append(f"{sub}/{dst.relative_to(OUT / sub)}  ←  " + credits.get(f"{cls}/{f.name}", f"{cls}/{f.name}"))
(OUT / "flowers_small" / "LICENSE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# 2. 语音指令：8 个词每个 100 条，重采样到 8000 Hz、单声道 16 位
speech_src = SRC / "mini_speech_commands"
for word in ["down", "go", "left", "no", "right", "stop", "up", "yes"]:
    files = sorted((speech_src / word).glob("*.wav"))
    random.shuffle(files)
    for f in files[:100]:
        with wave.open(str(f)) as w:
            sr, n = w.getframerate(), w.getnframes()
            x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
        t_new = np.arange(0, len(x) / sr, 1 / 8000)
        y = np.interp(t_new, np.arange(len(x)) / sr, x).astype(np.int16)
        dst = OUT / "speech_commands_small" / word / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(y.tobytes())
print("完成")
