# 实验 11-3（通用版）导出 ONNX，比一比大小和速度
# %% 第 1 步：把第 7 章的模型导出成 ONNX 格式
import os, time
import numpy as np
import keras
model = keras.models.load_model("../../ch07/keras/flowers_transfer.keras")   # 先完成实验 7-1
x = np.array(keras.utils.load_img("../../data/test_photos/roses_1.jpg",
                                  target_size=(128, 128)), dtype="float32")[None]   # 一张测试照片
print("Keras 预测：", model.predict(x, verbose=0)[0].round(3))   # 导出前要先让模型运行一次
model.export("flowers.onnx", format="onnx")              # 先运行 !pip install tf2onnx onnx onnxruntime
print("Keras 模型：", round(os.path.getsize("../../ch07/keras/flowers_transfer.keras") / 1e6, 1), "MB")
print("ONNX 模型：", round(os.path.getsize("flowers.onnx") / 1e6, 1), "MB")

# %% 第 2 步：用 ONNX Runtime 运行，和 Keras 比一比速度
import onnxruntime as ort
sess = ort.InferenceSession("flowers.onnx")              # 加载 ONNX 模型，不需要 Keras
name = sess.get_inputs()[0].name                         # 输入的名字
print("ONNX  预测：", sess.run(None, {name: x})[0][0].round(3))   # 结果应该几乎一样

def timeit(f, n=100):                                    # 运行 n 次，算平均每次多少毫秒
    f(); start = time.time()
    for _ in range(n):
        f()
    return (time.time() - start) / n * 1000

print(f"Keras 每张 {timeit(lambda: model.predict(x, verbose=0)):.1f} 毫秒")
print(f"ONNX  每张 {timeit(lambda: sess.run(None, {name: x})):.1f} 毫秒")

# %% 第 3 步：量化：把参数从 32 位小数压缩成 8 位整数
from onnxruntime.quantization import (quantize_static, quant_pre_process,
                                      CalibrationDataReader, QuantFormat)
train_ds = keras.utils.image_dataset_from_directory("../../data/flowers_small", validation_split=0.2,
               subset="training", seed=42, image_size=(128, 128), batch_size=1)
samples = [{name: xb.numpy()} for xb, _ in train_ds.take(64)]   # 64 张训练照片，用来“校准”

class Reader(CalibrationDataReader):                     # 固定写法：把校准照片一张一张交出去
    def __init__(self): self.it = iter(samples)
    def get_next(self): return next(self.it, None)

quant_pre_process("flowers.onnx", "flowers_pre.onnx")    # 量化前的准备（固定写法）
quantize_static("flowers_pre.onnx", "flowers_int8.onnx", Reader(),
                quant_format=QuantFormat.QDQ, per_channel=True)
print("量化后：", round(os.path.getsize("flowers_int8.onnx") / 1e6, 1), "MB")
sess8 = ort.InferenceSession("flowers_int8.onnx")
print(f"量化后每张 {timeit(lambda: sess8.run(None, {name: x})):.1f} 毫秒")

# %% 第 4 步：量化后准确率掉了多少？在验证集上比一比
val_ds = keras.utils.image_dataset_from_directory("../../data/flowers_small", validation_split=0.2,
                                                  subset="validation", seed=42, image_size=(128, 128))
right32 = right8 = total = 0
for xb, yb in val_ds:
    xb, yb = xb.numpy(), yb.numpy()
    right32 += (sess.run(None, {name: xb})[0].argmax(1) == yb).sum()
    right8 += (sess8.run(None, {name: xb})[0].argmax(1) == yb).sum()
    total += len(yb)
print(f"验证准确率：原模型 {right32 / total:.2%}，量化后 {right8 / total:.2%}")
