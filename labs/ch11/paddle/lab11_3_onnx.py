# 实验 11-3（国产版）导出 ONNX，比一比大小和速度
# %% 第 1 步：把第 7 章的模型导出成 ONNX 格式
import os, time
import numpy as np
import paddle
from paddle import nn
from paddle.vision import transforms as T
from paddle.vision.datasets import DatasetFolder
net = paddle.vision.models.mobilenet_v2()                # 搭出和训练时一样的网络，再装上参数
net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())
net.set_state_dict(paddle.load("../../ch07/paddle/flowers_transfer.pdparams"))   # 先完成实验 7-1
net.eval()
spec = paddle.static.InputSpec([None, 3, 128, 128], "float32", "x")   # 输入的形状：几张都行 × 3 × 128 × 128
paddle.onnx.export(net, "flowers", input_spec=[spec], opset_version=13)   # 先运行 !pip install paddle2onnx onnxruntime
print("飞桨参数文件：", round(os.path.getsize("../../ch07/paddle/flowers_transfer.pdparams") / 1e6, 1), "MB")
print("ONNX 模型：", round(os.path.getsize("flowers.onnx") / 1e6, 1), "MB")

# %% 第 2 步：用 ONNX Runtime 运行，和飞桨比一比速度
import onnxruntime as ort
tf = T.Compose([T.Resize((128, 128)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
from PIL import Image
x = tf(Image.open("../../data/test_photos/roses_1.jpg").convert("RGB"))[None].numpy()   # 一张测试照片
sess = ort.InferenceSession("flowers.onnx")              # 加载 ONNX 模型，不需要飞桨
print("飞桨预测：", net(paddle.to_tensor(x))[0].numpy().round(3))
print("ONNX 预测：", sess.run(None, {"x": x})[0][0].round(3))    # 结果应该几乎一样

def timeit(f, n=100):                                    # 运行 n 次，算平均每次多少毫秒
    f(); start = time.time()
    for _ in range(n):
        f()
    return (time.time() - start) / n * 1000

print(f"飞桨每张 {timeit(lambda: net(paddle.to_tensor(x))):.1f} 毫秒")
print(f"ONNX 每张 {timeit(lambda: sess.run(None, {'x': x})):.1f} 毫秒")

# %% 第 3 步：量化：把参数从 32 位小数压缩成 8 位整数
from onnxruntime.quantization import (quantize_static, quant_pre_process,
                                      CalibrationDataReader, QuantFormat)
paddle.seed(42)
full = DatasetFolder("../../data/flowers_small", transform=tf)
train_ds, val_ds = paddle.io.random_split(full, [600, 150])     # 和实验 7-1 一样的划分
samples = [{"x": train_ds[i][0][None].numpy()} for i in range(64)]   # 64 张训练照片，用来“校准”

class Reader(CalibrationDataReader):                     # 固定写法：把校准照片一张一张交出去
    def __init__(self): self.it = iter(samples)
    def get_next(self): return next(self.it, None)

quant_pre_process("flowers.onnx", "flowers_pre.onnx")    # 量化前的准备（固定写法）
quantize_static("flowers_pre.onnx", "flowers_int8.onnx", Reader(),
                quant_format=QuantFormat.QDQ, per_channel=True)
print("量化后：", round(os.path.getsize("flowers_int8.onnx") / 1e6, 1), "MB")
sess8 = ort.InferenceSession("flowers_int8.onnx")
print(f"量化后每张 {timeit(lambda: sess8.run(None, {'x': x})):.1f} 毫秒")

# %% 第 4 步：量化后准确率掉了多少？在验证集上比一比
right32 = right8 = 0
for img, label in val_ds:
    right32 += int(sess.run(None, {"x": img[None].numpy()})[0].argmax() == int(label))
    right8 += int(sess8.run(None, {"x": img[None].numpy()})[0].argmax() == int(label))
print(f"验证准确率：原模型 {right32 / len(val_ds):.2%}，量化后 {right8 / len(val_ds):.2%}")
