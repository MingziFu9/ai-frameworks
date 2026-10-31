# app.py：发布到魔搭创空间 / AI Studio / Hugging Face Spaces 用的花卉识别网页（实验 11-2）

import numpy as np
import keras
model = keras.models.load_model("flowers_transfer.keras")   # 把实验 7-1 保存的模型文件和本文件放在一起
NAMES = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]   # 和训练时的类别顺序一致

def predict(img):
    img = img.convert("RGB").resize((128, 128))          # 统一成训练时的大小
    x = np.array(img, dtype="float32")[None]             # [None]：变成“只有一张的一批”
    p = model.predict(x, verbose=0)[0]
    return {NAMES[i]: float(p[i]) for i in range(5)}     # 字典：花名 → 概率

import gradio as gr                                      # 先运行 !pip install gradio
demo = gr.Interface(
    fn=predict,                                          # 用哪个函数
    inputs=gr.Image(type="pil", label="上传一张花的照片"),   # 输入：图片
    outputs=gr.Label(num_top_classes=3, label="识别结果"),    # 输出：前 3 名和概率
    title="花卉识别助手",                                 # ← 改这里：网页标题
    description="能认雏菊、蒲公英、玫瑰、向日葵、郁金香五种花。",
    examples=["roses_1.jpg", "sunflowers_1.jpg"],         # 两张示例照片也一起上传
)
demo.launch()
