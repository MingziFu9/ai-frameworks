# 实验 11-1（国产版）做一个花卉识别网页
# %% 第 1 步：加载第 7 章训练好的模型
import paddle
from paddle import nn
from paddle.vision import transforms as T
net = paddle.vision.models.mobilenet_v2()                # 先搭出和训练时一模一样的网络
net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())
model = paddle.Model(net)
model.load("../../ch07/paddle/flowers_transfer")         # 再装上训练好的参数（先完成实验 7-1）
net.eval()                                               # 切换到“预测模式”（Dropout 不再起作用）
NAMES = ["雏菊", "蒲公英", "玫瑰", "向日葵", "郁金香"]   # 和训练时的类别顺序一致
tf = T.Compose([T.Resize((128, 128)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   # 和训练时一样的预处理

# %% 第 2 步：写一个“预测函数”：输入一张图片，输出每种花的概率
def predict(img):
    x = tf(img.convert("RGB"))[None]                     # [None]：变成“只有一张的一批”
    p = net(x)[0].numpy()
    return {NAMES[i]: float(p[i]) for i in range(5)}     # 字典：花名 → 概率

# %% 第 3 步：用 Gradio 把函数变成网页
import gradio as gr                                      # 先运行 !pip install gradio
demo = gr.Interface(
    fn=predict,                                          # 用哪个函数
    inputs=gr.Image(type="pil", label="上传一张花的照片"),   # 输入：图片
    outputs=gr.Label(num_top_classes=3, label="识别结果"),    # 输出：前 3 名和概率
    title="花卉识别助手",                                 # ← 改这里：网页标题
    description="能认雏菊、蒲公英、玫瑰、向日葵、郁金香五种花。",
    examples=["../../data/test_photos/roses_1.jpg", "../../data/test_photos/sunflowers_1.jpg"],
)
demo.launch()                                            # 运行后点击输出的网址；云平台上加 share=True
