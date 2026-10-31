# 实验 3-1（国产版）把一张图片变成数字
# %% 第 1 步：读入一张照片，看看它有多大
import numpy as np                              # numpy：处理一大堆数字的工具
import paddle
from PIL import Image                           # PIL：Python 常用的图片工具
PHOTO = "../../data/test_photos/roses_1.jpg"    # ← 改这里：换一张照片
img = Image.open(PHOTO).convert("RGB")          # 读入照片，统一成红绿蓝三色
print("照片大小（宽, 高）：", img.size)

# %% 第 2 步：照片变成数字
x = np.array(img, dtype="float32")              # 变成一个三维的数字表格
print("形状：", x.shape)                        # （高, 宽, 3 个颜色通道）
print("左上角那个点的红、绿、蓝：", x[0, 0])     # ← 改这里：换个位置看看，如 x[100, 100]
print("最小值：", x.min(), " 最大值：", x.max())

# %% 第 3 步：整理成模型要的样子
from paddle.vision import transforms as T
tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])  # 缩放到 128×128，再变成 0～1 的张量
t = tf(img)
batch = t.unsqueeze(0)                          # 包装成“一批”，这一批只有 1 张
print("缩放后：", t.shape, " 一批：", batch.shape)   # 飞桨把颜色通道放在最前面
print("数值范围：", float(t.min()), "～", float(t.max()))

# %% 第 4 步（选做）：文字和声音也能变成数字
vocab = {"花": 1, "很": 2, "好看": 3, "不": 4, "新鲜": 5}   # 给每个词一个编号
sentence = ["花", "很", "新鲜"]                             # ← 改这里：换一句话
print("文字变数字：", [vocab[w] for w in sentence])
from scipy.io import wavfile                                # 读声音文件的工具
rate, wave = wavfile.read("../../data/speech_commands_small/yes/004ae714_nohash_0.wav")
print("声音：每秒", rate, "个数字，一共", wave.shape[0], "个，前 5 个：", wave[:5])
