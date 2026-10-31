---
title: 附录 D 术语表
description: 本书主要术语的中英文名称、简明定义和例子，按章节顺序排列
---

# 附录 D 术语表

本表按章节顺序列出本书的主要术语，每个术语给出英文名称、简明定义和例子。正文中带虚线下划线的术语，把鼠标停在上面就能看到定义，点击可以跳到本表。

通用术语（如“过拟合”“精确率”）的译名和定义以本系列教材共用的《系列公共术语表》为准，各书保持一致。

## 第 1 章 认识人工智能开发框架

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-人工智能开发框架"></span>人工智能开发框架 | AI Development Framework | 把数字计算、自动求导、GPU 加速、常用网络层和模型保存等功能做成现成工具的软件，也叫深度学习框架 | Keras、PyTorch、飞桨、昇思 |
| <span id="g-标签"></span>标签 | Label | 样本的正确答案，也就是希望模型学会预测的那个量 | 一张花的照片标着“玫瑰” |
| <span id="g-模型"></span>模型 | Model | 机器学习从数据中学到的、能把输入转换成预测结果的规律；在框架里就是一个会学习的程序 | 输入一张照片，输出“郁金香的可能性 92%”的花卉识别模型 |
| <span id="g-训练"></span>训练 | Training | 把数据和标签反复交给模型，让模型对照答案不断调整自己、学会规律的过程 | 用 600 张标好的花卉照片训练 10 轮 |
| <span id="g-推理"></span>推理 | Inference | 用训练好的模型对新数据给出预测结果的过程 | 顾客拍一张照片，模型回答“这是向日葵” |
| <span id="g-TensorFlow"></span>TensorFlow | TensorFlow | 谷歌公司 2015 年开源的深度学习框架，本书通用版中 Keras 的计算“发动机” | `import tensorflow as tf` |
| <span id="g-Keras"></span>Keras | Keras | 简单易用的高层深度学习接口，可以运行在 TensorFlow、PyTorch、JAX 上，本书通用版的主要框架 | `keras.Sequential([...])` 几行搭出一个网络 |
| <span id="g-PyTorch"></span>PyTorch | PyTorch | Meta 公司 2016 年推出的深度学习框架，写法灵活，开源模型大多用它编写 | `import torch` |
| <span id="g-飞桨"></span>飞桨 | PaddlePaddle | 百度公司 2016 年开源的国产深度学习平台，产业套件丰富，本书国产版的主要框架 | `import paddle`；PaddleOCR 文字识别 |
| <span id="g-昇思"></span>昇思 | MindSpore | 华为公司 2020 年开源的国产深度学习框架，与昇腾芯片配合紧密 | 在昇腾服务器上训练模型 |
| <span id="g-CPU"></span>CPU | Central Processing Unit | 中央处理器，电脑的“大脑”，核心少但每个核心能力强 | 笔记本电脑的 4 核处理器 |
| <span id="g-GPU"></span>GPU | Graphics Processing Unit | 图形处理器（显卡），核心多达几千个，适合同时做大量简单计算，常用来训练模型 | 学校服务器上的 A40 显卡 |
| <span id="g-云算力"></span>云算力 | Cloud Computing Power | 通过网络租用云平台机房里的 CPU、GPU 等计算资源，按使用时长计费 | 在 AI Studio 上用每天的免费 GPU 时长 |
| <span id="g-Notebook"></span>Notebook | Notebook | 把代码、运行结果和文字说明放在同一页面、按单元格逐段运行的编程环境，文件扩展名 .ipynb | 魔搭社区的云端 Jupyter Notebook |
| <span id="g-单元格"></span>单元格 | Cell | Notebook 中的一个代码块或文字块，可以单独运行，结果显示在它下方 | 按 Shift + Enter 运行一个单元格 |
| <span id="g-变量"></span>变量 | Variable | 给数据起的名字，像贴了标签的盒子，用 = 把值放进去 | `hours = 72` |
| <span id="g-函数"></span>函数 | Function | 完成某个功能、可以反复调用的一段程序，用“名字(参数)”的形式使用 | `print("你好")` |
| <span id="g-注释"></span>注释 | Comment | 代码中 # 后面写给人看的说明，运行时被跳过 | `# 把 0～255 缩到 0～1` |

## 第 2 章 第一次用框架：调用现成模型

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-预训练模型"></span>预训练模型 | Pretrained Model | 别人已经用大量数据训练好、可以直接下载使用的模型 | 在 ImageNet 上训练好的 MobileNetV2 |
| <span id="g-ImageNet"></span>ImageNet | ImageNet | 一个有 1000 个类别、约 128 万张图片的大型图片数据集，很多图像模型都在它上面预训练 | 类别里有“雏菊”“花瓶”“蜜蜂” |
| <span id="g-模型参数"></span>模型参数 | Model Parameter | 模型在训练中学到、保存在文件里的大量数字，决定模型怎样答题，也叫权重 | MobileNetV2 约有 350 万个参数 |
| <span id="g-置信度"></span>置信度 | Confidence | 模型对某个答案有多大把握，用 0～1 或百分比表示 | “雏菊 79%”中的 79% |
| <span id="g-Top-5"></span>Top-5 | Top-5 | 模型给出的把握最大的前 5 个答案 | 雏菊、蜜蜂、花瓶、花盆、尖桩篱栅 |
| <span id="g-文字识别"></span>文字识别 | Optical Character Recognition, OCR | 让模型从图片里找出文字并把它变成可编辑文本的技术 | 拍一张小票，得到“合计 58 元” |
| <span id="g-模型社区"></span>模型社区 | Model Hub | 集中发布、下载和在线体验各种模型的网站 | 魔搭社区、飞桨 AI Studio、Hugging Face |
| <span id="g-参数"></span>参数 | Argument | 调用函数时写在括号里交给函数的值 | `print("你好")` 中的 "你好" |
| <span id="g-关键字参数"></span>关键字参数 | Keyword Argument | 用“名字=值”形式写的参数，不用记顺序 | `top=5` |
| <span id="g-字符串"></span>字符串 | String | 用引号括起来的一段文字 | `"roses_1.jpg"` |
| <span id="g-列表"></span>列表 | List | 用方括号括起来、按顺序存放多个值的容器 | `["daisy_1.jpg", "roses_1.jpg"]` |
| <span id="g-for-循环"></span>for 循环 | for Loop | 把列表中的每个元素依次取出来、对它做同样操作的写法 | `for name in photos:` 逐张识别照片 |

## 第 3 章 数据在框架里的样子

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-通道"></span>通道 | Channel | 图片中同一种颜色的数字排成的一张表；彩色图片有红、绿、蓝 3 个通道，黑白图片只有 1 个 | 形状 (128, 128, 3) 中的 3 |
| <span id="g-形状"></span>形状 | Shape | 张量每个方向上有多少个数，写成括号里的一组数 | 一批 32 张彩色照片的形状 (32, 128, 128, 3) |
| <span id="g-归一化"></span>归一化 | Normalization | 把数据按比例缩放到一个统一的小范围里（如 0～1），让模型更容易学 | 像素值除以 255 |
| <span id="g-张量"></span>张量 | Tensor | 框架中存放数据的多维数字容器，可以看成一层套一层的数字表 | 一张照片、一句话的编号、一段录音 |
| <span id="g-批"></span>批 | Batch | 一次交给模型处理的一小组样本；每批的数量叫批大小 | 每批 32 张照片 |
| <span id="g-数据集"></span>数据集 | Dataset | 为训练和评估模型而收集整理的一批样本及其标签 | 5 类花各 150 张的 flowers_small |
| <span id="g-训练集"></span>训练集 | Training Set | 用来训练模型、让模型从中学习规律的那部分数据 | 750 张中的 600 张 |
| <span id="g-验证集"></span>验证集 | Validation Set | 训练时不给模型学、专门用来检查模型学得怎么样的那部分数据 | 750 张中留出的 150 张 |
| <span id="g-数据增强"></span>数据增强 | Data Augmentation | 训练时对样本做随机翻转、旋转、缩放等小变化，让模型见到更多样的数据 | RandomFlip、RandomRotation |
| <span id="g-索引"></span>索引 | Index | 用方括号和位置编号取出数据中某个元素的写法，编号从 0 开始 | `names[2]`、`x[0, 0]` |
| <span id="g-字典"></span>字典 | Dictionary | 用花括号存放“键: 值”对、按键查值的容器 | `{"roses": "玫瑰"}` |

## 第 4 章 搭积木：构建神经网络

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-神经元"></span>神经元 | Neuron | 神经网络中最小的计算单元，把输入的数各乘一个权重再相加、加上偏置，经过激活函数后输出一个数 | 总分 = 花瓣长 × 0.8 + 花瓣宽 × 0.5 + 0.1 |
| <span id="g-神经网络"></span>神经网络 | Neural Network | 由许多神经元分层连接而成的模型，数据从输入层经过若干隐藏层传到输出层 | 输入 4 个数 → 隐藏层 8 个神经元 → 输出 3 个概率 |
| <span id="g-激活函数"></span>激活函数 | Activation Function | 神经元算出总分后再经过的一道变换，给网络加入“拐弯”的能力；最常用的是 ReLU | ReLU(-2) = 0，ReLU(3.5) = 3.5 |
| <span id="g-全连接层"></span>全连接层 | Fully Connected Layer | 每个神经元都和上一层所有输出相连的层；Keras 中叫 Dense，飞桨中叫 Linear | `layers.Dense(64)`、`nn.Linear(12544, 64)` |
| <span id="g-Softmax"></span>Softmax | Softmax | 把一组分数变成加起来等于 1 的概率的函数，常用在分类网络的输出层 | 输出 [0.1, 0.7, 0.2] |
| <span id="g-卷积层"></span>卷积层 | Convolutional Layer | 用许多个小窗口（卷积核）在图片上逐格滑动、寻找局部特征的层；Keras 中叫 Conv2D | `layers.Conv2D(16, 3)`：16 个 3×3 的窗口 |
| <span id="g-池化层"></span>池化层 | Pooling Layer | 把图片按小方块（如 2×2）缩小、只保留每块中最大值的层，用来缩小数据、保留明显特征 | 126×126 变成 63×63 |
| <span id="g-参数量"></span>参数量 | Number of Parameters | 模型中所有要在训练中学习的数（权重和偏置）的总个数，反映模型的大小 | 花卉网络 826,789 个参数 |
| <span id="g-对象"></span>对象 | Object | 程序中把数据和它能做的操作打包在一起的东西，如搭好的模型 model | `model = keras.Sequential([...])` |
| <span id="g-方法"></span>方法 | Method | 属于某个对象的函数，用“对象.方法名()”调用 | `model.summary()` |

## 第 5 章 训练模型：fit 背后发生了什么

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-损失函数"></span>损失函数 | Loss Function | 衡量模型的预测和标准答案相差多少的计算方法，算出的数叫损失，越小越好 | 分类用交叉熵；五选一瞎猜时损失约 1.61 |
| <span id="g-优化器"></span>优化器 | Optimizer | 根据损失调整模型参数的方法；常用的有 SGD、Adam | `keras.optimizers.Adam(learning_rate=0.001)` |
| <span id="g-准确率"></span>准确率 | Accuracy | 预测正确的样本数占全部样本数的比例 | 150 张验证照片认对 82 张，准确率约 54.7% |
| <span id="g-轮次"></span>轮次 | Epoch | 把整个训练集从头到尾学一遍叫一轮；epochs 设定一共学几轮 | `epochs=10`，600 张照片学 10 遍 |
| <span id="g-反向传播"></span>反向传播 | Backpropagation | 从损失出发、从输出层往输入层逐层倒推，算出每个参数的梯度的方法 | `loss.backward()` |
| <span id="g-梯度"></span>梯度 | Gradient | 每个参数该往哪个方向改、改多少才能让损失变小的指示，由反向传播算出 | 梯度为正，说明这个参数应该调小 |
| <span id="g-学习率"></span>学习率 | Learning Rate | 每次更新参数时迈出的步子大小，太大学不稳，太小学得慢 | Adam 的默认学习率 0.001 |

## 第 6 章 看懂训练曲线，学会调参

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-训练曲线"></span>训练曲线 | Training Curve | 把每一轮训练集和验证集的损失、准确率画成的折线图，用来判断训练情况 | `plt.plot(history.history["val_loss"])` |
| <span id="g-欠拟合"></span>欠拟合 | Underfitting | 模型没学到规律，训练集和验证集上成绩都差的现象 | 学习率 0.00001 训练 30 轮，训练、验证准确率都在五成左右 |
| <span id="g-过拟合"></span>过拟合 | Overfitting | 模型在训练集上表现好、在新数据上表现差的现象，好比死记硬背 | 训练准确率 100%，验证准确率只有 53% |
| <span id="g-回调"></span>回调 | Callback | 交给 fit 的“助手”，在每一轮训练前后自动执行某些操作，如早停、保存模型、记录日志 | `callbacks=[EarlyStopping(...)]` |
| <span id="g-早停"></span>早停 | Early Stopping | 验证成绩连续若干轮没有变好就自动停止训练，并保留最好那一轮的模型 | `patience=5`：连续 5 轮验证损失没降就停 |
| <span id="g-Dropout"></span>Dropout | Dropout | 训练时每一批随机让一部分神经元暂时不工作，防止网络过分依赖某几个神经元，从而减轻过拟合 | `layers.Dropout(0.5)` |
| <span id="g-超参数"></span>超参数 | Hyperparameter | 训练前由人设定、不由模型自己学习的设置，如学习率、轮次、批大小、Dropout 比例 | 学习率 0.001、轮次 30 |

## 第 7 章 迁移学习：站在别人的肩膀上

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-迁移学习"></span>迁移学习 | Transfer Learning | 把在一个任务（通常是大数据集）上训练好的模型，拿来作为新任务的起点继续训练的方法 | 用 ImageNet 预训练的 MobileNetV2 认 5 种花 |
| <span id="g-特征"></span>特征 | Feature | 网络从输入数据中提炼出来、对完成任务有用的信息，如边缘、纹理、形状 | MobileNetV2 从一张照片提炼出的 1280 个数 |
| <span id="g-冻结"></span>冻结 | Freezing | 训练时让某些层的参数保持不变、不参与更新 | `base.trainable = False` |
| <span id="g-微调"></span>微调 | Fine-tuning | 在迁移学习中解冻预训练模型的部分层，用很小的学习率继续训练，让它更适应新任务 | 解冻最后 30 层，学习率 0.00001 |
| <span id="g-混淆矩阵"></span>混淆矩阵 | Confusion Matrix | 按“实际类别 × 预测类别”交叉统计样本数量的表格，用来看模型把哪些类别搞混了 | 郁金香那一行有 4 张被认成玫瑰 |

## 第 8 章 图像应用：检测与文字识别

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-目标检测"></span>目标检测 | Object Detection | 找出图片中有哪些物体、各在什么位置的任务，每个物体用一个框加类别和置信度表示 | YOLO 在街景照片里框出 3 辆车、2 辆自行车 |
| <span id="g-边界框"></span>边界框 | Bounding Box | 检测结果中框住一个物体的矩形，通常用左上角和右下角两个点的坐标表示 | car 的框 (243,43)-(310,69)，宽 67、高 26 像素 |
| <span id="g-图像分割"></span>图像分割 | Image Segmentation | 判断图片中每个像素属于哪个物体或类别，得到物体精确轮廓的任务 | 把照片里的两辆自行车精确地“抠”出来 |
| <span id="g-置信度阈值"></span>置信度阈值 | Confidence Threshold | 只保留置信度不低于某个值的结果的门槛；调低多报、调高少报 | conf=0.4：把握低于 40% 的框不显示 |
| <span id="g-COCO-数据集"></span>COCO 数据集 | Common Objects in Context, COCO | 微软发布的常用目标检测数据集，标注了人、汽车、杯子等 80 类日常物体 | YOLO11 在 COCO 上训练，所以不认识“花” |
| <span id="g-文字检测"></span>文字检测 | Text Detection | 文字识别的第一步：在图片中找出哪里有文字，并把每一块文字框出来 | 在进货单上框出“红玫瑰”“50”“3.50”等文字块 |
| <span id="g-结构化数据"></span>结构化数据 | Structured Data | 按行和列组织、每一列有固定含义的数据，便于计算、查询和导入系统 | OCR 结果整理成“品名、数量、单价、金额”四列的表格 |

## 第 9 章 文本应用：让模型读懂文字

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-分词"></span>分词 | Tokenization | 把一段文字切分成一个个小单位（字、词或词元）的过程，是文字变数字的第一步 | “骑手态度很好”按词切成“骑手 / 态度 / 很 / 好” |
| <span id="g-词表"></span>词表 | Vocabulary | 训练数据中出现的所有字或词与编号的对照表；本书 0 留给填充、1 留给词表外的字 | “很”→ 4，“好”→ 7，没见过的“惜”→ 1 |
| <span id="g-填充"></span>填充 | Padding | 在长度不够的序列后面补 0，使每条数据长度相同；超出的部分则截断 | 9 个字的评论补 31 个 0，变成 40 个数 |
| <span id="g-词向量"></span>词向量 | Word Embedding | 用一串数表示一个字或词，由嵌入层在训练中学出来，意思相近的字或词彼此靠近 | 每个字用 16 个数表示，嵌入层参数 = 字数 × 16 |
| <span id="g-情感分析"></span>情感分析 | Sentiment Analysis | 判断一段文字表达的是正面还是负面态度的任务 | “味道很好，下次还点”→ 好评 |
| <span id="g-大语言模型"></span>大语言模型 | Large Language Model, LLM | 参数规模巨大、在海量文本上预训练、具有通用语言理解与生成能力的模型 | DeepSeek、通义千问、文心一言 |
| <span id="g-提示词"></span>提示词 | Prompt | 给大模型的指令和输入，一般要说清角色、任务和输出格式 | “你是评论审核员……只回答‘好评’或‘差评’” |
| <span id="g-API"></span>API | Application Programming Interface | 应用程序接口：程序之间约定好的调用方式，程序把请求发给服务，服务把结果返回 | 用 openai 库把评论发给大模型平台，收到“差评” |

## 第 10 章 语音应用：让模型听懂声音

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-采样率"></span>采样率 | Sample Rate | 每秒对声音信号记录多少个数值，单位赫兹（Hz）；越高声音越细致，数据也越多 | 8000 Hz：1 秒的录音有 8000 个数 |
| <span id="g-波形"></span>波形 | Waveform | 把声音的采样值按时间顺序画出来的图，能看出声音何时开始、何时结束、哪里大哪里小 | “yes”在 0.43 秒开始，“ye”声音大，“s”声音小而密 |
| <span id="g-声谱图"></span>声谱图 | Spectrogram | 横轴为时间、纵轴为频率、亮度表示强弱的图，展示每一小段时间里各频率声音的强弱，可以当图片来识别 | “no”的声谱图整段都是清楚的横纹 |
| <span id="g-语音识别"></span>语音识别 | Automatic Speech Recognition, ASR | 把说话的声音转换成文字的技术 | 对手机说“下一单”，得到文字“下一单” |
| <span id="g-语音合成"></span>语音合成 | Text to Speech, TTS | 把文字转换成说话声音的技术 | 把“您有新的外卖订单”变成语音播报 |
| <span id="g-指令词识别"></span>指令词识别 | Keyword Spotting | 只识别少数几个固定词的语音任务，常用于语音唤醒和语音控制，模型小、反应快 | 听懂 up、down、left、right 等 8 个指令词 |

## 第 11 章 把模型做成应用

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-部署"></span>部署 | Deployment | 把训练好的模型放到实际使用的环境中（网页、接口服务、手机、设备等），让用户能用上 | 把花卉识别模型发布成网页，店员用手机打开就能识别 |
| <span id="g-Gradio"></span>Gradio | Gradio | 为机器学习模型快速生成网页的 Python 库，只需提供预测函数和输入、输出组件 | gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label()) |
| <span id="g-ONNX"></span>ONNX | Open Neural Network Exchange | 开放的神经网络模型交换格式，各框架训练的模型都可转换成它，用 ONNX Runtime 运行而不需要原框架 | Keras 模型导出为 flowers.onnx，预测一张约 1 毫秒 |
| <span id="g-量化"></span>量化 | Quantization | 用更少的位数（如 8 位整数代替 32 位小数）存储模型参数，使模型变小，在合适的硬件上更快 | 花卉模型量化后从 8.9 MB 变成 2.7 MB，准确率不变 |

## 第 12 章 国产 AI 生态、数据安全与综合项目

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
