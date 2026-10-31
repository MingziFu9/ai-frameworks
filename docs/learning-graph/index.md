---
title: 学习图谱
description: 全书核心概念及其先修关系，点一个概念就能看到学习路径
---

# 学习图谱

学习图谱把全书的核心概念和它们之间的“先修关系”画在一张图上：要理解一个概念，得先懂哪些更基础的概念；顺着箭头往回找，就是一条学习路径。

<iframe class="sim-frame" src="main.html" height="900" loading="lazy" title="学习图谱"></iframe>

[全屏打开](main.html){ target=_blank }

## 怎样使用

- **预习**：学新的一章前，点开本章的概念，看看橙色的先修概念自己是否都掌握了，没掌握的先回去复习；
- **复习**：期末复习时，从最后几章的概念出发，顺着学习路径往回梳理，就能把全书串起来；
- **查漏补缺**：做练习时遇到不懂的概念，在搜索框里输入它，看它依赖哪些更基础的概念。

<!-- graph:start -->

图谱共有 **101 个概念、123 条先修关系**，概念按章排列，颜色表示 8 个类别（基本概念、框架与平台、Python 基础、数据处理、网络结构、训练与调参、应用任务、部署与合规）。概念的定义都来自[附录 D 术语表](../appendix/glossary.md)，先修关系由编者根据各章内容整理。

## 文字版

下面按章列出每个概念的直接先修概念，便于打印或用读屏软件阅读。

??? note "第 1 章 认识框架（18 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [人工智能开发框架](../appendix/glossary.md#g-人工智能开发框架) | 框架与平台 | — |
    | [标签](../appendix/glossary.md#g-标签) | 基本概念 | — |
    | [模型](../appendix/glossary.md#g-模型) | 基本概念 | — |
    | [训练](../appendix/glossary.md#g-训练) | 基本概念 | [模型](../appendix/glossary.md#g-模型)、[标签](../appendix/glossary.md#g-标签) |
    | [推理](../appendix/glossary.md#g-推理) | 基本概念 | [训练](../appendix/glossary.md#g-训练)、[模型](../appendix/glossary.md#g-模型) |
    | [TensorFlow](../appendix/glossary.md#g-TensorFlow) | 框架与平台 | [人工智能开发框架](../appendix/glossary.md#g-人工智能开发框架) |
    | [Keras](../appendix/glossary.md#g-Keras) | 框架与平台 | [TensorFlow](../appendix/glossary.md#g-TensorFlow) |
    | [PyTorch](../appendix/glossary.md#g-PyTorch) | 框架与平台 | [人工智能开发框架](../appendix/glossary.md#g-人工智能开发框架) |
    | [飞桨](../appendix/glossary.md#g-飞桨) | 框架与平台 | [人工智能开发框架](../appendix/glossary.md#g-人工智能开发框架) |
    | [昇思](../appendix/glossary.md#g-昇思) | 框架与平台 | [人工智能开发框架](../appendix/glossary.md#g-人工智能开发框架) |
    | [CPU](../appendix/glossary.md#g-CPU) | 框架与平台 | — |
    | [GPU](../appendix/glossary.md#g-GPU) | 框架与平台 | [CPU](../appendix/glossary.md#g-CPU) |
    | [云算力](../appendix/glossary.md#g-云算力) | 框架与平台 | [GPU](../appendix/glossary.md#g-GPU) |
    | [Notebook](../appendix/glossary.md#g-Notebook) | 框架与平台 | [云算力](../appendix/glossary.md#g-云算力) |
    | [单元格](../appendix/glossary.md#g-单元格) | 框架与平台 | [Notebook](../appendix/glossary.md#g-Notebook) |
    | [变量](../appendix/glossary.md#g-变量) | Python 基础 | — |
    | [函数](../appendix/glossary.md#g-函数) | Python 基础 | [变量](../appendix/glossary.md#g-变量) |
    | [注释](../appendix/glossary.md#g-注释) | Python 基础 | — |

??? note "第 2 章 调用现成模型（12 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [预训练模型](../appendix/glossary.md#g-预训练模型) | 基本概念 | [训练](../appendix/glossary.md#g-训练)、[ImageNet](../appendix/glossary.md#g-ImageNet) |
    | [ImageNet](../appendix/glossary.md#g-ImageNet) | 框架与平台 | — |
    | [模型参数](../appendix/glossary.md#g-模型参数) | 基本概念 | [模型](../appendix/glossary.md#g-模型) |
    | [置信度](../appendix/glossary.md#g-置信度) | 基本概念 | [推理](../appendix/glossary.md#g-推理) |
    | [Top-5](../appendix/glossary.md#g-Top-5) | 基本概念 | [置信度](../appendix/glossary.md#g-置信度) |
    | [文字识别](../appendix/glossary.md#g-文字识别) | 应用任务 | [预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [模型社区](../appendix/glossary.md#g-模型社区) | 框架与平台 | [预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [参数](../appendix/glossary.md#g-参数) | Python 基础 | [函数](../appendix/glossary.md#g-函数) |
    | [关键字参数](../appendix/glossary.md#g-关键字参数) | Python 基础 | [参数](../appendix/glossary.md#g-参数) |
    | [字符串](../appendix/glossary.md#g-字符串) | Python 基础 | [变量](../appendix/glossary.md#g-变量) |
    | [列表](../appendix/glossary.md#g-列表) | Python 基础 | [变量](../appendix/glossary.md#g-变量) |
    | [for 循环](../appendix/glossary.md#g-for-循环) | Python 基础 | [列表](../appendix/glossary.md#g-列表) |

??? note "第 3 章 数据与张量（11 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [通道](../appendix/glossary.md#g-通道) | 数据处理 | [形状](../appendix/glossary.md#g-形状) |
    | [形状](../appendix/glossary.md#g-形状) | 数据处理 | [索引](../appendix/glossary.md#g-索引) |
    | [归一化](../appendix/glossary.md#g-归一化) | 数据处理 | — |
    | [张量](../appendix/glossary.md#g-张量) | 数据处理 | [形状](../appendix/glossary.md#g-形状)、[归一化](../appendix/glossary.md#g-归一化) |
    | [批](../appendix/glossary.md#g-批) | 数据处理 | [张量](../appendix/glossary.md#g-张量) |
    | [数据集](../appendix/glossary.md#g-数据集) | 数据处理 | [标签](../appendix/glossary.md#g-标签) |
    | [训练集](../appendix/glossary.md#g-训练集) | 数据处理 | [数据集](../appendix/glossary.md#g-数据集) |
    | [验证集](../appendix/glossary.md#g-验证集) | 数据处理 | [数据集](../appendix/glossary.md#g-数据集) |
    | [数据增强](../appendix/glossary.md#g-数据增强) | 数据处理 | [训练集](../appendix/glossary.md#g-训练集) |
    | [索引](../appendix/glossary.md#g-索引) | Python 基础 | [列表](../appendix/glossary.md#g-列表) |
    | [字典](../appendix/glossary.md#g-字典) | Python 基础 | [列表](../appendix/glossary.md#g-列表) |

??? note "第 4 章 搭建网络（10 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [神经元](../appendix/glossary.md#g-神经元) | 网络结构 | [模型参数](../appendix/glossary.md#g-模型参数) |
    | [神经网络](../appendix/glossary.md#g-神经网络) | 网络结构 | [神经元](../appendix/glossary.md#g-神经元)、[激活函数](../appendix/glossary.md#g-激活函数) |
    | [激活函数](../appendix/glossary.md#g-激活函数) | 网络结构 | [神经元](../appendix/glossary.md#g-神经元) |
    | [全连接层](../appendix/glossary.md#g-全连接层) | 网络结构 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [Softmax](../appendix/glossary.md#g-Softmax) | 网络结构 | [全连接层](../appendix/glossary.md#g-全连接层)、[置信度](../appendix/glossary.md#g-置信度) |
    | [卷积层](../appendix/glossary.md#g-卷积层) | 网络结构 | [通道](../appendix/glossary.md#g-通道)、[神经网络](../appendix/glossary.md#g-神经网络) |
    | [池化层](../appendix/glossary.md#g-池化层) | 网络结构 | [卷积层](../appendix/glossary.md#g-卷积层) |
    | [参数量](../appendix/glossary.md#g-参数量) | 基本概念 | [模型参数](../appendix/glossary.md#g-模型参数)、[全连接层](../appendix/glossary.md#g-全连接层)、[卷积层](../appendix/glossary.md#g-卷积层) |
    | [对象](../appendix/glossary.md#g-对象) | Python 基础 | [函数](../appendix/glossary.md#g-函数) |
    | [方法](../appendix/glossary.md#g-方法) | Python 基础 | [对象](../appendix/glossary.md#g-对象) |

??? note "第 5 章 训练模型（7 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [损失函数](../appendix/glossary.md#g-损失函数) | 训练与调参 | [标签](../appendix/glossary.md#g-标签)、[Softmax](../appendix/glossary.md#g-Softmax) |
    | [优化器](../appendix/glossary.md#g-优化器) | 训练与调参 | [反向传播](../appendix/glossary.md#g-反向传播)、[学习率](../appendix/glossary.md#g-学习率) |
    | [准确率](../appendix/glossary.md#g-准确率) | 基本概念 | [验证集](../appendix/glossary.md#g-验证集)、[推理](../appendix/glossary.md#g-推理) |
    | [轮次](../appendix/glossary.md#g-轮次) | 训练与调参 | [训练集](../appendix/glossary.md#g-训练集)、[批](../appendix/glossary.md#g-批) |
    | [反向传播](../appendix/glossary.md#g-反向传播) | 训练与调参 | [梯度](../appendix/glossary.md#g-梯度) |
    | [梯度](../appendix/glossary.md#g-梯度) | 训练与调参 | [损失函数](../appendix/glossary.md#g-损失函数) |
    | [学习率](../appendix/glossary.md#g-学习率) | 训练与调参 | — |

??? note "第 6 章 训练曲线与调参（7 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [训练曲线](../appendix/glossary.md#g-训练曲线) | 训练与调参 | [轮次](../appendix/glossary.md#g-轮次)、[准确率](../appendix/glossary.md#g-准确率) |
    | [欠拟合](../appendix/glossary.md#g-欠拟合) | 训练与调参 | [训练曲线](../appendix/glossary.md#g-训练曲线) |
    | [过拟合](../appendix/glossary.md#g-过拟合) | 训练与调参 | [训练曲线](../appendix/glossary.md#g-训练曲线) |
    | [回调](../appendix/glossary.md#g-回调) | 训练与调参 | [方法](../appendix/glossary.md#g-方法) |
    | [早停](../appendix/glossary.md#g-早停) | 训练与调参 | [回调](../appendix/glossary.md#g-回调)、[过拟合](../appendix/glossary.md#g-过拟合) |
    | [Dropout](../appendix/glossary.md#g-Dropout) | 网络结构 | [过拟合](../appendix/glossary.md#g-过拟合)、[神经网络](../appendix/glossary.md#g-神经网络) |
    | [超参数](../appendix/glossary.md#g-超参数) | 训练与调参 | [学习率](../appendix/glossary.md#g-学习率)、[轮次](../appendix/glossary.md#g-轮次) |

??? note "第 7 章 迁移学习（5 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [迁移学习](../appendix/glossary.md#g-迁移学习) | 训练与调参 | [预训练模型](../appendix/glossary.md#g-预训练模型)、[特征](../appendix/glossary.md#g-特征) |
    | [特征](../appendix/glossary.md#g-特征) | 基本概念 | [卷积层](../appendix/glossary.md#g-卷积层) |
    | [冻结](../appendix/glossary.md#g-冻结) | 训练与调参 | [迁移学习](../appendix/glossary.md#g-迁移学习) |
    | [微调](../appendix/glossary.md#g-微调) | 训练与调参 | [冻结](../appendix/glossary.md#g-冻结)、[学习率](../appendix/glossary.md#g-学习率) |
    | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) | 训练与调参 | [准确率](../appendix/glossary.md#g-准确率)、[验证集](../appendix/glossary.md#g-验证集) |

??? note "第 8 章 图像应用（7 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [目标检测](../appendix/glossary.md#g-目标检测) | 应用任务 | [预训练模型](../appendix/glossary.md#g-预训练模型)、[边界框](../appendix/glossary.md#g-边界框)、[COCO 数据集](../appendix/glossary.md#g-COCO-数据集) |
    | [边界框](../appendix/glossary.md#g-边界框) | 应用任务 | — |
    | [图像分割](../appendix/glossary.md#g-图像分割) | 应用任务 | [目标检测](../appendix/glossary.md#g-目标检测) |
    | [置信度阈值](../appendix/glossary.md#g-置信度阈值) | 应用任务 | [置信度](../appendix/glossary.md#g-置信度)、[目标检测](../appendix/glossary.md#g-目标检测) |
    | [COCO 数据集](../appendix/glossary.md#g-COCO-数据集) | 框架与平台 | — |
    | [文字检测](../appendix/glossary.md#g-文字检测) | 应用任务 | [文字识别](../appendix/glossary.md#g-文字识别) |
    | [结构化数据](../appendix/glossary.md#g-结构化数据) | 数据处理 | [文字检测](../appendix/glossary.md#g-文字检测)、[字典](../appendix/glossary.md#g-字典) |

??? note "第 9 章 文本应用（8 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [分词](../appendix/glossary.md#g-分词) | 数据处理 | [字符串](../appendix/glossary.md#g-字符串) |
    | [词表](../appendix/glossary.md#g-词表) | 数据处理 | [分词](../appendix/glossary.md#g-分词) |
    | [填充](../appendix/glossary.md#g-填充) | 数据处理 | [词表](../appendix/glossary.md#g-词表)、[形状](../appendix/glossary.md#g-形状) |
    | [词向量](../appendix/glossary.md#g-词向量) | 数据处理 | [词表](../appendix/glossary.md#g-词表)、[张量](../appendix/glossary.md#g-张量) |
    | [情感分析](../appendix/glossary.md#g-情感分析) | 应用任务 | [词向量](../appendix/glossary.md#g-词向量)、[预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [大语言模型](../appendix/glossary.md#g-大语言模型) | 框架与平台 | — |
    | [提示词](../appendix/glossary.md#g-提示词) | 应用任务 | [大语言模型](../appendix/glossary.md#g-大语言模型) |
    | [API](../appendix/glossary.md#g-API) | 框架与平台 | [函数](../appendix/glossary.md#g-函数)、[大语言模型](../appendix/glossary.md#g-大语言模型) |

??? note "第 10 章 语音应用（6 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [采样率](../appendix/glossary.md#g-采样率) | 数据处理 | — |
    | [波形](../appendix/glossary.md#g-波形) | 数据处理 | [采样率](../appendix/glossary.md#g-采样率)、[张量](../appendix/glossary.md#g-张量) |
    | [声谱图](../appendix/glossary.md#g-声谱图) | 数据处理 | [波形](../appendix/glossary.md#g-波形) |
    | [语音识别](../appendix/glossary.md#g-语音识别) | 应用任务 | [声谱图](../appendix/glossary.md#g-声谱图)、[预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [语音合成](../appendix/glossary.md#g-语音合成) | 应用任务 | [预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [指令词识别](../appendix/glossary.md#g-指令词识别) | 应用任务 | [声谱图](../appendix/glossary.md#g-声谱图)、[卷积层](../appendix/glossary.md#g-卷积层) |

??? note "第 11 章 模型部署（4 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [部署](../appendix/glossary.md#g-部署) | 部署与合规 | [推理](../appendix/glossary.md#g-推理) |
    | [Gradio](../appendix/glossary.md#g-Gradio) | 部署与合规 | [函数](../appendix/glossary.md#g-函数)、[部署](../appendix/glossary.md#g-部署) |
    | [ONNX](../appendix/glossary.md#g-ONNX) | 部署与合规 | [部署](../appendix/glossary.md#g-部署) |
    | [量化](../appendix/glossary.md#g-量化) | 部署与合规 | [ONNX](../appendix/glossary.md#g-ONNX)、[模型参数](../appendix/glossary.md#g-模型参数) |

??? note "第 12 章 生态与安全（6 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [昇腾](../appendix/glossary.md#g-昇腾) | 框架与平台 | [昇思](../appendix/glossary.md#g-昇思)、[GPU](../appendix/glossary.md#g-GPU) |
    | [模型卡](../appendix/glossary.md#g-模型卡) | 部署与合规 | [部署](../appendix/glossary.md#g-部署)、[准确率](../appendix/glossary.md#g-准确率) |
    | [个人信息](../appendix/glossary.md#g-个人信息) | 部署与合规 | — |
    | [敏感个人信息](../appendix/glossary.md#g-敏感个人信息) | 部署与合规 | [个人信息](../appendix/glossary.md#g-个人信息) |
    | [脱敏](../appendix/glossary.md#g-脱敏) | 部署与合规 | [个人信息](../appendix/glossary.md#g-个人信息)、[正则表达式](../appendix/glossary.md#g-正则表达式) |
    | [正则表达式](../appendix/glossary.md#g-正则表达式) | Python 基础 | [字符串](../appendix/glossary.md#g-字符串) |

<!-- graph:end -->
