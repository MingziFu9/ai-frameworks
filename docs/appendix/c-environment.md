---
title: 附录 C 实验环境与数据说明
description: 云端 Notebook 和本机安装步骤、实测版本、数据与模型的来源和许可、待实测实验清单和课时安排建议
---

# 附录 C 实验环境与数据说明

## C.1 三种做实验的方式 { #c1 }

| 方式 | 适合 | 说明 |
|---|---|---|
| 云端 Notebook（推荐） | 全部实验 | 打开网页就能用，不用安装。通用版推荐魔搭社区，国产版推荐飞桨 AI Studio，CPU 环境免费 |
| 本机 CPU | 除“待实测”以外的实验 | 普通笔记本电脑（内存 8 GB 以上）即可，随书数据都很小，每次训练不超过 10 分钟 |
| 课堂演示 | 需要大显卡的内容 | 学校 A40 显卡服务器（每块 48 GB 显存），由老师在课堂上演示 |

本书所有标了运行结果的实验，都是编者在一台**没有显卡**的电脑上用 CPU 跑出来的，所以你用 CPU 也能完成。

## C.2 在云端 Notebook 上做实验 { #c2 }

注册和新建 Notebook 的步骤见 [1.4.2 节](../chapters/01-intro/index.md#142)。下面补充把随书代码和数据放上去的方法。

1. 在电脑上下载本书的实验文件夹 `labs`（老师会给出下载地址，或在本书 GitHub 仓库页面点 “Code → Download ZIP”）；
2. 在 Notebook 左侧的文件栏点“上传”，把压缩包 `labs.zip` 传上去；
3. 新建一个单元格，运行下面这行解压（前面的 `!` 表示“这是一条系统命令，不是 Python 代码”）：

    ```python
    !unzip -q labs.zip
    ```

4. 在文件栏里打开 `labs/ch01/keras`（或 `labs/ch01/paddle`），双击 `.ipynb` 文件就可以开始做实验。

!!! tip "云端平台缺少某个库怎么办"
    报错 `ModuleNotFoundError: No module named 'jieba'` 时，在单元格里运行 `!pip install jieba`，装完后再运行原来的单元格。每个实验需要的库都写在代码开头的注释里。

!!! warning "注意保存"
    云端环境长时间不用会自动关闭。训练好的模型文件（`.keras`、`.pdparams`）要及时下载到自己电脑，或者存在平台的“持久化”目录里（各平台名称不同，以平台说明为准）。

## C.3 在自己电脑上安装 { #c3 }

### C.3.1 安装 Python { #c31 }

1. 从 <https://www.python.org/downloads/> 下载 **Python 3.11**（本书在 3.11 上测试；3.10、3.12 一般也可以，但个别库可能不支持）；
2. Windows 安装时**一定勾选 “Add python.exe to PATH”**；
3. 打开“命令提示符”（Windows）或“终端”（macOS），输入 `python --version`，看到 `Python 3.11.x` 就成功了。

### C.3.2 建一个虚拟环境（推荐） { #c32 }

虚拟环境就是一个“独立的工具箱”，在里面装的库不会和电脑上别的软件打架。

```bash
python -m venv aif          # 新建一个叫 aif 的虚拟环境
aif\Scripts\activate        # Windows 进入环境
source aif/bin/activate     # macOS 进入环境
```

进入后命令行前面会出现 `(aif)`。以后每次做实验，先运行这一行进入环境。

### C.3.3 安装本书用到的库 { #c33 }

在 `labs` 文件夹里运行：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 分成“公共部分”“通用版”“国产版”三段，只学一个版本的同学可以用 `#` 把另一段注释掉。`-i` 后面是清华大学的国内镜像，下载更快。

### C.3.4 编者实测的版本 { #c34 }

下面是编者 2026 年 10 月实际运行通过的版本（Python 3.11，CPU）。框架更新很快，你装到的版本号可能更新，一般不影响本书的实验；如果遇到报错，可以按下表指定版本安装，例如 `pip install keras==3.15.1`。

| 库 | 版本 | 用于 |
|---|---|---|
| tensorflow-cpu / keras | 2.21.0 / 3.15.1 | 通用版全书 |
| paddlepaddle | 3.3.1 | 国产版全书 |
| numpy / pandas / matplotlib | 2.3.5 / 3.1.0 / 3.11.2 | 全书 |
| tensorboard / visualdl | 2.21.0 / 2.5.3 | 第 6 章 |
| ultralytics | 8.4.194 | 第 8 章 |
| rapidocr-onnxruntime / paddleocr / paddlex | 1.4.4 / 3.7.0 / 3.7.2 | 第 2、7、8 章 |
| jieba | 0.42.1 | 第 9、12 章 |
| openai | 3.42.0 | 第 9 章（调用大模型 API） |
| sherpa-onnx / soundfile | 2.0.1 / 0.14.0 | 第 10 章 |
| gradio | 6.31.0 | 第 11 章 |
| onnx / onnxruntime / tf2onnx / paddle2onnx | 1.17.0 / 1.31.1 / 1.17.0 / 2.1.0 | 第 11 章 |
| torch / torchvision | 2.15.0 / 0.29.2 | 只在实验 5-2（通用版）里对照训练循环 |
| mindspore | 2.11.0（需要 numpy 小于 2，建议单独建环境） | 第 12 章 |

### C.3.5 安装时的常见问题 { #c35 }

| 现象 | 原因 | 解决办法 |
|---|---|---|
| `'pip' 不是内部或外部命令` | 安装 Python 时没勾选 PATH | 重新运行安装程序，选 “Modify”，勾选 “Add Python to environment variables”；或改用 `python -m pip install ...` |
| 下载很慢、`Read timed out` | 从国外服务器下载 | 加上 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `No matching distribution found for tensorflow` | Python 版本太新或太旧，或电脑是 32 位 | 换成 64 位的 Python 3.11 |
| 装完 MindSpore 后其他实验报错 | MindSpore 要求 numpy 小于 2，把 numpy 降级了 | MindSpore 单独建一个虚拟环境 |
| 第一次运行很久不动 | 正在下载预训练模型 | 耐心等待；网络不通时见 C.5 节 |
| `jupyter` 命令找不到 | 没有进入虚拟环境 | 先运行 C.3.2 节的进入命令 |

## C.4 数据集 { #c4 }

随书数据都在 `labs/data` 文件夹里，总共约 22 MB，不需要另外下载。

| 数据 | 内容和规模 | 来源 | 许可 | 用于 |
|---|---|---|---|---|
| `flowers_small` | 5 类花卉照片，每类 150 张，共 750 张 | TensorFlow 官方数据集 flower_photos（原 3670 张），编者缩小到长边 200 像素 | CC BY 2.0（使用时要署名，作者名单见文件夹内 LICENSE.txt） | 第 3–7、11 章 |
| `test_photos` | 每类 2 张，共 10 张，不参与训练 | 同上 | 同上 | 第 2、5、7、11 章 |
| `detect_photos` | 4 张原尺寸照片 | 同上 | 同上 | 第 8 章 |
| `ocr_samples` | 告示、采购单、小票 3 张文字图片 | 编者用程序生成 | 内容虚构，可自由用于教学 | 第 2、8 章 |
| `reviews/reviews.csv` | 1600 条外卖评论，好评、差评各半 | 编者用程序生成（固定随机种子 42） | 内容虚构，可自由用于教学 | 第 9 章 |
| `speech_commands_small` | 8 个英文指令词，每个 100 条 1 秒录音 | Google Speech Commands 的子集 mini_speech_commands | CC BY 4.0 | 第 10 章 |
| `service_records` | 5 条客服记录 | 编者编写 | 内容虚构；身份证号用了不存在的地区码，不对应任何真人 | 第 12 章 |
| 鸢尾花 | 150 朵花、每朵 4 个测量值 | scikit-learn 自带 | 公开数据，教学通用 | 第 12 章（第 4 章借用它“4 个数、3 类”的形状） |

## C.5 预训练模型的来源和许可 { #c5 }

这些模型不放在随书文件里，第一次运行时自动下载。许可决定了模型能不能用于商业用途，做自己的项目前一定要查清楚（第 12 章）。

| 模型 | 版本 | 从哪里下载 | 许可 | 用于 |
|---|---|---|---|---|
| MobileNetV2（ImageNet） | 通用版 | Keras 官方（storage.googleapis.com） | Apache 2.0 | 第 2、7、11 章 |
| PP-LCNet、MobileNetV2（ImageNet） | 国产版 | 飞桨服务器（bcebos.com） | Apache 2.0 | 第 2、7 章 |
| YOLO11n | 通用版 | ultralytics 的 GitHub 发布页 | AGPL-3.0（商用需另购许可） | 第 8 章 |
| PP-YOLOE+ | 国产版 | 飞桨服务器 | Apache 2.0 | 第 8 章 |
| RapidOCR（PaddleOCR 模型的 ONNX 版） | 通用版 | 随 pip 包一起安装 | Apache 2.0 | 第 2、8 章 |
| PaddleOCR | 国产版 | 飞桨服务器 | Apache 2.0 | 第 2、8 章 |
| 中文情感分析（uer/roberta-base-finetuned-jd-binary-chinese） | 通用版 | Hugging Face | 以模型页面说明为准 | 第 9 章 |
| Taskflow 情感分析、命名实体识别 | 国产版 | 飞桨服务器 | Apache 2.0 | 第 9、12 章 |
| Whisper tiny | 通用版 | sherpa-onnx 的 GitHub 发布页 | MIT | 第 10 章 |
| Paraformer-zh-small | 国产版 | sherpa-onnx 的 GitHub 发布页 | 以模型页面说明为准 | 第 10 章 |
| matcha-icefall-zh-baker + vocos | 两版共用 | sherpa-onnx 的 GitHub 发布页 | 训练数据（标贝）只允许非商业使用 | 第 10 章 |

!!! tip "下载不了模型怎么办"
    国内访问 Hugging Face 常常不稳定，可以在运行前加一行 `import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` 使用国内镜像。飞桨的模型在 AI Studio 上下载最快。GitHub 下载慢时，可以请老师提前下载好放到课程网盘。

## C.6 标为“待实测”的内容 { #c6 }

编者的编写环境连接不上飞桨模型服务器、Hugging Face 和大模型平台，下面这些内容的代码检查过写法，但没有给出真实运行结果。书里每处都放了记录表，请你运行后自己填写；**建议任课教师开课前在 AI Studio 上跑一遍，把结果补进课件**。

| 位置 | 版本 | 需要什么 | 说明 |
|---|---|---|---|
| [实验 2-1、2-2](../chapters/02-pretrained/index.md#2-1) | 国产版 | 下载 PP-LCNet | 通用版有实测结果可对照 |
| [实验 2-3](../chapters/02-pretrained/index.md#2-3)、[8.3.2](../chapters/08-image-apps/index.md#832) | 国产版 | 下载 PaddleOCR 模型 | 通用版的 RapidOCR 用的是同一套模型，结果应接近 |
| [实验 7-1、7-2、7-3](../chapters/07-transfer/index.md#7-1) | 国产版 | 下载 ImageNet 预训练参数 | 代码已在不加载预训练参数时跑通；通用版验证准确率 82.7% 可作参考 |
| [7.6 节](../chapters/07-transfer/index.md#76) | 国产版 | PaddleX 低代码训练 | 只做介绍 |
| [8.2.1](../chapters/08-image-apps/index.md#821)、实验 8-1、8-3 | 国产版 | 下载 PP-YOLOE+ | 通用版 YOLO11n 有实测结果 |
| [9.2.1](../chapters/09-text-apps/index.md#921)、实验 9-1 | 两版 | Hugging Face / 飞桨模型 | jieba 关键词部分是实测的 |
| [9.4.3](../chapters/09-text-apps/index.md#943)、实验 9-3 | 两版 | 大模型 API 密钥（DeepSeek、千帆） | 调用流程已用模拟服务器检查；调用会产生少量费用 |
| [11.4 节](../chapters/11-deploy/index.md#114)、实验 11-2 | 两版 | 魔搭创空间 / AI Studio / Hugging Face 账号 | 发布步骤按平台公开文档整理 |
| [11.5.2](../chapters/11-deploy/index.md#1152)、实验 11-3 | 国产版 | 实验 7-1 的预训练结果 | 大小和速度是实测的，只有准确率待测 |
| [12.4.3](../chapters/12-ecosystem/index.md#1243)、实验 12-1 | 国产版 | 下载 PaddleNLP 命名实体识别模型 | 通用版 jieba 结果是实测的 |
| 实验 1-2 第二档 | 两版 | GPU 环境 | 比较 CPU 和 GPU 的用时 |

## C.7 课时安排建议 { #c7 }

全课 72 学时：线上理论 46、线下理论 14、线上实验 12。每章 1 学时实验，从该章 3 个实验里选做（★ 为建议必做），其余作为课后练习。

| 章 | 内容 | 理论 | 其中线下 | 实验 | 建议实验 |
|---|---|---|---|---|---|
| 1 | 认识人工智能开发框架 | 4 | 1 | 1 | 1-1、1-2 |
| 2 | 第一次用框架：调用现成模型 | 4 | 1 | 1 | 2-1、2-2 |
| 3 | 数据在框架里的样子 | 5 | 1 | 1 | 3-1、3-2 |
| 4 | 搭积木：构建神经网络 | 5 | 1 | 1 | 4-1、4-2 |
| 5 | 训练模型：fit 背后发生了什么 | 6 | 1 | 1 | 5-1、5-2 |
| 6 | 看懂训练曲线，学会调参 | 5 | 1 | 1 | 6-1、6-2 |
| 7 | 迁移学习 | 5 | 1 | 1 | 7-1、7-2 |
| 8 | 图像应用：检测与文字识别 | 5 | 1 | 1 | 8-1、8-2 |
| 9 | 文本应用 | 5 | 1 | 1 | 9-1、9-2 |
| 10 | 语音应用 | 5 | 1 | 1 | 10-1、10-2 |
| 11 | 把模型做成应用 | 6 | 2 | 1 | 11-1、11-2 |
| 12 | 国产 AI 生态、数据安全与综合项目 | 5 | 2 | 1 | 12-1、12-2 |
| | **合计** | **60** | **14** | **12** | |

线下课建议这样用：第 1 章带学生注册平台、跑通第一个 Notebook；第 5、7 章现场排错；第 11 章各组试用彼此的网页；第 12 章综合项目开题和最后的展示答辩。需要大显卡的演示（比如在完整的 3670 张花卉数据上训练、微调更大的模型）放在线下课上，用学校的 A40 服务器完成。
