---
title: 附录 B 通用版与国产版代码对照
description: 同一件事在 Keras 3、飞桨和昇思中的写法对照，以及两版代码最容易混淆的几处差别
---

# 附录 B 通用版与国产版代码对照

本书每个实验都有通用版（Keras 3）和国产版（飞桨）两套代码。这个附录把书里**真正用到的写法**按“要做的事”排成表，方便你：

- 只学一个版本时，快速查到某个操作怎么写；
- 想换到另一个版本时，一行一行对着改；
- 考试（开卷）时，查某个函数是做什么的。

最后一列是第 12 章体验过的昇思 MindSpore，只列实验 12-3 用到的写法，供感兴趣的同学参考。

!!! tip "怎么用这张表"
    表里只写了关键的一行。完整的、能运行的代码在“书中位置”一列给出的实验里。看不懂某一行时，先回到那个实验看注释，再看附录 A 的 Python 速查。

## B.1 安装与导入 { #b1 }

| 要做的事 | 通用版（Keras 3） | 国产版（飞桨） | 昇思 | 书中位置 |
|---|---|---|---|---|
| 安装 | `pip install tensorflow keras` | `pip install paddlepaddle` | `pip install mindspore "numpy<2"` | [实验 1-3](../chapters/01-intro/index.md#1-3) |
| 导入框架 | `import keras` | `import paddle` | `import mindspore as ms` | [实验 1-2](../chapters/01-intro/index.md#1-2) |
| 导入“积木” | `from keras import layers` | `from paddle import nn` | `from mindspore import nn` | [实验 4-1](../chapters/04-build-network/index.md#4-1) |
| 看版本 | `keras.__version__` | `paddle.__version__` | `ms.__version__` | 实验 1-2 |
| 有没有 GPU | `tf.config.list_physical_devices("GPU")` | `paddle.device.is_compiled_with_cuda()` | — | 实验 1-2 |
| 固定随机种子 | `keras.utils.set_random_seed(42)` | `paddle.seed(42)` | `ms.set_seed(42)` | [实验 5-1](../chapters/05-training/index.md#5-1) |

## B.2 调用现成模型 { #b2 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 取一个图像分类模型 | `mobilenet_v2.MobileNetV2(weights="imagenet")` | `create_model(model_name="PP-LCNet_x1_0")`（PaddleX） | [实验 2-1](../chapters/02-pretrained/index.md#2-1) |
| 读图并缩放 | `keras.utils.load_img(路径, target_size=(224, 224))` | PaddleX 自动完成 | 实验 2-1 |
| 预测 | `model.predict(x)` | `model.predict(路径, topk=5)` | 实验 2-1 |
| 把结果翻译成类别名 | `mobilenet_v2.decode_predictions(pred, top=5)` | 结果里直接有 `res["label_names"]`、`res["scores"]` | 实验 2-1 |
| 文字识别 | `RapidOCR()`（用的是 PaddleOCR 的模型） | `PaddleOCR()` | [实验 2-3](../chapters/02-pretrained/index.md#2-3)、[8-2](../chapters/08-image-apps/index.md#8-2) |
| 目标检测 | `YOLO("yolo11n.pt")`（ultralytics） | `create_model("PP-YOLOE_plus-S")`（PaddleX） | [实验 8-1](../chapters/08-image-apps/index.md#8-1) |
| 情感分析 | `pipeline("sentiment-analysis", model=...)`（transformers） | `Taskflow("sentiment_analysis")`（PaddleNLP） | [实验 9-1](../chapters/09-text-apps/index.md#9-1) |

两版都是“取模型 → 交给它数据 → 读结果”三步，区别只在函数名和结果的格式。

## B.3 读入数据 { #b3 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 按文件夹读图片（每个子文件夹一类） | `keras.utils.image_dataset_from_directory(DATA, ...)` | `DatasetFolder(DATA, transform=tf)` | [实验 3-2](../chapters/03-data/index.md#3-2) |
| 统一图片大小 | 参数 `image_size=(128, 128)` | `T.Resize((128, 128))` | 实验 3-2 |
| 变成 0～1 的数 | 网络里加 `layers.Rescaling(1 / 255)` | `T.ToTensor()`（顺便变成 0～1） | 实验 3-2、5-1 |
| 分出验证集 | 参数 `validation_split=0.2, subset="both"` | `paddle.io.random_split(full, [600, 150])` | 实验 3-2 |
| 分批 | 参数 `batch_size=32` | `DataLoader(ds, batch_size=32, shuffle=True)`，或在 `fit` 里写 `batch_size=32` | 实验 3-2、5-1 |
| 类别名 | `train_ds.class_names` | `full.classes` | 实验 3-2 |
| 取一批看看 | `for images, labels in train_ds.take(1):` | `images, labels = next(iter(loader))` | 实验 3-2 |

=== "通用版"

    ```python
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        DATA, validation_split=0.2, subset="both", seed=42,
        image_size=(128, 128), batch_size=32)
    ```

=== "国产版"

    ```python
    tf = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    full = DatasetFolder(DATA, transform=tf)
    train_ds, val_ds = paddle.io.random_split(full, [600, 150])
    ```

## B.4 数据增强 { #b4 }

| 效果 | 通用版（写成网络的层） | 国产版（写在 transform 里） | 书中位置 |
|---|---|---|---|
| 左右翻转 | `layers.RandomFlip("horizontal")` | `T.RandomHorizontalFlip()` | [实验 3-3](../chapters/03-data/index.md#3-3)、[6-2](../chapters/06-tuning/index.md#6-2) |
| 旋转 | `layers.RandomRotation(0.1)`（0.1 圈 = 36 度） | `T.RandomRotation(36)`（直接写度数） | 同上 |
| 放大 / 裁剪 | `layers.RandomZoom(0.2)` | `T.RandomResizedCrop(128, scale=(0.6, 1.0))` | 同上 |

注意旋转角度的写法不一样：Keras 写“转几分之一圈”，飞桨写“转多少度”。

## B.5 搭网络 { #b5 }

| 积木 | 通用版 | 国产版 | 昇思 | 书中位置 |
|---|---|---|---|---|
| 按顺序叠起来 | `keras.Sequential([...])` | `nn.Sequential(...)` | `nn.SequentialCell(...)` | [实验 4-1](../chapters/04-build-network/index.md#4-1) |
| 说明输入形状 | `keras.Input(shape=(128, 128, 3))` | 不用写 | 不用写 | 实验 4-2 |
| 全连接层 | `layers.Dense(64, activation="relu")` | `nn.Linear(输入数, 64), nn.ReLU()` | `nn.Dense(输入数, 64), nn.ReLU()` | 实验 4-1 |
| 卷积层 | `layers.Conv2D(16, 3, activation="relu")` | `nn.Conv2D(3, 16, 3), nn.ReLU()` | — | [实验 4-2](../chapters/04-build-network/index.md#4-2) |
| 池化层 | `layers.MaxPooling2D()` | `nn.MaxPool2D(2)` | — | 实验 4-2 |
| 展平 | `layers.Flatten()` | `nn.Flatten()` | — | 实验 4-2 |
| 随机丢弃 | `layers.Dropout(0.3)` | `nn.Dropout(0.3)` | — | 实验 6-2 |
| 输出概率 | `layers.Dense(5, activation="softmax")` | `nn.Linear(64, 5), nn.Softmax()` | 不加，预测时再用 `ms.ops.softmax` | 实验 4-2、[12-3](../chapters/12-ecosystem/index.md#12-3) |
| 词向量层 | `layers.Embedding(词表大小, 16)` | `nn.Embedding(词表大小, 16)` | — | [实验 9-2](../chapters/09-text-apps/index.md#9-2) |
| 看结构表和参数量 | `model.summary()` | `paddle.summary(net, (1, 3, 128, 128))` | — | 实验 4-1、[4-3](../chapters/04-build-network/index.md#4-3) |

=== "通用版"

    ```python
    model = keras.Sequential([
        keras.Input(shape=(128, 128, 3)),
        layers.Rescaling(1 / 255),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(5, activation="softmax"),
    ])
    ```

=== "国产版"

    ```python
    net = nn.Sequential(
        nn.Conv2D(3, 16, 3), nn.ReLU(), nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Linear(16 * 63 * 63, 5), nn.Softmax())
    ```

## B.6 训练、评估、保存 { #b6 }

| 要做的事 | 通用版 | 国产版 | 昇思 | 书中位置 |
|---|---|---|---|---|
| 套上“外壳” | 不用（`Sequential` 本身就是模型） | `model = paddle.Model(net)` | `Model(net, loss_fn=..., optimizer=..., metrics=...)` | [实验 5-1](../chapters/05-training/index.md#5-1) |
| 训练前三项设置 | `model.compile(loss=..., optimizer=..., metrics=["accuracy"])` | `model.prepare(optimizer, loss, metrics)` | 写在 `Model(...)` 里 | 实验 5-1 |
| 损失函数 | `"sparse_categorical_crossentropy"` | `nn.CrossEntropyLoss(use_softmax=False)` | `nn.CrossEntropyLoss()` | 实验 5-1 |
| 优化器 | `keras.optimizers.Adam(learning_rate=0.001)` | `paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())` | `nn.Adam(net.trainable_params(), learning_rate=0.01)` | 实验 5-1、[5-3](../chapters/05-training/index.md#5-3) |
| 准确率 | `metrics=["accuracy"]` | `paddle.metric.Accuracy()` | `metrics={"accuracy"}` | 实验 5-1 |
| 开始训练 | `model.fit(train_ds, validation_data=val_ds, epochs=10)` | `model.fit(train_ds, val_ds, epochs=10, batch_size=32)` | `model.train(100, train_ds)` | 实验 5-1 |
| 在验证集上评估 | `model.evaluate(val_ds)` | `model.evaluate(val_ds)` | `model.eval(val_ds)` | 实验 5-1 |
| 保存 | `model.save("flowers.keras")`（一个文件） | `model.save("flowers")`（得到 .pdparams 等文件） | — | 实验 5-1 |
| 读回来 | `keras.models.load_model("flowers.keras")` | 先搭好同样的网络，再 `model.load("flowers")` | — | [实验 11-1](../chapters/11-deploy/index.md#11-1) |
| 预测一张 | `model.predict(x)` | `net.eval()` 后 `net(x)` | `net(x)` | 实验 5-1 |

## B.7 调参 { #b7 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 早停 | `keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)` | `paddle.callbacks.EarlyStopping(monitor="loss", mode="min", patience=5, save_best_model=True)` | [实验 6-1](../chapters/06-tuning/index.md#6-1) |
| 保存最好的模型 | `keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True)` | 早停里的 `save_best_model=True`，存到 `save_dir/best_model` | 实验 6-1 |
| 把辅助工具交给训练 | `model.fit(..., callbacks=[...])` | `model.fit(..., callbacks=[...])` | 实验 6-1 |
| 训练记录 | `history.history["val_accuracy"]` | 书中自带的 `History` 回调（`history.py`） | 实验 6-1 |
| 可视化面板 | `keras.callbacks.TensorBoard("logs")`，命令 `tensorboard --logdir logs` | `paddle.callbacks.VisualDL("vdl_log")`，命令 `visualdl --logdir vdl_log` | [实验 6-3](../chapters/06-tuning/index.md#6-3) |

## B.8 迁移学习 { #b8 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 借一个预训练模型，去掉原输出层 | `keras.applications.MobileNetV2(include_top=False, weights="imagenet")` | `paddle.vision.models.mobilenet_v2(pretrained=True)`，再替换 `net.classifier` | [实验 7-1](../chapters/07-transfer/index.md#7-1) |
| 冻结 | `base.trainable = False` | `for p in net.parameters(): p.stop_gradient = True` | 实验 7-1 |
| 按模型要求换算像素 | `layers.Rescaling(1 / 127.5, offset=-1)` | `T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | 实验 7-1 |
| 新的输出层 | `layers.GlobalAveragePooling2D()`、`layers.Dense(5, activation="softmax")` | `nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 5), nn.Softmax())` | 实验 7-1 |
| 只训练新层 | 冻结后直接 `fit` | 优化器只交 `net.classifier.parameters()` | 实验 7-1 |
| 微调：解冻后几层 | `base.trainable = True` 后把前面的层再设为 `False`，学习率调小 | 对后几层的参数设 `stop_gradient = False`，学习率调小 | [实验 7-3](../chapters/07-transfer/index.md#7-3) |

## B.9 文本与声音 { #b9 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 把文字变成编号 | `layers.TextVectorization(split="character", output_sequence_length=40)`，再 `vec.adapt(texts)` | 用 `Counter` 数字频，自己写 `encode()` 函数 | [实验 9-2](../chapters/09-text-apps/index.md#9-2) |
| 中文分词、找关键词 | `jieba.lcut()`、`jieba.analyse.extract_tags()`（两版一样） | 同左 | 实验 9-1 |
| 调用大模型 API | `OpenAI(base_url="https://api.deepseek.com")` | `OpenAI(base_url="https://qianfan.baidubce.com/v2")` | [实验 9-3](../chapters/09-text-apps/index.md#9-3) |
| 读声音文件 | `scipy.io.wavfile.read()` | `paddle.audio.load()` | [实验 10-1](../chapters/10-speech-apps/index.md#10-1) |
| 声谱图 | `layers.MelSpectrogram()` | `paddle.audio.features.LogMelSpectrogram()` | 实验 10-1、[10-2](../chapters/10-speech-apps/index.md#10-2) |
| 语音识别与合成 | sherpa-onnx + Whisper tiny | sherpa-onnx + Paraformer | [实验 10-3](../chapters/10-speech-apps/index.md#10-3) |

## B.10 做成应用 { #b10 }

| 要做的事 | 通用版 | 国产版 | 书中位置 |
|---|---|---|---|
| 网页小工具 | `gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=3))`（两版一样） | 同左 | [实验 11-1](../chapters/11-deploy/index.md#11-1) |
| 导出 ONNX | 先 `predict` 一次，再 `model.export("flowers.onnx", format="onnx")` | `paddle.onnx.export(net, "flowers", input_spec=[...], opset_version=13)` | [实验 11-3](../chapters/11-deploy/index.md#11-3) |
| 用 ONNX 预测 | `ort.InferenceSession("flowers.onnx").run(None, {名字: x})`（两版一样） | 同左 | 实验 11-3 |
| 量化 | `quantize_static(...)`（onnxruntime，两版一样） | 同左 | 实验 11-3 |

## B.11 两版最容易弄混的五处 { #b11 }

| 差别 | 通用版 | 国产版 | 不注意会怎样 |
|---|---|---|---|
| 颜色通道放在哪 | 最后：`(批, 高, 宽, 3)` | 最前：`(批, 3, 高, 宽)` | 形状对不上，报 shape 错误；画图前飞桨要先 `transpose([1, 2, 0])` |
| 层要不要写输入数 | 只写输出数：`Dense(64)` | 输入、输出都写：`Linear(3136, 64)` | 飞桨里输入数写错会报形状错误 |
| 激活函数 | 写在层的参数里：`activation="relu"` | 单独一层：`nn.ReLU()` | 漏写激活，网络会退化成“直线” |
| Softmax 和损失 | 输出层写 `softmax`，损失用 `sparse_categorical_crossentropy` | 本书在末层加 `nn.Softmax()`，损失写 `use_softmax=False` | 两处都做 Softmax 或都不做，训练会变慢或结果不是概率 |
| 优化器 | 不用交参数 | 要写 `parameters=model.parameters()` | 飞桨里漏写会直接报错 |

!!! note "为什么飞桨这一列字更多"
    Keras 替你“猜”了很多东西（输入有多少个数、参数交给谁），所以写得短；飞桨要你说清楚，写得长一点，但每一步做了什么更明白。两种风格各有好处，学会一种以后，换另一种通常只要一两天。
