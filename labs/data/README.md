# 实验数据说明

本文件夹里的数据都很小，普通笔记本电脑的 CPU 就能跑完全部实验。公开数据集仅用于教学，请遵守原项目的许可；编者自编的材料内容为虚构，其中的人名、店名、电话、证件号码都不对应真实的人和机构。

| 文件夹或文件 | 内容 | 规模 | 来源与许可 | 使用章节 |
|---|---|---|---|---|
| `flowers_small/` | 5 类花卉照片（雏菊、蒲公英、玫瑰、向日葵、郁金香），长边缩到 200 像素 | 750 张（每类 150），8.7 MB | TensorFlow 官方数据集 flower_photos，CC BY 2.0，作者见文件夹内 LICENSE.txt | 第 3–7、11 章 |
| `test_photos/` | 每类 2 张“新照片”，不参与训练，用来试预测 | 10 张 | 同上 | 第 1、3、5、7、11 章 |
| `detect_photos/` | 含多个物体的原图（未缩放），用于检测和分割 | 4 张 | 同上，作者见 LICENSE.txt | 第 8 章 |
| `ocr_samples/` | 花店告示、采购单、小票三张文字图片 | 3 张 | 教材编写（`make_ocr_samples.py` 生成），内容为虚构 | 第 8 章 |
| `reviews/reviews.csv` | 外卖评论，列为 text、label（1 好评 / 0 差评） | 1600 条 | 教材编写（`make_reviews.py` 生成，种子 42），内容为虚构 | 第 9 章 |
| `speech_commands_small/` | 8 个英文指令词（down go left no right stop up yes）的 1 秒录音，编者转成了 8 kHz | 800 条（每类 100），13 MB | Google Speech Commands 子集 mini_speech_commands，CC BY 4.0，见 LICENSE.txt | 第 10 章 |
| `service_records/records.txt` | 5 条客服记录，含姓名、手机号、身份证号等，用来练习脱敏 | 5 条 | 教材编写，内容为虚构；证件号用不存在的地区码 990101 | 第 12 章 |

编者用来生成这些文件的脚本也放在这里（`make_datasets.py`、`make_ocr_samples.py`、`make_reviews.py`），学生不需要运行。

**需要联网下载的模型**（不放在仓库里，第一次运行时自动下载）：Keras 预训练权重（第 7 章）、YOLO11n 与 RapidOCR 模型（第 8 章）、sherpa-onnx 语音模型（第 10 章）、飞桨 PaddleX / PaddleNLP 模型（第 7–9 章国产版）。来源和许可见[附录 C](../../docs/appendix/c-environment.md)。
