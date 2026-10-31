# 人工智能开发框架技术

**面向成人教育人工智能技术应用专业（专科）的《课程名称》教材**

<!-- 书名、课程信息框和“目录与实验”表由 python tools/course.py sync 根据 course.yml 生成；全书概览表由 python tools/book_stats.py 生成 -->

<!-- course:start -->
!!! abstract "课程信息"
    4 学分 · 72 学时（理论 60 学时 + 实验 12 学时） · 第 4 学期 · 12 章 · 36 个实验 · 期末开卷
<!-- course:end -->

## 这本书讲什么

两三段话：这门课研究什么、学生身边哪里用得到、全书从哪里讲到哪里。

## 本书的特点

- **通俗易懂**：用生活中的例子和“算一算”小计算讲清原理，公式只保留最必要的部分；
- **每章动手实验**：★ 为基础实验，☆ 为拓展实验，教师可按课时选做，其余作为课后练习；
- **结果真实**：书中的运行结果都是编者实际运行得到的；编写环境无法运行的部分标为“待实测”，并提供记录表；
- **在线练习**：每章课后的单选、多选和判断题可以在网页上直接作答，并查看解析；
- **两个版本**：每段代码都有通用版（Keras）和国产版（飞桨）两种写法，点代码块上方的标签切换，选一次全书统一。

## 全书概览

<!-- stats:start -->

| 章 | 正文字数 | 实验 | 课后练习<br>（可在线作答） | 术语 | 参考文献 | 交互动画 |
|---|---|---|---|---|---|---|
| [第 1 章](chapters/01-intro/index.md) | 0.6 万 | 3 | 20（16） | 18 | 8 | 1 |
| [第 2 章](chapters/02-pretrained/index.md) | 0.6 万 | 3 | 20（16） | 12 | 8 | 1 |
| [第 3 章](chapters/03-data/index.md) | 0.6 万 | 3 | 20（16） | 11 | 7 | 2 |
| [第 4 章](chapters/04-build-network/index.md) | 0.5 万 | 3 | 20（16） | 10 | 7 | 1 |
| [第 5 章](chapters/05-training/index.md) | 0.6 万 | 3 | 20（16） | 7 | 6 | 2 |
| [第 6 章](chapters/06-tuning/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 7 章](chapters/07-transfer/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 8 章](chapters/08-image-apps/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 9 章](chapters/09-text-apps/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 10 章](chapters/10-speech-apps/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 11 章](chapters/11-deploy/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| [第 12 章](chapters/12-ecosystem/index.md) | 0.1 万 | 3 | 5（3） | 0 | 1 | 0 |
| **合计** | **3.4 万** | **36** | **135（101）** | **58** | **43** | **7** |

正文字数只统计汉字，不含代码和课后练习；单选、多选和判断题可以在网页上直接作答。配套学习资源：[交互动画](sims/index.md) 7 个 · [学习图谱](learning-graph/index.md) 2 个概念 · [常见问题](faq/index.md) 2 问 · [术语表](appendix/glossary.md) 58 条。

<!-- stats:end -->

## 目录与实验

<!-- toc:start -->
| 章节 | 实验 |
|---|---|
| [第 1 章 认识人工智能开发框架](chapters/01-intro/index.md) | 实验 1-1 ★ 打开云端 Notebook，运行第一段代码<br>实验 1-2 ★ 环境自检<br>实验 1-3 ☆ 在自己的电脑上安装框架 |
| [第 2 章 第一次用框架：调用现成模型](chapters/02-pretrained/index.md) | 实验 2-1 ★ 让模型认出一张花的照片<br>实验 2-2 ★ 换图试试：认一批照片<br>实验 2-3 ☆ 一行代码体验文字识别 |
| [第 3 章 数据在框架里的样子](chapters/03-data/index.md) | 实验 3-1 ★ 把一张图片变成数字<br>实验 3-2 ★ 读入花卉数据集<br>实验 3-3 ☆ 数据增强效果对比 |
| [第 4 章 搭积木：构建神经网络](chapters/04-build-network/index.md) | 实验 4-1 ★ 搭一个三层网络，看懂结构表<br>实验 4-2 ★ 搭一个识别花卉的卷积网络<br>实验 4-3 ☆ 加减层，看参数量怎么变 |
| [第 5 章 训练模型：fit 背后发生了什么](chapters/05-training/index.md) | 实验 5-1 ★ 训练第一个花卉识别模型<br>实验 5-2 ★ 打开盒子：训练循环逐行对应五个步骤<br>实验 5-3 ☆ 改学习率和轮次，看训练结果怎么变 |
| [第 6 章 看懂训练曲线，学会调参](chapters/06-tuning/index.md) | 实验 6-1 ★ 画训练曲线，用早停自动保存最好的模型<br>实验 6-2 ★ 四组对比实验：数据增强和 Dropout 管不管用<br>实验 6-3 ☆ 用 TensorBoard / VisualDL 比较两次训练 |
| [第 7 章 迁移学习：站在别人的肩膀上](chapters/07-transfer/index.md) | 实验 7-1 ★ 迁移学习：借 MobileNetV2 认花<br>实验 7-2 ★ 错误分析：混淆矩阵和认错的照片<br>实验 7-3 ☆ 微调：解冻后面几层 |
| [第 8 章 图像应用：检测与文字识别](chapters/08-image-apps/index.md) | 实验 8-1 ★ 目标检测：框出照片里的东西<br>实验 8-2 ★ 文字识别：把进货单变成表格<br>实验 8-3 ☆ 做一个“数一数”小应用 |
| [第 9 章 文本应用：让模型读懂文字](chapters/09-text-apps/index.md) | 实验 9-1 ★ 一行代码做情感分析<br>实验 9-2 ★ 训练一个外卖评论分类器<br>实验 9-3 ☆ 用大模型 API 做同一件事 |
| [第 10 章 语音应用：让模型听懂声音](chapters/10-speech-apps/index.md) | 实验 10-1 ★ 实验名称<br>实验 10-2 ★ 实验名称<br>实验 10-3 ★ 实验名称 |
| [第 11 章 把模型做成应用](chapters/11-deploy/index.md) | 实验 11-1 ★ 实验名称<br>实验 11-2 ★ 实验名称<br>实验 11-3 ★ 实验名称 |
| [第 12 章 国产 AI 生态、数据安全与综合项目](chapters/12-ecosystem/index.md) | 实验 12-1 ★ 实验名称<br>实验 12-2 ★ 实验名称<br>实验 12-3 ★ 实验名称 |

附录：[A 零基础 Python 入门](appendix/a-python.md) · [B 通用版与国产版代码对照](appendix/b-libraries.md) · [C 实验环境与数据说明](appendix/c-environment.md) · [D 术语表](appendix/glossary.md) · [E 实验报告模板与综合项目指南](appendix/e-project.md) · [F 期末模拟卷](appendix/f-mock-exam.md)
<!-- toc:end -->

## 配套资源

- [交互动画](sims/index.md)、[学习图谱](learning-graph/index.md)、[常见问题](faq/index.md)
- 实验代码与数据：仓库中的 `labs/` 目录
- 课件、教案和 Word/PDF 版教材：见仓库的 Releases 页面
