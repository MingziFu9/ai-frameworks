---
title: 附录 A 零基础 Python 入门
description: 本书实验用到的 16 个 Python 知识点，每个配一个可以直接运行的小例子，并注明在书中哪里用到
---

# 附录 A 零基础 Python 入门

本书实验用到的 Python 知识一共只有十几个。这里把它们按“从简单到复杂”的顺序排好，每个知识点配一个可以直接复制到 Notebook 里运行的小例子，并注明书里在哪里用到。第 1～4 章的“Python 小课堂”讲过的，这里再集中复习一遍。

!!! tip "怎样用这个附录"
    不用从头背到尾。做实验时遇到看不懂的写法，来这里查；把例子复制到 Notebook 里运行一下，改一改再运行，比看十遍都管用。

## A.1 print：让电脑说话 { #a1 }

```python
print("你好，人工智能")              # 打印一句话
print("1 + 2 =", 1 + 2)             # 逗号隔开，可以打印好几样东西
```

书中用到：几乎每个实验（[1.5.1](../chapters/01-intro/index.md#151)）。

## A.2 变量：贴了名字的盒子 { #a2 }

```python
name = "郁金香"                      # 把文字放进名叫 name 的盒子
count = 150                          # 整数
rate = 0.001                         # 小数
print(name, count, rate)
count = count + 10                   # 盒子里的东西可以换
print(count)                         # 160
```

变量名用英文字母、数字和下划线，不能以数字开头，区分大小写。书中用到：`DATA = "../../data/flowers_small"`、`learning_rate=0.001`（[1.5.2](../chapters/01-intro/index.md#152)）。

## A.3 import：把工具箱拿过来 { #a3 }

```python
import numpy as np                   # 导入 numpy，起个简称 np
from pathlib import Path             # 从 pathlib 里只拿 Path 这一样
import keras                         # 通用版；国产版是 import paddle
```

`import 库名 as 简称` 很常见：`np`、`pd`、`plt` 都是大家约定俗成的简称。报 `No module named 'xxx'` 说明这个库没装，运行 `!pip install xxx`。书中用到：每个实验的第一行（[1.5.3](../chapters/01-intro/index.md#153)）。

## A.4 函数调用和参数 { #a4 }

```python
print(round(3.14159, 2))             # round(数, 保留几位) → 3.14
print(max(3, 8, 5))                  # 8
print(sorted([3, 1, 2], reverse=True))   # reverse=True 是“关键字参数”，从大到小 → [3, 2, 1]
```

函数名后面跟括号，括号里是参数。`名字=值` 形式的叫关键字参数，可以不按顺序写，书里大量使用，比如 `epochs=10`、`batch_size=32`。书中用到：[2.4.1](../chapters/02-pretrained/index.md#241)。

## A.5 字符串：用引号括起来的文字 { #a5 }

```python
s = "味道很好，下次还点"
print(len(s))                        # 有几个字 → 9
print(s[0], s[-1])                   # 第一个字、最后一个字 → 味 点
print(s[:4])                         # 前 4 个字 → 味道很好
print("好" in s)                     # 包含“好”吗 → True
print(s.replace("很好", "一般"))      # 替换
path = "roses/123.jpg"
print(path.split("/"))               # 按 / 切开 → ['roses', '123.jpg']
```

书中用到：第 9 章评论处理、第 12 章脱敏（[2.4.2](../chapters/02-pretrained/index.md#242)）。

## A.6 f-string：把变量填进文字里 { #a6 }

```python
name, prob = "郁金香", 0.8134
print(f"认成了{name}，把握 {prob:.1%}")     # → 认成了郁金香，把握 81.3%
print(f"损失 {1.23456:.4f}")                # 保留 4 位小数 → 损失 1.2346
print(f"{'玫瑰':6s}|")                      # 占 6 个字的位置，用来对齐
```

字符串前面加 `f`，花括号里放变量；冒号后面写格式：`.1%` 百分数保留 1 位，`.4f` 保留 4 位小数。书中用到：几乎所有打印结果的地方。

## A.7 列表和 for 循环 { #a7 }

```python
flowers = ["雏菊", "蒲公英", "玫瑰"]
flowers.append("向日葵")              # 末尾加一个
print(flowers[0], len(flowers))      # 第一个，一共几个 → 雏菊 4
for f in flowers:                    # 挨个取出来
    print("我认识", f)               # 循环里的代码要缩进 4 个空格
for i in range(3):                   # range(3) 是 0, 1, 2
    print("第", i + 1, "轮")
```

**缩进**是 Python 的规矩：属于循环（以及后面的 if、def）的代码，都要往里缩进 4 个空格。书中用到：[2.4.3](../chapters/02-pretrained/index.md#243)，第 5 章训练循环等。

## A.8 if：做判断 { #a8 }

```python
prob = 0.45
if prob > 0.5:
    print("好评")
elif prob > 0.3:                     # 否则如果
    print("拿不准")
else:                                # 否则
    print("差评")
```

比较用 `>`、`<`、`>=`、`==`（等于，两个等号）、`!=`（不等于）。书中用到：第 9 章判断好评差评、第 11 章“不确定”提示。

## A.9 字典：查表 { #a9 }

```python
price = {"红玫瑰": 3.5, "白百合": 6.0}  # 键 → 值
print(price["红玫瑰"])                # 查表 → 3.5
price["满天星"] = 8.0                 # 加一项
print(price.get("向日葵", 0))         # 查不到就给默认值 0
for k, v in price.items():           # 同时取出键和值
    print(k, v)
```

书中用到：第 9 章词表 `word2id.get(ch, 1)`、第 11 章预测函数返回 `{花名: 概率}`、第 8 章按行整理文字。

## A.10 def：自己写一个函数 { #a10 }

```python
def mask(s, head, tail):             # 定义函数：名字、参数
    """只留前 head 位和后 tail 位，中间换成 *"""
    return s[:head] + "*" * (len(s) - head - tail) + s[len(s) - tail:]

print(mask("13900001234", 3, 4))     # → 139****1234

def count_down(n=3):                 # 参数可以有默认值
    for i in range(n, 0, -1):
        print(i)
count_down()                         # 不写参数就用默认值 3
```

把常用的几行代码打包、起个名字，以后一行就能调用。`return` 把结果交回去。书中用到：第 8 章 `count_objects`、第 9 章 `encode`、第 11 章 `predict`、第 12 章 `mask`。

## A.11 列表推导式：一行生成一个列表 { #a11 }

```python
nums = [1, 2, 3, 4]
print([n * n for n in nums])                 # → [1, 4, 9, 16]
words = ["骑手", "很", "态度", "好"]
print([w for w in words if len(w) == 2])     # 只要两个字的 → ['骑手', '态度']
```

读法：“对 nums 里的每个 n，算 n * n，收集起来”。等价于 for 循环加 append，只是更短。书中用到：第 8 章 `names = [r.names[int(c)] for c in r.boxes.cls]`、第 9 章 `encode`。

## A.12 zip 和 enumerate：同时取好几样 { #a12 }

```python
names = ["雏菊", "玫瑰"]
probs = [0.9, 0.1]
for n, p in zip(names, probs):       # 两个列表一一配对
    print(n, p)
for i, n in enumerate(names):        # 同时得到序号和元素
    print(i, n)                      # 0 雏菊 / 1 玫瑰
```

书中用到：第 8 章检测结果 `zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)`、第 9 章打印测试评论。

## A.13 读写文件和路径 { #a13 }

```python
open("note.txt", "w", encoding="utf-8").write("第一行\n第二行")   # 写文件
text = open("note.txt", encoding="utf-8").read()                # 读文件
print(text.splitlines())                                        # 按行切开 → ['第一行', '第二行']

from pathlib import Path
for p in sorted(Path(".").glob("*.txt")):                       # 找出当前文件夹所有 .txt 文件
    print(p.name)
```

中文文件一定写 `encoding="utf-8"`。路径里的 `.` 是当前文件夹，`..` 是上一级，所以 `../../data` 是“往上两级再进 data”。书中用到：第 2、7 章认一批照片，第 12 章读写客服记录。

## A.14 NumPy：一大堆数字 { #a14 }

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)                       # 形状：2 行 3 列 → (2, 3)
print(a / 255)                       # 每个数都除以 255
print(a.max(), a.argmax())           # 最大值，以及它在第几个 → 6 5
b = a[None]                          # 前面加一维 → 形状 (1, 2, 3)，“只有一张的一批”
print(b.shape)
p = np.array([0.1, 0.7, 0.2])
print(p.argmax())                    # 概率最大的是第几个 → 1
```

书中用到：第 3 章图片变数字、各章预测时的 `[None]` 和 `argmax()`。

## A.15 pandas：用代码操作的 Excel { #a15 }

```python
import pandas as pd
df = pd.DataFrame({"品名": ["红玫瑰", "白百合"], "金额": [175.0, 120.0]})
print(df)                            # 一张表
print(df["金额"].sum())               # 一列求和 → 295.0
print(df[df["金额"] > 150])           # 筛选行
df.to_csv("表格.csv", index=False, encoding="utf-8-sig")   # 存成 Excel 能打开的文件
df2 = pd.read_csv("表格.csv")        # 读回来
print(df2.head())                    # 看前几行
```

书中用到：第 8 章进货单变表格、第 9 章读评论数据。

## A.16 lambda：一次性的小函数 { #a16 }

```python
square = lambda x: x * x             # 等价于 def square(x): return x * x
print(square(5))                     # → 25
import re
print(re.sub(r"\d", lambda m: "*", "电话 139"))   # 每个数字都换成 * → 电话 ***
```

`lambda 参数: 结果` 适合只用一次、只有一行的小函数，常作为参数传给别的函数。书中用到：第 11 章测速 `timeit(lambda: ...)`、第 12 章脱敏 `re.sub`。

## A.17 看懂报错信息 { #a17 }

报错时，**从最后一行看起**：冒号前面是错误类型，后面是说明。

| 错误类型 | 一般是什么问题 | 例子 |
|---|---|---|
| `NameError` | 用了没定义的名字（拼错了，或者前面的单元格没运行） | `name 'model' is not defined` |
| `ModuleNotFoundError` | 库没装 | `No module named 'gradio'` |
| `FileNotFoundError` | 文件路径不对 | `No such file or directory: 'xx.jpg'` |
| `SyntaxError` | 写法错误（括号、引号没配对，中文标点） | `invalid syntax` |
| `IndentationError` | 缩进不对 | `expected an indented block` |
| `KeyError` | 字典或表格里没有这个键 | `KeyError: 'text'` |
| `IndexError` | 序号超出范围 | `list index out of range` |
| `ValueError` | 值不对（形状、类型不合要求） | `could not convert string to float` |

最常见的低级错误：**用了中文标点**（中文逗号“，”、中文括号“（）”、中文引号“”），Python 只认英文标点。各章的“常见报错对照”表收集了和框架有关的报错。
