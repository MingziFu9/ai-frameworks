---
title: 附录 A Python 速查
description: 本书实验中常用的 Python 基础语法、文件读写、正则表达式、NumPy 与 pandas 操作速查
---

# 附录 A Python 速查

本附录汇总了本书实验中最常用的 Python 知识点，供同学们在做实验时随时查阅。它不是完整的 Python 教程，如果你还没有学过 Python，建议先学习一门 Python 程序设计入门课程。

## A.1 字符串

处理文本，最常打交道的就是字符串。

```python
s = "  自然语言处理 NLP  "
s.strip()                  # 去掉两端空白 -> "自然语言处理 NLP"
s.lower(), s.upper()       # 英文转小写 / 大写
len("自然语言")             # 长度（字符数）-> 4
"语言" in s                # 是否包含 -> True
s.replace("NLP", "技术")    # 替换
"a,b,c".split(",")         # 按分隔符切分 -> ['a', 'b', 'c']
"/".join(["我", "爱", "NLP"])  # 用分隔符连接 -> "我/爱/NLP"
"自然语言处理"[0:2]          # 切片：取前两个字 -> "自然"
"自然语言处理"[-2:]          # 取最后两个字 -> "处理"
s.startswith("  自"), s.endswith("  ")
name, score = "小王", 92.5
f"{name}的成绩是{score:.1f}分"   # f-string 格式化 -> "小王的成绩是92.5分"
f"{name:<6}|{score:>8.2f}"      # 左对齐占 6 位，右对齐占 8 位保留 2 位小数
```

## A.2 列表、字典、集合

```python
words = ["我", "爱", "自然", "语言", "处理"]
words.append("！")              # 末尾添加
words[1], words[-1]            # 按位置取元素
[w for w in words if len(w) > 1]   # 列表推导式：筛选 -> ['自然', '语言', '处理']
sorted(words, key=len, reverse=True)   # 按长度从长到短排序

freq = {"好吃": 3, "难吃": 1}    # 字典：键 -> 值
freq["好吃"]                    # 取值 -> 3
freq.get("一般", 0)             # 取不到时返回默认值 0
freq["一般"] = 2                # 添加或修改
for word, count in freq.items():   # 遍历键和值
    print(word, count)

a, b = {"我", "爱", "你"}, {"你", "爱", "他"}   # 集合：自动去重
a & b, a | b, a - b             # 交集、并集、差集

from collections import Counter
Counter(["好", "好", "差"]).most_common(1)   # 计数 -> [('好', 2)]
```

## A.3 函数、条件与循环

```python
def clean(text, lower=True):
    """函数的说明文字写在这里"""
    text = text.strip()
    if lower:
        text = text.lower()
    return text

for i, w in enumerate(["a", "b"]):   # 同时得到序号和元素
    print(i, w)
for x, y in zip([1, 2], ["一", "二"]):   # 同时遍历两个列表
    print(x, y)

try:                                 # 捕获异常
    n = int("abc")
except ValueError as e:
    print("转换失败：", e)
```

## A.4 文件读写

**中文文件一定要指定 `encoding="utf-8"`**，否则在 Windows 上很容易出现乱码或 `UnicodeDecodeError`。

```python
text = open("article.txt", encoding="utf-8").read()          # 读取全部内容
lines = open("words.txt", encoding="utf-8").read().splitlines()   # 按行读成列表

with open("out.txt", "w", encoding="utf-8") as f:            # 写文件（with 会自动关闭文件）
    f.write("第一行\n")

import json
data = json.load(open("data.json", encoding="utf-8"))        # 读 JSON
json.dump(data, open("out.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)                      # 写 JSON，中文不转义
json.loads('{"a": 1}'), json.dumps({"名字": "小王"}, ensure_ascii=False)   # 字符串与对象互转

import os
os.path.exists("a.txt")          # 文件是否存在
os.environ.get("DEEPSEEK_API_KEY")   # 读取环境变量
```

## A.5 正则表达式

正则表达式用于按“模式”查找和替换文本，第 2、7、12 章大量使用。

| 写法 | 含义 | 示例 |
|---|---|---|
| `\d` | 一个数字 | `\d{11}` 匹配 11 位数字 |
| `\w` | 字母、数字、下划线或汉字 | |
| `\s` | 空白字符（空格、换行、制表符） | |
| `.` | 除换行外的任意字符 | |
| `[abc]` / `[^abc]` | a、b、c 中的一个 / 不是 a、b、c | `[一-龥]` 匹配一个汉字 |
| `*` `+` `?` | 0 次以上 / 1 次以上 / 0 或 1 次 | |
| `{m,n}` | 重复 m 到 n 次 | `[一-龥]{2,4}` |
| `^` `$` | 开头 / 结尾 | |
| `(…)` / `(?:…)` | 分组 / 不保存的分组 | |
| `a|b` | a 或 b | `(有限公司|大学)` |
| `(?<!\d)` `(?!\d)` | 前面不是数字 / 后面不是数字 | 防止从更长的数字中截取 |

```python
import re
re.findall(r"1[3-9]\d{9}", "电话13800001234")   # 找出所有匹配 -> ['13800001234']
re.sub(r"\s+", " ", "a   b\n c")              # 替换 -> "a b c"
re.split(r"[。！？]", "你好。再见！")            # 按模式切分
m = re.search(r"(\d+)元", "共计35元")
m.group(), m.group(1), m.start()              # '35元'、'35'、匹配开始的位置
```

!!! tip "小技巧"
    正则表达式前面加 `r`（如 `r"\d+"`），表示“原始字符串”，反斜杠不会被 Python 转义。写不出来的正则，可以描述需求让大模型帮忙写，但一定要用多个正例和反例测试。

## A.6 NumPy

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])
a.shape                      # 形状 -> (2, 2)
a.T, a @ a                   # 转置、矩阵乘法
a.sum(), a.mean(axis=0)      # 求和、按列求平均
np.zeros((2, 3)), np.random.randn(2, 3)   # 全零矩阵、正态分布随机矩阵
np.argmax([0.1, 0.7, 0.2])   # 最大值的位置 -> 1
v1, v2 = np.array([1, 0]), np.array([1, 1])
v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))   # 余弦相似度
```

## A.7 pandas

```python
import pandas as pd
df = pd.read_csv("waimai_10k.csv")       # 读 CSV（Excel 用 pd.read_excel）
df.head(), df.shape, df.columns          # 前 5 行、行列数、列名
df["label"].value_counts()               # 统计每个值出现的次数
df[df["label"] == 1]                     # 筛选行
df["review"].str.len()                   # 每条评论的长度
df["review"].str.contains("好吃")         # 是否包含某个词
df["words"] = df["review"].apply(lambda t: t[:10])   # 对每个元素执行函数
df.drop_duplicates("review").dropna()    # 去重、去掉空值
df.sample(5, random_state=7)             # 随机抽 5 行
df.to_csv("out.csv", index=False, encoding="utf-8-sig")   # 保存（utf-8-sig 让 Excel 打开不乱码）
```

## A.8 常见错误速查

| 错误信息 | 常见原因 |
|---|---|
| `ModuleNotFoundError: No module named 'xxx'` | 库没有安装，或者安装到了另一个 Python 环境中 |
| `FileNotFoundError` | 文件路径不对；数据文件没有放在程序所在的目录 |
| `UnicodeDecodeError` | 读取文件时没有指定 `encoding="utf-8"`，或文件本身是 GBK 编码 |
| `IndentationError` | 缩进不一致，混用了空格和 Tab |
| `KeyError` | 字典中没有这个键；DataFrame 中没有这个列名 |
| `IndexError: list index out of range` | 下标超出了列表的长度 |
| `TypeError: ... NoneType` | 某个函数返回了 `None`，又被当作字符串或列表使用 |
| `ValueError: shapes ... not aligned` | 矩阵乘法时两个矩阵的形状不匹配 |
