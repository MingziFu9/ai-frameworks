# 实验 9-1（国产版）用现成模型做情感分析，用 jieba 提取关键词
# %% 第 1 步：一行代码请出情感分析模型
from paddlenlp import Taskflow                          # 先运行 !pip install paddlenlp
senta = Taskflow("sentiment_analysis")                  # 飞桨训练好的中文情感分析模型

# %% 第 2 步：判断几条评论
tests = ["味道很好，送得也快", "等了两个小时，菜都凉了", "虽然有点贵，但是真的好吃",
         "一点都不好吃", "包装很好，可惜味道太咸了"]      # ← 改这里：写几句自己的评论
for r in senta(tests):
    print(f"{r['text']:16s} → {r['label']}（把握 {r['score']:.0%}）")

# %% 第 3 步：分词和关键词（jieba，不用下载模型）
import jieba, jieba.analyse                             # 先运行 !pip install jieba
import pandas as pd
print(jieba.lcut("骑手态度很好，汤一点没洒，下次还点"))  # 把一句话切成词
df = pd.read_csv("../../data/reviews/reviews.csv")
bad = "，".join(df[df["label"] == 0]["text"])           # 把所有差评拼成一大段
print("差评关键词：", jieba.analyse.extract_tags(bad, topK=10))   # 找出最有代表性的 10 个词
