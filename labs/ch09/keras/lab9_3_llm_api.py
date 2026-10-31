# 实验 9-3（通用版）用大模型 API 判断好评差评
# %% 第 1 步：准备 API Key（千万不要直接写进代码！）
import os
from getpass import getpass
api_key = os.environ.get("DEEPSEEK_API_KEY") or getpass("请粘贴你的 API Key：")   # 先找环境变量，没有就让你粘贴，屏幕上不显示

# %% 第 2 步：连上大模型，写好提示词
from openai import OpenAI                               # 先运行 !pip install openai
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")   # DeepSeek 的接口和 OpenAI 一样，换成其他兼容 OpenAI 的平台只要改 base_url 和 model
PROMPT = "你是外卖平台的评论审核员。判断下面这条评论是好评还是差评，只回答“好评”或“差评”两个字。\n评论：{}"

def ask(text):
    r = client.chat.completions.create(model="deepseek-chat",          # ← 改这里：换一个模型
                                       messages=[{"role": "user", "content": PROMPT.format(text)}],
                                       temperature=0)       # 0 表示回答尽量稳定，不要“发挥”
    return r.choices[0].message.content.strip()

tests = ["味道很好，送得也快", "等了两个小时，菜都凉了", "虽然有点贵，但是真的好吃",
         "一点都不好吃", "包装很好，可惜味道太咸了"]      # ← 改这里：写几句自己的评论
for t in tests:
    print(f"{t:16s} → {ask(t)}")

# %% 第 3 步：在 50 条验证评论上比一比准确率和用时
import time
import pandas as pd
val = pd.read_csv("../../data/reviews/reviews.csv")[1200:1250]   # 和实验 9-2 的验证集是同一批
start = time.time()
answers = [ask(t) for t in val["text"]]
right = sum(a == ("好评" if y == 1 else "差评") for a, y in zip(answers, val["label"]))
print(f"准确率 {right / len(val):.0%}，用时 {time.time() - start:.1f} 秒")
