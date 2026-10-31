# 实验 12-1（通用版）给一份客服记录做脱敏
# %% 第 1 步：读入客服记录（内容为虚构）
text = open("../../data/service_records/records.txt", encoding="utf-8").read()
print(text)

# %% 第 2 步：用正则表达式找出手机号、身份证号等，遮住中间部分
import re
def mask(s, head, tail):                             # 只留前 head 位和后 tail 位，中间换成 *
    return s[:head] + "*" * (len(s) - head - tail) + s[len(s) - tail:]

RULES = [                                            # （名称，正则表达式，留前几位，留后几位）
    ("身份证号", r"\d{17}[\dXx]", 6, 4),              # 17 位数字 + 1 位数字或 X
    ("银行卡号", r"\d{4} \d{4} \d{4} \d{4}", 0, 4),   # 4 组 4 位数字
    ("手机号", r"1[3-9]\d-?\d{4}-?\d{4}", 3, 4),      # 1 开头，第二位 3～9，共 11 位（可以有短横线）
    ("门牌号", r"\d+号[\d栋单元室]*", 0, 0),          # 数字 + “号” + 楼栋房号，全部遮住
]                                                    # ← 改这里：可以加上自己的规则
for name, pattern, head, tail in RULES:
    found = re.findall(pattern, text)
    print(name, "找到", len(found), "个：", found)
    text = re.sub(pattern, lambda m: mask(m.group(), head, tail), text)
text = re.sub(r"[\w.]+@", "***@", text)              # 邮箱：@ 前面的部分全部遮住

# %% 第 3 步：遮住人名
import jieba.posseg as pseg                         # 先运行 !pip install jieba
KEEP = {"顾客", "银行卡", "红玫瑰", "满天星"}       # 不是人名却被当成人名的词，第一次运行后发现的   ← 改这里
def hide_names(line):
    for word, flag in pseg.cut(line):                # 切词，并标出每个词的词性
        if flag == "nr" and len(word) >= 2 and word not in KEEP:   # nr 表示人名
            line = line.replace(word, word[0] + "*" * (len(word) - 1))   # 只留姓
    return line

safe = "\n".join(hide_names(line) for line in text.splitlines())
print(safe)
open("records_safe.txt", "w", encoding="utf-8").write(safe)   # 存成脱敏后的新文件
