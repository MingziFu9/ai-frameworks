"""生成实验 9-2 用的外卖评论数据（编者用程序随机组合生成，无版权问题，可自由使用）。
运行：python make_reviews.py   → 生成 reviews/reviews.csv（text,label；1=好评，0=差评）
"""
import random, csv
random.seed(42)
POS = {
    "味道": ["味道很好", "好吃", "味道不错", "特别香", "口味正宗", "很入味", "酸甜适中", "辣得过瘾", "米饭香软", "肉很嫩"],
    "速度": ["送得很快", "半小时就到了", "比预计早到", "配送超快", "骑手很准时"],
    "包装": ["包装很严实", "包装干净", "汤一点没洒", "餐盒很精致", "保温做得好"],
    "分量": ["分量足", "量很大", "两个人都吃饱了", "肉给得多", "性价比高"],
    "服务": ["骑手态度好", "老板很热情", "还送了饮料", "备注都照做了", "客服回复很快"],
}
NEG = {
    "味道": ["太咸了", "很难吃", "味道一般般", "有股怪味", "油太多", "肉不新鲜", "一点都不好吃", "米饭夹生", "菜都凉了", "不好吃"],
    "速度": ["送了两个小时", "等了好久", "超时了", "配送太慢", "骑手迟到很久"],
    "包装": ["汤洒了一半", "包装破了", "餐盒漏油", "袋子都湿了", "包装很简陋"],
    "分量": ["分量太少", "量少得可怜", "没吃饱", "肉少得可怜", "价格太贵"],
    "服务": ["骑手态度差", "少送了一份", "备注没看", "送错了", "客服不理人"],
}
END_POS = ["", "", "下次还点", "推荐", "五星好评", "会回购的", "满意"]
END_NEG = ["", "", "再也不点了", "不推荐", "差评", "失望", "退款了"]
BUT = ["不过", "但是", "就是"]

def clause(table, used):
    asp = random.choice([a for a in table if a not in used]); used.add(asp)
    return random.choice(table[asp])

def review(label):
    main, other = (POS, NEG) if label == 1 else (NEG, POS)
    used = set()
    parts = [clause(main, used) for _ in range(random.randint(1, 3))]
    if random.random() < 0.25:                       # 四分之一的评论夹一句相反的话，让任务不那么简单
        twist = clause(other, used)
        if random.random() < 0.5:
            parts = ["虽然" + twist, "但是" + parts[0]] + parts[1:]   # 先抑后扬 / 先扬后抑
        else:
            parts = parts + [random.choice(BUT) + twist]
    end = random.choice(END_POS if label == 1 else END_NEG)
    text = "，".join(parts) + ("，" + end if end else "") + random.choice(["。", "！", "", "~"])
    if random.random() < 0.03:                       # 3% 的标签故意标错，模拟真实数据里的噪声
        label = 1 - label
    return text, label

rows = [review(i % 2) for i in range(1600)]
random.shuffle(rows)
with open("reviews/reviews.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text", "label"]); w.writerows(rows)
print("生成", len(rows), "条评论，例如：", rows[:3])
