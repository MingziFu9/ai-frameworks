# 实验 8-2（国产版）拍照识别进货单，存成表格
# %% 第 1 步：识别图片里的每一块文字
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False)          # 关掉用不到的功能，速度更快
PHOTO = "../../data/ocr_samples/purchase_list.png"     # ← 改这里：换成你的单据照片
res = ocr.predict(PHOTO)[0]
for b, text, score in zip(res["rec_boxes"], res["rec_texts"], res["rec_scores"]):
    print(f"左上角 {[int(b[0]), int(b[1])]}  {text}  （把握 {score:.0%}）")

# %% 第 2 步：按“上下位置”把文字块排成一行一行
rows = {}
for b, text in zip(res["rec_boxes"], res["rec_texts"]):
    y = round(b[1] / 20)                        # 左上角的纵坐标，每 20 像素算一行
    rows.setdefault(y, []).append((b[0], text)) # 记下横坐标和文字
table = [[t for x, t in sorted(cells)] for y, cells in sorted(rows.items())]   # 每行按从左到右排好
for line in table:
    print(line)

# %% 第 3 步：挑出 4 列的行，存成表格文件
import pandas as pd
data = [line for line in table if len(line) == 4]       # 只要“品名 数量 单价 金额”这样的行
df = pd.DataFrame(data[1:], columns=data[0])            # 第一行当表头
df["金额"] = df["金额"].astype(float)                    # 文字变成数字，才能求和
print(df)
print("合计：", df["金额"].sum())
df.to_csv("purchase_list.csv", index=False, encoding="utf-8-sig")   # 用 Excel 打开不会乱码
