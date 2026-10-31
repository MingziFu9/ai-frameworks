"""编者生成文字识别（OCR）示例图片的脚本（学生不需要运行）。图片中的店名、电话、金额都是虚构的。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"   # 文泉驿正黑字体
OUT = Path(__file__).parent / "ocr_samples"
OUT.mkdir(exist_ok=True)


def draw(name, size, lines, bg="white"):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    for x, y, text, pt, color in lines:
        d.text((x, y), text, font=ImageFont.truetype(FONT, pt), fill=color)
    img.save(OUT / name)


draw("flower_shop_notice.png", (640, 360), [
    (40, 30, "花语小铺 · 母亲节特惠", 36, "#c2185b"),
    (40, 110, "康乃馨花束  每束 39 元", 28, "black"),
    (40, 160, "向日葵花束  每束 59 元", 28, "black"),
    (40, 210, "营业时间：9:00-21:00", 24, "#37474f"),
    (40, 260, "订花电话：020-8888 6666", 24, "#37474f")])
draw("receipt.png", (480, 520), [
    (150, 20, "花语小铺 销售小票", 26, "black"),
    (30, 80, "日期：2026-10-31  单号：A1027", 18, "black"),
    (30, 130, "商品            数量    金额", 20, "black"),
    (30, 170, "红玫瑰花束        1     99.00", 20, "black"),
    (30, 210, "满天星            2     30.00", 20, "black"),
    (30, 250, "贺卡              1      5.00", 20, "black"),
    (30, 300, "合计                   134.00", 22, "black"),
    (30, 350, "实收（微信）           134.00", 20, "black"),
    (30, 420, "谢谢惠顾，欢迎再来！", 20, "#555555")])
rows = [("品名", "数量", "单价", "金额"), ("红玫瑰", "50", "3.50", "175.00"), ("白百合", "20", "6.00", "120.00"),
        ("向日葵", "30", "4.00", "120.00"), ("康乃馨", "40", "2.50", "100.00"), ("满天星", "10", "8.00", "80.00")]
cells = [(220, 30, "花语小铺 进货单", 28, "black"), (40, 85, "供货商：某某花卉批发市场    日期：2026-10-30", 18, "#37474f")]
for r, row in enumerate(rows):
    for c, text in enumerate(row):
        cells.append((60 + c * 150, 140 + r * 45, text, 22, "black"))
cells.append((60, 140 + len(rows) * 45 + 15, "合计金额：595.00 元", 22, "#c2185b"))
draw("purchase_list.png", (680, 480), cells)
print("完成")
