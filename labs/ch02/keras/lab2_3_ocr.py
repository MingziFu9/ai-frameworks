# 实验 2-3（通用版）一行代码体验文字识别
# %% 第 1 步：请出现成的文字识别模型
from rapidocr_onnxruntime import RapidOCR   # RapidOCR：把飞桨 PaddleOCR 的模型转成通用格式，装好就能用

ocr = RapidOCR()                            # 模型文件随安装包一起下载，不用另外下载

# %% 第 2 步：识别一张图片里的文字
PHOTO = "../../data/ocr_samples/flower_shop_notice.png"   # ← 改这里：换成你拍的招牌、小票照片
result, _ = ocr(PHOTO)                      # 真正干活的就这一行
for box, text, score in result:             # 每一行文字：位置、内容、把握
    print(f"{text}    （把握 {score:.0%}）")
