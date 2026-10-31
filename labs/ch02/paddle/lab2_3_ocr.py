# 实验 2-3（国产版）一行代码体验文字识别
# %% 第 1 步：请出现成的文字识别模型
from paddleocr import PaddleOCR               # 飞桨文字识别套件 PaddleOCR

ocr = PaddleOCR(use_doc_orientation_classify=False,   # 照片是正的，不用判断方向
                use_doc_unwarping=False,              # 不用拉平弯曲的纸张
                use_textline_orientation=False)       # 文字行不用判断方向

# %% 第 2 步：识别一张图片里的文字
PHOTO = "../../data/ocr_samples/flower_shop_notice.png"   # ← 改这里：换成你拍的招牌、小票照片
for res in ocr.predict(PHOTO):                 # 真正干活的就这一行
    for text, score in zip(res["rec_texts"], res["rec_scores"]):
        print(f"{text}    （把握 {score:.0%}）")
