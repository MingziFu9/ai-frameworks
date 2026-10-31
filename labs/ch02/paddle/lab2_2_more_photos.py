# 实验 2-2（国产版）换图试试：用现成模型认一批照片
# %% 第 1 步：准备模型
from pathlib import Path

from paddlex import create_model

model = create_model(model_name="PP-LCNet_x1_0")

# %% 第 2 步：把文件夹里的照片一张张交给模型
FOLDER = Path("../../data/test_photos")        # ← 改这里：换成放你自己照片的文件夹
for photo in sorted(FOLDER.glob("*.jpg")):
    res = next(iter(model.predict(str(photo), batch_size=1, topk=1)))
    print(f"{photo.name:18s} → {res['label_names'][0]:16s} {res['scores'][0]:.1%}")
