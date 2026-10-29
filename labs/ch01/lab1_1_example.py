"""实验 1-1 示例实验：计算准确率（模板示例，写第 1 章时替换）"""
import pandas as pd

df = pd.DataFrame({"y_true": [1, 0, 1], "y_pred": [1, 0, 0]})
acc = (df.y_true == df.y_pred).mean()
print(f"准确率：{acc:.1%}")
