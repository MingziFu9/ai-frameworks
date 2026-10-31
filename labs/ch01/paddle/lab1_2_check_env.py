# 实验 1-2（国产版）环境自检
# %% 第 1 步：看看都装了什么
import platform
import time

import paddle

print("Python 版本：", platform.python_version())
print("飞桨版本：", paddle.__version__)

# %% 第 2 步：有没有 GPU
print("当前设备：", paddle.device.get_device())
print("能用 GPU 吗：", paddle.device.is_compiled_with_cuda(), "（False 表示只能用 CPU，本课的必做实验用 CPU 也能完成）")

# %% 第 3 步：飞桨自带的安装检查
paddle.utils.run_check()

# %% 第 4 步：让框架做一道大题，计时
a = paddle.rand([2000, 2000])                       # 2000×2000 的随机数表格
start = time.time()
c = paddle.matmul(a, a)                             # 矩阵乘法：约 80 亿次乘法和加法
print(f"2000×2000 矩阵乘法用时：{time.time() - start:.3f} 秒")
