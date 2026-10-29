# 实验代码

按章存放，`chNN/` 对应第 NN 章，文件名为 `labN_M_英文短名.py`（第 N 章实验 N-M），公共数据放在 `data/`，见 `data/README.md`。

安装实验环境（Python 3.11）：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

每个脚本都要能在 `labs/` 目录下直接运行（`python ch01/lab1_1_example.py`），运行结果和教材正文中的“运行结果”一致。需要 GPU、联网下载模型或调用大模型 API 的脚本，文件开头的注释写明运行条件和预计时间。
