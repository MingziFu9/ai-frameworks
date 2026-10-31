"""把实验脚本 labs/chNN/*/labN_M_xxx.py 转成同名的 .ipynb，方便在云端 Notebook 上打开。

脚本用 “# %% 标题” 分隔单元格，第一行注释作为笔记本标题。
用法：python tools/py2nb.py            （转换全部实验脚本）
      python tools/py2nb.py labs/ch05   （只转换某个目录）
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def convert(py):
    text = py.read_text(encoding="utf-8")
    lines = text.split("\n")
    title = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else py.stem
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n",
              "从上到下依次运行每个单元格（Shift + Enter）。标有“← 改这里”的地方是“改一改”练习要修改的位置。"]}]
    chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
    for k in range(1, len(chunks), 2):
        head, body = chunks[k].strip(), chunks[k + 1].strip("\n")
        if head:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"### {head}"]})
        src = body.split("\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [l + "\n" for l in src[:-1]] + [src[-1]]})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    py.with_suffix(".ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "labs"
n = 0
for py in sorted(base.rglob("lab*_*.py")):
    convert(py)
    n += 1
print(f"已生成 {n} 个笔记本")
