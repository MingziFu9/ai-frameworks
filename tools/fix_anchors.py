"""给含英文字母的编号标题加显式锚点，避免自动生成的锚点带英文单词、回看链接失效。

用法：python tools/fix_anchors.py 01     （处理第 1 章；不带参数处理全部章节）
规则：## 1.4 云端 Notebook → { #14 }；### 1.4.2 … → { #142 }；### 实验 1-2 ★ … → { #1-2 }
已有 { #… } 的标题不动。
"""
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs/chapters"


def fix(path):
    out, n = [], 0
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^(#{2,3}) (.*)$", line)
        if m and "{ #" not in line and re.search(r"[A-Za-z]", m.group(2)):
            title = m.group(2)
            num = re.match(r"(\d+(?:\.\d+)+) ", title)
            lab = re.match(r"(?:实验|体验) (\d+-\d+) ", title)
            if num:
                line = f"{line} {{ #{num.group(1).replace('.', '')} }}"
                n += 1
            elif lab:
                line = f"{line} {{ #{lab.group(1)} }}"
                n += 1
        out.append(line)
    path.write_text("\n".join(out), encoding="utf-8")
    return n


targets = sys.argv[1:] or [p.name[:2] for p in sorted(DOCS.iterdir())]
for t in targets:
    for p in DOCS.glob(f"{t}-*/index.md"):
        print(p.parent.name, "加锚点", fix(p), "处")
