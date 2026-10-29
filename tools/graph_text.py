"""检查学习图谱数据，并生成 docs/learning-graph/index.md 里的统计和“文字版”。

用法：python tools/graph_text.py   （可重复运行；内容写在两个 graph 注释之间）

检查项：先修关系里的概念都存在、没有循环依赖、每个概念的锚点在术语表里都有。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DATA = DOCS / "learning-graph/data.js"
PAGE = DOCS / "learning-graph/index.md"
GLOSSARY = DOCS / "appendix/glossary.md"
START, END = "<!-- graph:start -->", "<!-- graph:end -->"


def load():
    s = DATA.read_text(encoding="utf-8")
    s = re.sub(r"^\s*//.*$", "", s, flags=re.M)
    vals = {}
    for name in ("CATS", "CHAPTERS", "CHDIRS", "NODES", "EDGES"):
        m = re.search(rf"var {name}\s*=\s*(.*?);\s*(?=var |\Z)", s, re.S)
        vals[name] = json.loads(m.group(1))
    return vals


def check(v):
    errors = []
    ids = {n["id"] for n in v["NODES"]}
    anchors = set(re.findall(r'<span id="(g-[^"]+)"></span>', GLOSSARY.read_text(encoding="utf-8")))
    for n in v["NODES"]:
        if n["a"] not in anchors:
            errors.append(f"术语表里没有锚点 {n['a']}（概念“{n['id']}”）")
        if not 0 <= n["cat"] < len(v["CATS"]) or not 1 <= n["ch"] < len(v["CHAPTERS"]):
            errors.append(f"概念“{n['id']}”的类别或章号超出范围")
    pre = {i: [] for i in ids}
    for a, b in v["EDGES"]:
        for x in (a, b):
            if x not in ids:
                errors.append(f"先修关系里的“{x}”不在概念列表中")
        if a in ids and b in ids:
            pre[b].append(a)
    state = {}

    def visit(x, path):
        if state.get(x) == 1:
            errors.append("循环依赖：" + " → ".join(path + [x]))
            return
        if state.get(x) == 2:
            return
        state[x] = 1
        for p in pre[x]:
            visit(p, path + [x])
        state[x] = 2

    for i in ids:
        visit(i, [])
    return errors, pre


def main():
    v = load()
    errors, pre = check(v)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    link = lambda n: f"[{n['id']}](../appendix/glossary.md#{n['a']})"
    by_id = {n["id"]: n for n in v["NODES"]}
    cats = "、".join(v["CATS"])
    lines = [START, "", f"图谱共有 **{len(v['NODES'])} 个概念、{len(v['EDGES'])} 条先修关系**，概念按章排列，颜色表示 {len(v['CATS'])} 个类别（{cats}）。概念的定义都来自[附录 D 术语表](../appendix/glossary.md)，先修关系由编者根据各章内容整理。", "", "## 文字版", "", "下面按章列出每个概念的直接先修概念，便于打印或用读屏软件阅读。", ""]
    for ch in range(1, len(v["CHAPTERS"])):
        nodes = [n for n in v["NODES"] if n["ch"] == ch]
        if not nodes:
            continue
        lines += [f'??? note "第 {ch} 章 {v["CHAPTERS"][ch]}（{len(nodes)} 个概念）"', "", "    | 概念 | 类别 | 直接先修 |", "    |---|---|---|"]
        for n in nodes:
            pres = "、".join(link(by_id[p]) for p in pre[n["id"]]) or "—"
            lines.append(f"    | {link(n)} | {v['CATS'][n['cat']]} | {pres} |")
        lines.append("")
    lines.append(END)
    s = PAGE.read_text(encoding="utf-8")
    s = s[:s.index(START)] + "\n".join(lines) + s[s.index(END) + len(END):]
    PAGE.write_text(s, encoding="utf-8")
    print(f"学习图谱：{len(v['NODES'])} 个概念，{len(v['EDGES'])} 条先修关系，检查通过")


if __name__ == "__main__":
    main()
