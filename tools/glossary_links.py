"""术语表锚点与正文术语链接工具。

用法：
    python tools/glossary_links.py anchors          # 给附录 D 每个术语加锚点（可重复运行）
    python tools/glossary_links.py link 06          # 把第 6 章正文中本章术语的首次加粗出现改成术语链接
    python tools/glossary_links.py concepts 06      # 按术语表重写第 6 章开头的“本章概念清单”

术语链接的效果：虚线下划线，鼠标悬停显示定义，点击跳到术语表。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY = ROOT / "docs/appendix/glossary.md"
CHAPTERS = ROOT / "docs/chapters"
SPAN = re.compile(r'<span id="(g-[^"]+)"></span>')


def anchor_id(term):
    return "g-" + re.sub(r"[\s/（）()、，,]+", "-", term).strip("-")


def parse():
    """返回 {章号: [(术语, 锚点, 定义), ...]}"""
    result, chap = {}, None
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        m = re.match(r"## 第 (\d+) 章", line)
        if m:
            chap = int(m.group(1))
            result[chap] = []
            continue
        if chap and line.startswith("| ") and not line.startswith("| 术语") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            term = SPAN.sub("", cells[0]).strip()
            result[chap].append((term, anchor_id(term), cells[2]))
    return result


def add_anchors():
    out = []
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and not line.startswith("| 术语") and not SPAN.search(line):
            cells = line.split("|")
            term = cells[1].strip()
            cells[1] = f' <span id="{anchor_id(term)}"></span>{term} '
            line = "|".join(cells)
        out.append(line)
    GLOSSARY.write_text("\n".join(out) + "\n", encoding="utf-8")


def link_plain(parts, term, aid, definition):
    """正文里没有加粗形式时，链接第一次以普通文字出现的位置（跳过标题、表格、代码、提示框标题、已有链接）。"""
    title = definition.replace('"', "“")
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        lines = part.split("\n")
        for j, line in enumerate(lines):
            st = line.lstrip()
            if not st or st.startswith(("#", "|", "!!!", "???", "<", ">", "$")):
                continue
            # 去掉行内代码、链接、加粗后再找，保证不落在它们里面
            masked = re.sub(r"`[^`]*`|\[[^\]]*\]\([^)]*\)(\{[^}]*\})?|\*\*[^*]*\*\*|<[^>]+>",
                            lambda m: "\0" * len(m.group(0)), line)
            pos = masked.find(term)
            if pos >= 0:
                link = f'[{term}](../../appendix/glossary.md#{aid} "{title}"){{ .term }}'
                lines[j] = line[:pos] + link + line[pos + len(term):]
                parts[i] = "\n".join(lines)
                return True
    return False


def link_bold(parts, name, aid, definition):
    """把第一个 **name** 改为术语链接；已经是链接的算作完成。"""
    target = f"**{name}**"
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        pos = part.find(target)
        if pos < 0:
            continue
        if part[max(0, pos - 1):pos] != "[":
            title = definition.replace('"', "“")
            link = f'[**{name}**](../../appendix/glossary.md#{aid} "{title}"){{ .term }}'
            parts[i] = part[:pos] + link + part[pos + len(target):]
        return True
    return False


def link_chapter(num):
    path = next(CHAPTERS.glob(f"{num:02d}-*/index.md"))
    text = path.read_text(encoding="utf-8")
    # 只处理“情景导入”到“本章小结”之间的正文，跳过代码块
    start = text.index("## 情景导入")
    end = text.index("## 本章小结")
    body = text[start:end]
    parts = re.split(r"(```.*?```)", body, flags=re.S)
    linked, missing = [], []
    for term, aid, definition in parse().get(num, []):
        done = any(f"#{aid} " in p for p in parts)  # 已经链接过
        if done:
            linked.append(term)
            continue
        # “词向量 / 词嵌入”这类术语，正文中出现其中任一个写法即可
        names = [term] + ([n.strip() for n in term.split(" / ")] if " / " in term else [])
        for name in names:
            done = link_bold(parts, name, aid, definition)
            if done:
                break
        if not done:
            for name in names:
                done = link_plain(parts, name, aid, definition)
                if done:
                    break
        (linked if done else missing).append(term)
    path.write_text(text[:start] + "".join(parts) + text[end:], encoding="utf-8")
    print(f"已链接 {len(linked)} 个术语；正文中没有找到加粗形式的：{'、'.join(missing) or '无'}")


def write_concepts(num):
    """按术语表中本章的术语，重写章首“本章概念清单”折叠框（框不存在时插在“先修要求”之后）。"""
    path = next(CHAPTERS.glob(f"{num:02d}-*/index.md"))
    text = path.read_text(encoding="utf-8")
    terms = parse().get(num, [])
    block = [f'??? abstract "本章概念清单（{len(terms)} 个，点击查看定义和例子）"']
    block += [f"    {i}. [{t}](../../appendix/glossary.md#{a})" for i, (t, a, _) in enumerate(terms, 1)]
    block = "\n".join(block) + "\n"
    m = re.search(r'\?\?\? abstract "本章概念清单.*?\n(?:    .*\n|\n(?=    ))*', text)
    if m:
        text = text[:m.start()] + block + text[m.end():]
    else:
        pos = text.index("## 情景导入")
        text = text[:pos] + block + "\n" + text[pos:]
    path.write_text(text, encoding="utf-8")
    print(f"第 {num} 章概念清单：{len(terms)} 个术语")


if __name__ == "__main__":
    if sys.argv[1] == "anchors":
        add_anchors()
    elif sys.argv[1] == "link":
        link_chapter(int(sys.argv[2]))
    elif sys.argv[1] == "concepts":
        write_concepts(int(sys.argv[2]))
