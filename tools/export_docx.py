"""把网页教材导出为 Word 版（之后可在 Word / WPS 里另存为 PDF）。

用法（在仓库根目录运行，需要先安装 pandoc 3.x）：

    python tools/export_docx.py              # 全书，按 mkdocs.yml 的 nav 顺序，输出 build/教材.docx
    python tools/export_docx.py 01-sample    # 只导出一章，输出 build/01-sample.docx
    python tools/export_docx.py --pdf        # 同时用 LibreOffice 转出 PDF（需要安装 LibreOffice）

网页特有的元素按下面的办法转换：交互动画换成“请在网页版查看”的提示；Mermaid 图换成提示
（需要图片时，在网页上截图后手动插入）；提示框变成带粗体标题的引用块；在线答题的选项照常列出，
答案与解析保留在题目下方；术语和站内链接只保留文字。
"""
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "build"


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)


def nav_pages():
    """按 nav 顺序列出正文页（跳过动画、学习图谱等只适合网页的页面）"""
    cfg = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    pages = []

    def walk(items):
        for it in items:
            if isinstance(it, str):
                pages.append(it)
            elif isinstance(it, dict):
                for v in it.values():
                    walk(v) if isinstance(v, list) else pages.append(v)

    walk(cfg["nav"])
    return [p for p in pages if p == "course-description.md" or p.startswith(("chapters/", "appendix/"))]


def convert(md):
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)
    md = re.sub(r"```mermaid\n.*?```", "> 【图示】本图为交互图，请在网页版查看。", md, flags=re.S)
    md = re.sub(r'<iframe[^>]*title="([^"]*)"[^>]*></iframe>', r"> 【交互动画】\1：请在网页版查看。", md)
    md = re.sub(r"^.*\{ \.sim-caption \}\n", "", md, flags=re.M)
    md = re.sub(r'<sup class="cite">\[\[(\d+)\]\]\([^)]*\)</sup>', r"^[\1]^", md)
    md = re.sub(r'<span class="ref-note">(.*?)</span>', r"\1", md)
    md = re.sub(r"<span id=\"[^\"]*\"></span>", "", md)
    md = re.sub(r"^</?div[^>]*>\s*$", "", md, flags=re.M)
    # 术语链接、站内链接只留文字
    md = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]*\)", r"\1", md)
    md = re.sub(r"\{ *[.#:a-z][^}]*\}", "", md)
    # 提示框：!!! / ??? 类型 "标题" + 缩进正文 → 引用块
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        m = re.match(r'^(!!!|\?\?\?\+?) *[\w-]+(?: +"([^"]*)")?\s*$', lines[i])
        if not m:
            out.append(lines[i]); i += 1; continue
        out.append(f"> **{m.group(2) or ''}**")
        out.append(">")
        i += 1
        while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
            if lines[i].strip() == "" and i + 1 < len(lines) and not lines[i + 1].startswith("    "):
                break
            out.append(("> " + lines[i][4:]).rstrip())
            i += 1
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    args = [a for a in sys.argv[1:] if a != "--pdf"]
    if args:
        pages, name = [f"chapters/{args[0]}/index.md"], args[0]
    else:
        pages, name = nav_pages(), "教材"
    title = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)["site_name"]
    body = "\n\n".join(convert((DOCS / p).read_text(encoding="utf-8")) for p in pages)
    tmp = OUT / f"{name}.md"
    tmp.write_text(f"---\ntitle: {title}\n---\n\n{body}", encoding="utf-8")
    dst = OUT / f"{name}.docx"
    subprocess.run(["pandoc", str(tmp), "-f", "markdown+tex_math_single_backslash", "-o", str(dst),
                    "--toc", "--toc-depth=2", "--resource-path", str(DOCS)], check=True)
    print(f"已导出 {dst.relative_to(ROOT)}（{len(pages)} 页）")
    if "--pdf" in sys.argv:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(dst)], check=True)


if __name__ == "__main__":
    main()
