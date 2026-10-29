"""每章一份 PowerPoint 课件（.pptx）。

用法（在仓库根目录运行）：
    python tools/make_slides.py draft 3         # 从第 3 章正文生成课件底稿 slides/ch03.md（已有则不覆盖，加 --force 覆盖）
    python tools/make_slides.py build 3         # 把 slides/ch03.md 排版成 build/课件/第03章 章名.pptx
    python tools/make_slides.py build all       # 全部章节

流程：先 draft 生成底稿，再把底稿精简成“一页一个要点”（底稿是从正文机械摘出来的，不能直接用），
最后 build。课件里的 Mermaid 图和动画截图来自 npm run figures（tools/render_figures.mjs），
没渲染时这些位置显示文字提示。底稿的写法见 slides/README.md。
"""
import math
import re
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from common import (BUILD, ROOT, admonition, chapter_by_num, chapter_heading, chapter_path, chapter_text, course,
                    front_matter, headings, lab_list, list_items, mermaid_blocks, mermaid_png, parse_quizzes,
                    plain, qr_png, safe_name, section, sim_png, sims_in, site_url)

SLIDES = ROOT / "slides"
FONT, MONO = "Microsoft YaHei", "Consolas"
ACCENT, INK, MUTED, SOFT, LINE = "3F51B5", "263238", "78909C", "E8EAF6", "CFD8DC"
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
TOP = Inches(1.35)            # 内容区上边
BOTTOM = H - Inches(0.55)     # 内容区下边（页脚之上）


# ======================================================================
# 一、从正文生成底稿
# ======================================================================

def first_sentence(text, limit=45):
    text = plain(text).replace("**", "")
    s = re.split(r"(?<=[。！？；])", text)[0].strip()
    return s if len(s) <= limit else s[:limit - 1] + "……"


def blocks_of(body):
    """把一段正文拆成段落、列表、表格、代码、提示框等块：[(类型, 文本), ...]"""
    out, lines, i = [], body.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith("#"):
            i += 1
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(("mermaid" if line.strip() == "```mermaid" else "code", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif line.strip() in ("\\[", "$$"):
            end = "\\]" if line.strip() == "\\[" else "$$"
            j = i + 1
            while j < len(lines) and lines[j].strip() != end:
                j += 1
            out.append(("math", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif re.match(r"^(!!!|\?\?\?)", line):
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
                j += 1
            out.append(("admonition", "\n".join(lines[i:j])))
            i = j
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("```", "!!!", "???", "#")):
                j += 1
            chunk = "\n".join(lines[i:j])
            if chunk.startswith("|"):
                out.append(("table", chunk))
            elif re.match(r"^(?:[-*]|\d+\.) ", chunk):
                out.append(("list", chunk))
            elif chunk.startswith("<iframe"):
                out.append(("sim", chunk))
            elif chunk.startswith("<") or chunk.startswith("{"):
                pass
            elif "{ .sim-caption }" not in chunk:
                out.append(("para", " ".join(l.strip() for l in chunk.split("\n"))))
            i = j
    return out


def draft_slide(title, body, mermaid_index):
    """正文的一个小节 → 底稿里的一页或几页"""
    bullets, notes, extras, tables = [], [], [], []
    for kind, text in blocks_of(body):
        if kind == "para" and re.match(r"^!\[[^\]]*\]\([^)]+\)$", text.strip()):
            extras.append(text.strip())
        elif kind == "para":
            bullets.append(first_sentence(text))
            notes.append(plain(text).replace("**", ""))
        elif kind == "list":
            bullets += [first_sentence(x, 50) for x in list_items(text)][:6]
        elif kind == "table" and text.count("\n") <= 9:
            tables.append(text)
        elif kind == "sim":
            extras += [f"!sim {name}" for name, _ in sims_in(text)]
        elif kind == "mermaid":
            extras.append(f"!mermaid {mermaid_index[text]}")
        elif kind == "admonition" and re.match(r'^!!! cert', text):
            bullets.append("证书考点：" + first_sentence(text.split("\n", 1)[1] if "\n" in text else "", 40))
    pages = []
    note = "\n::: notes\n" + "\n".join(notes)[:800] + "\n:::\n" if notes else ""
    if bullets or extras:
        pages.append(f"## {title}\n\n" + "\n".join(f"- {b}" for b in bullets[:6]) +
                     ("\n\n" + "\n".join(extras[:1]) if extras else "") + "\n" + note)
    for x in extras[1:]:
        pages.append(f"## {title}\n\n{x}\n")
    for t in tables:
        pages.append(f"## {title}\n\n{t}\n")
    return pages


def draft(num, force=False):
    c = course()
    ch = chapter_by_num(num, c)
    text = chapter_text(ch)
    path = SLIDES / f"ch{num:02d}.md"
    if path.exists() and not force:
        print(f"{path.relative_to(ROOT)} 已存在，没有覆盖（要重新生成请加 --force）")
        return
    mermaid_index = {code.rstrip("\n"): i for i, code in enumerate(mermaid_blocks(text), 1)}
    out = [f"---\nchapter: {num}\n---\n",
           "<!-- 课件底稿，由 python tools/make_slides.py draft 从正文机械摘出，请精简后再 build：\n"
           "     一页一个要点，每页 3–6 条，每条不超过 30 字；讲解的话写进 ::: notes（演讲者备注）。写法见 slides/README.md -->\n"]
    goals = [plain(x) for x in list_items(section(text, "学习目标"))]
    out.append("## 学习目标\n\n" + "\n".join(f"- {g}" for g in goals) + "\n")
    lab_title = c["lab_section"]
    for title, body in headings(text, 2):
        if title in ("学习目标", "参考文献", "课后练习"):
            continue
        if title == "情景导入":
            out += draft_slide("情景导入", body, mermaid_index)
        elif title == lab_title:
            out.append(f"# {lab_title}\n")
            overview = admonition(body, "abstract")
            if overview:
                table = "\n".join(l for l in overview[0][1].split("\n") if l.startswith("|"))
                out.append(f"## {c['lab_word']}总览\n\n{table}\n")
            for no, name, goals in lab_list(body, c["lab_word"]):
                out.append(f"## {c['lab_word']} {no} {name}\n\n" + "\n".join(f"- {g}" for g in goals) + "\n")
        elif title == "本章小结":
            items = [first_sentence(x, 60) for x in list_items(body)]
            out.append("## 本章小结\n\n" + "\n".join(f"- {x}" for x in items) + "\n")
            for code in mermaid_blocks(body):
                out.append(f"## 知识导图\n\n!mermaid {mermaid_index[code.rstrip(chr(10))]}\n")
            nxt = re.search(r"\*\*下一章预告\*\*\n\n(.+)", body)
            if nxt:
                out.append("## 下一章预告\n\n- " + first_sentence(nxt.group(1), 80) + "\n")
        else:
            subs = headings(body, 3)
            if subs:
                out.append(f"# {title}\n")
                intro = body.split("\n### ", 1)[0]
                if intro.strip():
                    out += draft_slide(title, intro, mermaid_index)
                for sub, sbody in subs:
                    out += draft_slide(sub, sbody, mermaid_index)
            else:
                out += draft_slide(title, body, mermaid_index)
    quizzes = [i for i, q in enumerate(parse_quizzes(text), 1) if q["kind"] in ("单选", "判断")][:3]
    out += [f"## 课堂练习\n\n!quiz {i}\n" for i in quizzes]
    SLIDES.mkdir(exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"已生成底稿 {path.relative_to(ROOT)}（{sum(1 for x in out if x.startswith('## '))} 页），请精简后运行 build")


# ======================================================================
# 二、解析底稿
# ======================================================================

def parse_deck(src):
    """底稿 → [{kind: section/content, title, items: [(类型, 内容)], notes}]"""
    _, body = front_matter(src)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    slides, cur, lines, i = [], None, body.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# ") or line.startswith("## "):
            cur = {"kind": "section" if line.startswith("# ") else "content", "title": line.split(" ", 1)[1].strip(),
                   "items": [], "notes": ""}
            slides.append(cur)
            i += 1
            continue
        if cur is None or not line.strip():
            i += 1
            continue
        if line.strip() == "::: notes":
            j = i + 1
            while j < len(lines) and lines[j].strip() != ":::":
                j += 1
            cur["notes"] += "\n".join(lines[i + 1:j]).strip() + "\n"
            i = j + 1
        elif line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            kind = "mermaid-code" if line.strip() == "```mermaid" else "code"
            cur["items"].append((kind, "\n".join(lines[i + 1:j])))
            i = j + 1
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            cur["items"].append(("table", lines[i:j]))
            i = j
        else:
            m = re.match(r"^( *)(?:[-*]|\d+\.) +(.*)$", line)
            if m:
                cur["items"].append(("bullet", (len(m.group(1)) // 2, m.group(2))))
            elif re.match(r"^!(sim|mermaid|quiz) +(\S+)", line):
                k, v = re.match(r"^!(sim|mermaid|quiz) +(\S+)", line).groups()
                cur["items"].append((k, v))
            elif re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line):
                cur["items"].append(("image", re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line).groups()))
            elif line.strip() == "|||":
                cur["items"].append(("split", None))
            else:
                cur["items"].append(("para", line.strip()))
            i += 1
    return slides


# ======================================================================
# 三、排版
# ======================================================================

def rgb(h):
    return RGBColor.from_string(h)


def style_run(run, size, color=INK, bold=False, mono=False):
    f = run.font
    f.size, f.bold, f.color.rgb = Pt(size), bold, rgb(color)
    f.name = MONO if mono else FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def add_rich(paragraph, text, size, color=INK, bold=False):
    """支持 **加粗**（加粗并用强调色）和 `代码`"""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", plain_keep(text)):
        if not part:
            continue
        run = paragraph.add_run()
        if part.startswith("**"):
            run.text = part[2:-2]
            style_run(run, size, ACCENT, True)
        elif part.startswith("`"):
            run.text = part[1:-1]
            style_run(run, size, color, bold, mono=True)
        else:
            run.text = part
            style_run(run, size, color, bold)


def plain_keep(s):
    """去掉链接等标记，但保留 **加粗** 和 `代码`"""
    s = re.sub(r'<sup class="cite">.*?</sup>', "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)(\{[^}]*\})?", r"\1", s)
    s = re.sub(r"\{ *[.#:a-z][^}]*\}", "", s)
    return re.sub(r"<[^>]+>", "", s)


def text_width(s):
    """估算显示宽度（以汉字为 1）"""
    s = re.sub(r"\*\*|`", "", plain_keep(s))
    return sum(1 if ord(ch) > 0x2E80 else 0.55 for ch in s)


def fit_size(lines, width, height, biggest=28, smallest=14):
    """找一个让全部文字放得下的字号。lines: [(缩进层级, 文本)]"""
    for size in range(biggest, smallest - 1, -1):
        per_line = (width / 12700 - 20) / size          # 每行能放多少个汉字（EMU → 磅）
        rows = sum(max(1, math.ceil(text_width(t) / max(per_line - lv * 2, 4))) for lv, t in lines)
        need = rows * size * 1.35 + len(lines) * size * 0.45
        if need <= height / 12700:
            return size
    return smallest


def textbox(slide, x, y, w, h, lines, size=None, color=INK, bullets=True, anchor=MSO_ANCHOR.TOP):
    size = size or fit_size(lines, w, h)
    tf = slide.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for k, (lv, text) in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.space_after = Pt(size * 0.45)
        p.line_spacing = 1.15
        p.alignment = PP_ALIGN.LEFT
        if bullets:
            mark = "■ " if lv == 0 else "– "
            r = p.add_run()
            r.text = "    " * lv + mark
            style_run(r, size * (0.6 if lv == 0 else 0.9), ACCENT if lv == 0 else MUTED)
        add_rich(p, text, size - 2 * lv, color)
    return tf


def box(slide, x, y, w, h):
    """自动换行的文本框（不换行的文本框在 WPS、LibreOffice 里会被居中）"""
    tf = slide.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    return tf


def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def trimmed(path):
    """裁掉截图四周的白边，返回新图片路径"""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    if not box or box == (0, 0) + im.size:
        return path
    pad = 16
    box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height))
    out = BUILD / "figures/trimmed" / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    im.crop(box).save(out)
    return out


def picture(slide, path, x, y, w, h):
    """在 (x, y, w, h) 框内按比例放图，居中（先裁白边）"""
    from PIL import Image
    path = trimmed(path)
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def placeholder(slide, x, y, w, h, text):
    s = rect(slide, x, y, w, h, "F5F7FA", LINE)
    tf = s.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    add_rich(p, text, 16, MUTED)


class Deck:
    def __init__(self, c, ch):
        self.c, self.ch = c, ch
        self.text = chapter_text(ch)
        self.mermaids = mermaid_blocks(self.text)
        self.quizzes = parse_quizzes(self.text)
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.page = 0
        self.missing = set()

    def new(self, footer=True):
        s = self.prs.slides.add_slide(self.blank)
        self.page += 1
        if footer:
            tf = box(s, MARGIN, H - Inches(0.45), Inches(9), Inches(0.35))
            tf.word_wrap = True
            add_rich(tf.paragraphs[0], f"{self.c['title']} · {chapter_heading(self.ch)}", 10, MUTED)
            tf = box(s, W - MARGIN - Inches(1), H - Inches(0.45), Inches(1), Inches(0.35))
            tf.word_wrap = True
            tf.paragraphs[0].alignment = PP_ALIGN.RIGHT
            add_rich(tf.paragraphs[0], str(self.page), 10, MUTED)
        return s

    def notes(self, slide, text):
        if text.strip():
            slide.notes_slide.notes_text_frame.text = plain(text.strip()).replace("**", "")

    # ---------- 版式 ----------
    def title_slide(self):
        s = self.new(footer=False)
        rect(s, 0, 0, W, H, ACCENT)
        rect(s, MARGIN, Inches(4.15), Inches(1.2), Inches(0.08), "FFFFFF")
        tf = box(s, MARGIN, Inches(1.6), W - 2 * MARGIN, Inches(1))
        add_rich(tf.paragraphs[0], f"第 {self.ch['num']} 章", 28, "C5CAE9")
        tf = box(s, MARGIN, Inches(2.5), W - 2 * MARGIN, Inches(1.6))
        tf.word_wrap = True
        add_rich(tf.paragraphs[0], self.ch["title"], 44, "FFFFFF", True)
        h = self.ch
        info = f"理论 {h['theory']} 学时" + (f" · 实验 {h['lab']} 学时" if h["lab"] else "")
        tf = box(s, MARGIN, Inches(4.5), W - 2 * MARGIN, Inches(1.2))
        add_rich(tf.paragraphs[0], f"《{self.c['title']}》", 22, "FFFFFF")
        add_rich(tf.add_paragraph(), info, 16, "C5CAE9")
        tf = box(s, MARGIN, H - Inches(0.9), W - 2 * MARGIN, Inches(0.5))
        add_rich(tf.paragraphs[0], "人工智能技术应用专业系列网络教材 · " + site_url(self.c), 12, "C5CAE9")

    def section_slide(self, title):
        s = self.new()
        rect(s, 0, 0, Inches(0.35), H, ACCENT)
        rect(s, 0, Inches(2.6), W, Inches(2.0), SOFT)
        m = re.match(r"^(\d+(?:\.\d+)*) +(.*)$", title)
        no, name = (m.group(1), m.group(2)) if m else ("", title)
        if no:
            tf = box(s, Inches(1), Inches(2.75), Inches(2.6), Inches(1.7))
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            add_rich(tf.paragraphs[0], no, 54, ACCENT, True)
        tf = box(s, Inches(3.6 if no else 1), Inches(2.75), Inches(9), Inches(1.7))
        tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
        add_rich(tf.paragraphs[0], name, 36, INK, True)
        return s

    def header(self, s, title):
        rect(s, MARGIN, Inches(0.45), Inches(0.12), Inches(0.6), ACCENT)
        tf = box(s, MARGIN + Inches(0.25), Inches(0.35), W - 2 * MARGIN, Inches(0.8))
        tf.vertical_anchor, tf.word_wrap = MSO_ANCHOR.MIDDLE, True
        tf.paragraphs[0].alignment = PP_ALIGN.LEFT
        add_rich(tf.paragraphs[0], title, 28 if text_width(title) < 28 else 22, INK, True)
        line = s.shapes.add_connector(1, MARGIN, Inches(1.15), W - MARGIN, Inches(1.15))
        line.line.color.rgb = rgb(LINE)

    def figure(self, s, kind, value, x, y, w, h):
        """放一张图：动画截图 + 二维码、Mermaid 图或普通图片"""
        if kind == "sim":
            url = f"{site_url(self.c)}sims/{value}/main.html"
            png, qr = sim_png(value), qr_png(url)
            cap = Inches(0.5)
            if png:
                picture(s, png, x, y, w - (Inches(1.3) if qr else 0), h - cap)
            else:
                self.missing.add("动画截图")
                placeholder(s, x, y, w - (Inches(1.3) if qr else 0), h - cap, f"交互动画：{value}\n（运行 npm run figures 生成截图）")
            if qr:
                s.shapes.add_picture(str(qr), x + w - Inches(1.2), y + h - cap - Inches(1.2), Inches(1.2), Inches(1.2))
            tf = box(s, x, y + h - cap, w, cap)
            tf.word_wrap = True
            add_rich(tf.paragraphs[0], "动手试一试：" + url, 11, MUTED)
        elif kind in ("mermaid", "mermaid-code"):
            code = self.mermaids[int(value) - 1] if kind == "mermaid" else value
            png = mermaid_png(code)
            if png:
                picture(s, png, x, y, w, h)
            else:
                self.missing.add("Mermaid 图")
                placeholder(s, x, y, w, h, "Mermaid 图（运行 npm run figures 生成图片）")
        elif kind == "image":
            alt, src = value
            p = (chapter_path(self.ch).parent / src).resolve()
            if not p.exists():
                p = (ROOT / src).resolve()
            if p.exists():
                picture(s, p, x, y, w, h)
            else:
                placeholder(s, x, y, w, h, f"找不到图片 {src}")

    def table(self, s, rows, x, y, w, h):
        cells = [[plain_keep(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|[\s:|-]+\|$", r.strip())]
        nr, nc = len(cells), max(len(r) for r in cells)
        longest = max(text_width(c) for r in cells for c in r)
        size = 24 if nr <= 4 else 20 if nr <= 6 else 16 if nr <= 9 else 13
        if longest > 24:
            size -= 3
        if longest > 40:
            size = min(size, 13)
        row_h = Pt(size * 2.1)
        shape = s.shapes.add_table(nr, nc, x, y, w, min(h, row_h * nr))
        tbl = shape.table
        widths = [max(text_width(r[j]) if j < len(r) else 0 for r in cells) + 2 for j in range(nc)]
        for j in range(nc):
            tbl.columns[j].width = int(w * widths[j] / sum(widths))
        for i, r in enumerate(cells):
            for j in range(nc):
                cell = tbl.cell(i, j)
                cell.text = ""
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.fill.solid()
                cell.fill.fore_color.rgb = rgb(ACCENT if i == 0 else ("FFFFFF" if i % 2 else "F5F7FA"))
                add_rich(cell.text_frame.paragraphs[0], r[j] if j < len(r) else "", size, "FFFFFF" if i == 0 else INK, i == 0)

    def code(self, s, code, x, y, w, h):
        lines = code.split("\n")
        size = max(10, min(18, int((h / 12700) / (len(lines) * 1.3 + 1))))
        box = rect(s, x, y, w, min(h, Pt(size * 1.3 * len(lines) + size * 2)), "F5F7FA", LINE)
        tf = box.text_frame
        tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.TOP
        tf.margin_left = tf.margin_right = Inches(0.2)
        for k, l in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = l
            style_run(r, size, INK, mono=True)

    def quiz(self, s, n):
        q = self.quizzes[int(n) - 1]
        lines = [(0, f"（{q['kind']}）{q['stem']}")] + [(1, o) for o in q["options"]]
        tf = textbox(s, MARGIN, TOP + Inches(0.2), W - 2 * MARGIN, BOTTOM - TOP - Inches(0.4), lines, bullets=False)
        for p in tf.paragraphs[1:]:
            p.level = 0
        self.notes(s, f"答案：{q['answer']}\n{q['explain']}")

    def content_slide(self, sd):
        s = self.new()
        self.header(s, sd["title"])
        items = sd["items"]
        text = [v for k, v in items if k == "bullet"] + [(0, v) for k, v in items if k == "para"]
        figs = [(k, v) for k, v in items if k in ("sim", "mermaid", "mermaid-code", "image")]
        tables = [v for k, v in items if k == "table"]
        codes = [v for k, v in items if k == "code"]
        quizzes = [v for k, v in items if k == "quiz"]
        x, y, w, h = MARGIN, TOP + Inches(0.1), W - 2 * MARGIN, BOTTOM - TOP - Inches(0.1)
        if quizzes:
            self.quiz(s, quizzes[0])
        elif figs:
            if text:
                fw = int(w * 0.58)
                for i, (k, v) in enumerate(figs):
                    fh = h // len(figs)
                    self.figure(s, k, v, x, y + i * fh, fw, fh - Inches(0.1))
                textbox(s, x + fw + Inches(0.3), y, w - fw - Inches(0.3), h, text, anchor=MSO_ANCHOR.MIDDLE)
            else:
                fw = w // len(figs)
                for i, (k, v) in enumerate(figs):
                    self.figure(s, k, v, x + i * fw, y, fw - Inches(0.15), h)
        elif tables or codes:
            if text:
                th = min(int(h * 0.35), Inches(0.55) * len(text) + Inches(0.2))
                textbox(s, x, y, w, th, text, size=min(20, fit_size(text, w, th, 20)))
                y, h = y + th + Inches(0.1), h - th - Inches(0.1)
            if tables:
                self.table(s, tables[0], x, y, w, h)
            else:
                self.code(s, codes[0], x, y, w, h)
        elif text:
            textbox(s, x, y, w, h, text)
        self.notes(s, sd["notes"])

    def build(self, slides):
        self.title_slide()
        for sd in slides:
            if sd["kind"] == "section":
                s = self.section_slide(sd["title"])
                self.notes(s, sd["notes"])
            else:
                self.content_slide(sd)
        return self.prs


def build(num):
    c = course()
    ch = chapter_by_num(num, c)
    src = SLIDES / f"ch{num:02d}.md"
    if not src.exists():
        print(f"没有 {src.relative_to(ROOT)}，先用正文自动生成底稿（课堂使用前请精简）")
        draft(num)
    deck = Deck(c, ch)
    prs = deck.build(parse_deck(src.read_text(encoding="utf-8")))
    out = BUILD / "课件" / f"第{num:02d}章 {safe_name(ch['title'])}.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    note = f"（缺少{'、'.join(sorted(deck.missing))}，先运行 npm run figures）" if deck.missing else ""
    print(f"已生成 {out.relative_to(ROOT)}，共 {deck.page} 页{note}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("draft", "build"):
        print(__doc__)
        sys.exit(1)
    nums = [ch["num"] for ch in course()["chapters"]] if sys.argv[2] == "all" else [int(sys.argv[2])]
    for n in nums:
        draft(n, "--force" in sys.argv) if sys.argv[1] == "draft" else build(n)
