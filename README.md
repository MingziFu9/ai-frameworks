# 人工智能技术应用专业网络教材模板

这是人工智能技术应用专业（成人大专）系列网络教材的公共模板，从《自然语言处理导论》（https://mingzifu9.github.io/nlp-textbook/ ）提炼而来。每门课从这个模板新建一个仓库，网页样式、在线答题、交互动画框架、学习图谱和维护脚本都已经配好，只需要写内容。

## 模板里有什么

| 路径 | 内容 |
|---|---|
| `course.yml` | **课程配置**：书名、学分学时、章节、附录、证书。改完运行 `python tools/course.py sync`，导航、首页、章节骨架等自动更新 |
| `编写规范.md` | **动笔前先读**：一章的结构、篇幅、实验、题型、文献、动画、术语表和配套资源的统一要求 |
| `系列公共术语表.md` | 七门课共用的 91 个术语，统一译名和定义 |
| `docs/chapters/01-sample/` | 示例章，演示每一种元素的写法；写第一章时整章替换 |
| `docs/sims/` | 交互动画，`common/` 是公共样式和脚本，`example-counter/` 是示例 |
| `docs/learning-graph/` | 学习图谱（概念和先修关系写在 `data.js`） |
| `docs/certificate.md` | 证书考点对照页，没有对应证书的课程从 nav 里删掉 |
| `docs/appendix/` | 附录 A Python 速查、B 工具库速查、C 实验环境与数据说明、D 术语表 |
| `labs/` | 实验代码（按章分目录）、数据和 `requirements.txt` |
| `slides/` | 每章课件底稿 `chNN.md`，写法见 `slides/README.md` |
| `lesson-plans/` | 每章教案底稿 `chNN.yml`，写法见 `lesson-plans/README.md` |
| `tools/` | 维护脚本，见下文 |
| `.github/workflows/pages.yml` | 推送到 `main` 后自动发布到 GitHub Pages |
| `.github/workflows/exports.yml` | 手动运行（或推送 `v1.0` 这样的标签），在 GitHub 上生成全部课件、教案和 Word/PDF 版供下载 |

## 用模板新建一门课

1. 在 GitHub 上打开本仓库，点 **Use this template → Create a new repository**，仓库名用课程的英文短名，例如 `ml-textbook`。
2. 新仓库的 **Settings → Pages → Source** 选 **GitHub Actions**。
3. 按大纲填写 `course.yml`（书名、仓库名、学分学时、各章目录名和标题、证书），运行 `python tools/course.py sync`：`mkdocs.yml` 的站点名和导航、首页课程信息和目录、术语表各章小节、学习图谱章节列表都会更新，还没有的章节按统一体例生成骨架。以后改章名、加章节、加动画，都是改完再 sync 一次。
4. 手动修改 `docs/index.md` 的简介、`docs/course-description.md`、`docs/certificate.md`；本 README 改写成本课的说明。写完第 1 章后删掉 `01-sample`和 `docs/sims/example-counter/` 里的示例内容。课件和教案底稿用 `make_slides.py draft N`、`make_lesson_plan.py draft N` 生成，格式示范见模板包的 `示例输出/`。
5. 推送到 `main`，几分钟后网页出现在 `https://<用户名>.github.io/<仓库名>/`。

没有实验的课（如导论）：`course.yml` 里把 `lab_section` 改成“在线体验”、`lab_word` 改成“体验”、`frontier_title` 改成“身边的 AI”，章节骨架、首页目录和统计会跟着用这些叫法。

## 维护脚本

```bash
python tools/glossary_links.py anchors     # 给术语表每个术语加锚点
python tools/glossary_links.py link 03     # 第 3 章正文中的术语改成悬停可见定义的链接
python tools/glossary_links.py concepts 03 # 生成第 3 章的“本章概念清单”
python tools/graph_text.py                 # 检查学习图谱并生成文字版
python tools/book_stats.py                 # 更新首页“全书概览”统计表
python tools/course.py sync                # 按 course.yml 更新导航、首页、章节骨架等
python tools/course.py new-sim 名称 3 "标题" # 新建第 3 章的交互动画骨架
python tools/export_docx.py --pdf          # 导出 Word 版和 PDF 版到 build/（需要 pandoc 和 LibreOffice）
```

## 课件、教案和 Word/PDF 版

```bash
pip install -r requirements-tools.txt      # 第一次：Python 库
npm install && npx playwright install chromium   # 第一次：渲染配图用的 Node 库和浏览器
npm run figures                            # 把 Mermaid 图和交互动画渲染成图片（课件、Word 版要用）

python tools/make_slides.py draft 3        # 从第 3 章正文生成课件底稿 slides/ch03.md，精简后再 build
python tools/make_slides.py build 3        # → build/课件/第03章 章名.pptx（all 表示全部章节）
python tools/make_lesson_plan.py draft 3   # 生成教案底稿 lesson-plans/ch03.yml，补写“【待填】”后再 build
python tools/make_lesson_plan.py build 3   # → build/教案/第03章 章名 教案.docx
python tools/export_docx.py --pdf          # → build/书名.docx 和 .pdf，动画换成截图和二维码
```

本机不想装这些工具时，在 GitHub 仓库的 **Actions → 导出课件、教案和 Word/PDF → Run workflow** 运行一次，几分钟后在运行页面底部的 Artifacts 下载全部文件。底稿（`slides/`、`lesson-plans/`）要提交到仓库，生成的文件不提交。

## 本地预览

```bash
pip install -r requirements-docs.txt
mkdocs serve          # 浏览器打开 http://127.0.0.1:8000
mkdocs build --strict # 提交前确认没有警告
```

## 说明

书中运行结果均为编者实测；需要 GPU、联网下载模型或调用大模型 API 的部分标为“待实测”，并附记录表。数据集来源与许可见 `labs/data/README.md`，仅用于教学。
