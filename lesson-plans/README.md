# 教案底稿

每章一个 `chNN.yml`，由 `python tools/make_lesson_plan.py build NN` 排版成 `build/教案/第NN章 章名 教案.docx`。学校没有统一模板，用通用格式：课程信息表、教学目标（知识 / 技能 / 素养）、教学重点与难点、教学资源、教学过程（课前线上 / 课中 / 课后线上）、作业与考核、证书考点、教学反思。

## 工作流程

1. `python tools/make_lesson_plan.py draft NN`：从正文和 `course.yml` 自动填好学时、线上 / 线下分配、教学目标、教学重点、动画、实验、课堂练习、作业。
2. 补写标着 `【待填】` 的地方（素养目标、教学难点和突破方法、各讲授环节的设计意图），检查时间分配，按实际教学安排增删环节。
3. `python tools/make_lesson_plan.py build NN`。还没填的 `【待填】` 在 Word 里显示为红字，build 时也会提示还剩几处。

教学资源（教材网址、动画、实验）和证书考点（正文里的 `!!! cert` 提示框）在 build 时直接从正文读取，不用写进底稿。

## 字段

| 字段 | 说明 |
|---|---|
| `chapter` | 章号 |
| `online` / `offline` | 线上 / 线下学时；默认按 `course.yml` 的 `lesson_plan.online_ratio` 拆分本章学时 |
| `methods` | 教学方法列表 |
| `objectives` | `knowledge` 知识目标、`skill` 技能目标、`quality` 素养目标，各是一个列表。默认按学习目标的认知层次分：记忆、理解归知识，应用及以上归技能 |
| `key_points` | 教学重点列表 |
| `difficulties` | 教学难点列表，每项 `point`（难点）和 `solution`（突破方法） |
| `resources` | 额外的教学资源（可选），会接在自动列出的资源后面 |
| `process` | 教学过程：每个阶段 `phase`（如“课前（线上）”）下有若干 `steps`，每步 `name` 环节、`minutes` 分钟、`content` 教学内容、`teacher` 教师活动、`student` 学生活动、`intent` 设计意图 |
| `homework` | 作业列表（默认是课后练习中的简答、实操题和实验报告） |
| `assessment` | 考核方式 |
| `reflection` | 教学反思，留空则在教案末尾留出手写位置 |

课中时间默认这样分：情景导入 10 分钟、课堂练习 10 分钟、小结 5 分钟，实验按本章实验学时，其余按各节篇幅分给讲授。线下学时 × 45 分钟 = 课中各环节分钟数之和，改时间时注意总数对得上。
