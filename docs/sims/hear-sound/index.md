---
title: 看见声音
description: 播放几段录音，看波形和声谱图同步变化，理解声音怎样变成数字、为什么声谱图可以当图片来识别（第 10 章）
---

# 看见声音

<iframe class="sim-frame" src="main.html" height="560" loading="lazy" title="看见声音"></iframe>

[全屏打开](main.html){ target=_blank } · 对应教材：[10.1 声音怎样变成数字](../../chapters/10-speech-apps/index.md#101)

## 教学说明

**学习目标**

1. 能说出波形和声谱图分别表示什么（记忆）；
2. 能从声谱图上指出不同词的区别，解释为什么可以把声谱图当图片来识别（理解）。

**画面与操作**

- 选一段声音，点“播放”，红线随声音在波形和声谱图上移动；
- 先播放“哆咪嗦”，看纯音在声谱图上是一条线、音越高线越靠上；
- 再比较 yes 和 no：yes 后半段的“s”是上方一片亮，no 整段都是横纹。

**数据来源**：四段录音来自谷歌 Speech Commands 数据集（CC BY 4.0，见 `labs/data/speech_commands_small/LICENSE.txt`）；“哆咪嗦”由编者用程序合成。声谱图由 Keras `MelSpectrogram` 层预先计算（与实验 10-2 的参数相同）。

**课堂用法建议**

1. 让学生闭眼听 yes 和 no，再看声谱图，说出“眼睛能看出的区别”；
2. 讨论：既然不同的词在声谱图上长得不一样，能不能用第 4 章认花的卷积网络来认词？引出 10.3 节。
