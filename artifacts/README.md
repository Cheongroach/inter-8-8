# Jev / Laya 与训练系统：从这里开始读

第一次阅读，请只打开[一步步认识 Jev / Laya 项目](一步步认识Jev-Laya项目-friendly.md)。

先读第一阶段的五步：业务去向 → 候选 → 概率输出 → 决策模型 → Jev / Laya。能回答每步的问题后，再进入下一阶段。embedding、分词、batch 等已有基础不再重复。

读完模型篇，再进入[一步步认识训练系统实验](一步步认识训练系统实验-friendly.md)。其余报告在需要查方法、数字或代码时再打开。

以下保留完整查证地图。逐步稿整理于 2026-10-27；决策质量数据截至 10 月 25 日，H20×16 数据截至 10 月 21 日。

## 研究范围

这个目录研究过两个问题。一个是“模型能不能把判断做对，给出的概率能不能信”；另一个是“把训练放在 H20 和昇腾上，能不能更快完成”。它们分别需要质量数据和系统数据，不能用训练速度回答模型能力问题。

Jev 是托管决策 API；Laya 是可下载、可训练的决策模型。决策模型直接从允许的选项中选择答案，并给出概率，例如把“转账还没到账”归入某个银行业务类别。H20 是 NVIDIA GPU，910B4 是昇腾 NPU；这里的“异构”指同时使用两类设备，“同构”指只用一类。

## 按需查证

1. 模型结构与来源：[模型与接口说明](../jev-laya-research-2026-10-18-friendly.md)。
2. 决策质量设置和完整结果：[三轮质量实验](decision_study/REPORT.md)、[Banking77 报告](decision_study/REPORT_BANKING77-friendly.md)。
3. 训练设置与证据边界：[H20×16 报告](16card-experiments/paper/report-friendly.md)、[混合设备实验后的选择](experiment-conclusions-friendly.md)。

遇到 accuracy、NLL、FSDP、HSDP、Local SGD 等缩写，可查[指标与术语](METRICS.md)。每份说明也会在首次使用时解释关键概念。

## 目前可以回答什么

| 问题 | 实际证据 | 当前答案 |
|---|---|---|
| Laya 在新闻分类中能否改善概率？ | 第二轮新保留集 512 条；准确率仍为 93.75%，NLL 0.1803 → 0.1746 | 本次概率评分改善，Brier 略差；不能称所有质量指标改善 |
| 银行业务分类能否提高准确率？ | 9,229 条训练标签；3,072 条测试；监督融合 91.67%，词法基线 90.82% | 净多答对 26 条，配对区间支持本次准确率改善；概率指标退化 |
| H20 在普通 Socket 网络下怎样训练更快？ | 两台共 16 卡；27B 全局分片 660.09 秒，机内分片 216.60 秒 | 并行布局有直接系统证据；每 16 步参数平均还可到 83.30 秒，但改变训练规则 |
| 混合 H20 / 910B4 能否胜过最佳同构？ | 历史 4＋4 已测；新版 NPU×16 与混合 8＋8 于 10 月 25 日确认取消 | 尚未证明；原三组强对照未完成 |
| Jev 或 Laya 是否达到 SOTA？ | Jev API 调用为 0，强微调及生成模型对照缺失 | 尚未建立同协议最先进水平证据 |

Banking77 融合是“75% 词法分类器概率＋25% 冻结 Laya Encoder 分类头概率”。Encoder 是把文本变成向量的主干，分类头是把向量变成类别分数的小模型。本轮只训练新分类头，原 Laya Encoder 和 typed head 都没有更新。

## 所有文档的阅读地图

有必要保留原始技术叙述、历史状态或实验前协议的文件，旁边增加 `-friendly` 说明；原稿保留为证据。入口和运行说明直接整理，保留原名字。数据归属说明保持原文。

| 内容 | 整理后的阅读版本 | 保留的技术 / 历史原稿 |
|---|---|---|
| 从具体请求逐步学习 | [模型篇](一步步认识Jev-Laya项目-friendly.md) | 后文链接原始证据 |
| 从同工作计时逐步学习 | [训练系统篇](一步步认识训练系统实验-friendly.md) | 后文链接原始证据 |
| Jev / Laya 结构、训练、推理 | [模型说明](../jev-laya-research-2026-10-18-friendly.md) | [原调研](../jev-laya-research-2026-10-18.md) |
| 三轮质量进展 | [REPORT](decision_study/REPORT.md) | 同名文件直接整理 |
| Banking77 完整结果 | [图解结果](decision_study/REPORT_BANKING77-friendly.md) | [原结果](decision_study/REPORT_BANKING77.md) |
| 第一轮实验规则 | [协议说明](decision_study/PROTOCOL-friendly.md) | [冻结协议](decision_study/PROTOCOL.md) |
| 第二轮实验规则 | [协议说明](decision_study/PROTOCOL_ROUND2-friendly.md) | [冻结协议](decision_study/PROTOCOL_ROUND2.md) |
| 第三轮实验规则 | [协议说明](decision_study/PROTOCOL_ROUND3-friendly.md) | [冻结协议](decision_study/PROTOCOL_ROUND3.md) |
| Banking77 复现、加载 | [操作步骤](decision_study/BANKING77_REPRODUCE.md) | 同名文件直接整理 |
| 数据归属与许可 | [ATTRIBUTION](decision_study/data/banking77/ATTRIBUTION.md) | 原文保留 |
| H20×16 主报告 | [图解报告](16card-experiments/paper/report-friendly.md) | [技术报告](16card-experiments/paper/report.md) |
| 新模型与三组对照 | [矩阵说明](16card-experiments/new-architecture-matrix-friendly.md) | [原矩阵](16card-experiments/new-architecture-matrix.md) |
| 排队历史与已回收数据 | [历史说明](16card-experiments/queue-status-friendly.md) | [排队记录](16card-experiments/queue-status.md) |
| 两组任务取消 | [终态说明](16card-experiments/cancellation-20261025-friendly.md) | [终态记录](16card-experiments/cancellation-20261025.md) |
| RDMA / Socket 日志 | [日志读法](16card-experiments/paper/transport-evidence-friendly.md) | [原日志](16card-experiments/paper/transport-evidence.txt) |
| 历史 4＋4 实验设计 | [设计说明](experiment-matrix-friendly.md) | [原矩阵](experiment-matrix.md) |
| 组合对照数据 | [表格读法](combination-comparison-friendly.md) | [完整表](combination-comparison.md) |
| rank 级性能记录 | [记录读法](experiment-results-report-friendly.md) | [完整记录](experiment-results-report.md) |
| 实验后的选择 | [结论说明](experiment-conclusions-friendly.md) | [原结论](experiment-conclusions.md) |
| 通信支持范围 | [可行性说明](communication-feasibility-friendly.md) | [技术原稿](communication-feasibility.md) |
| 两节点通信实测 | [实测说明](communication-results/report-friendly.md) | [原报告](communication-results/report.md) |
| 异构训练文献判断 | [比较逻辑](heterogeneous-training-decision-friendly.md) | [文献原稿](heterogeneous-training-decision.md) |
| 训练 Infra 调研 | [训练知识说明](research-friendly.md) | [完整调研](research.md) |
| 原项目方案 | [方案说明](proposal-friendly.md) | [完整方案](proposal.md) |
| 项目与简历表述 | [可用表述](project-pitches-friendly.md) | [历史模板](project-pitches.md) |

## 从解释回到证据

质量实验的[逐样本概率](decision_study/results_round3/test_predictions.jsonl)、[验证选择锁](decision_study/results_round3/selection.json)和[指标文件](decision_study/results_round3/results.json)可以独立复算。训练实验的[42 次计时](16card-experiments/paper/training-metrics.json)与[原始事件](16card-experiments/paper/events.jsonl)保留每次观测。

两张新图的[绘图入口](figures/render_friendly.py)和[绘图数据](figures/figure-data.json)随文档提供。这里只整理 Jev / Laya 与相关训练实验；其他项目目录的成果没有计入本项目。
