# Jev / Laya 与 H20 / 910B4：从问题读到证据

第一次阅读，请从[一步步认识 Jev / Laya 项目](artifacts/一步步认识Jev-Laya项目-friendly.md)开始，先只读第一阶段的五步。

这份文档参照 TensorRT 项目的逐步教材，每步只推进一个理解点，再用一个问题检查。沿用已掌握的 embedding、分词、batch 等基础，从一条银行消息走到分类、改进、验证和加载。

读完后再进入[一步步认识训练系统实验](artifacts/一步步认识训练系统实验-friendly.md)，理解多设备协作、分片布局、发布成本和实验取舍。

## 理解主线后，按需查证

| 需要查询 | 入口 |
|---|---|
| Laya 架构、输入输出、超参数与训练算法 | [Laya 模型技术说明](artifacts/Laya模型技术说明.md) |
| 三轮决策实验 | [完整报告](artifacts/decision_study/REPORT.md)、[Banking77 结果](artifacts/decision_study/REPORT_BANKING77-friendly.md) |
| H20 / NPU 系统证据 | [H20×16 报告](artifacts/16card-experiments/paper/report-friendly.md)、[历史组合结论](artifacts/experiment-conclusions-friendly.md) |
| 加载分类器或重新实验 | [运行步骤](artifacts/decision_study/BANKING77_REPRODUCE.md) |
| 全部原稿、报告与文件 | [文档地图](artifacts/README.md) |
| 决策历史与公共导出 | [历史说明](CONVERSATION_HISTORY-friendly.md)、[发布说明](PUBLICATION_NOTES.md) |

逐步稿整理于 2026-10-27，只重组已有证据，没有新模型实验。原始代码、数据和历史文档保留；基础权重不随包发布。`PUBLIC_SHA256SUMS` 校验公共文件。
