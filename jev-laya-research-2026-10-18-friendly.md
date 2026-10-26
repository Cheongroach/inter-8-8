# Jev 与 Laya：先理解决策模型，再选择训练或 Infra 工作

逐步学习入口：[一步步认识 Jev / Laya 项目](artifacts/一步步认识Jev-Laya项目-friendly.md)。这里用于读懂主线后查询模型结构与来源。

这是[2026-10-18 原调研](jev-laya-research-2026-10-18.md)的阅读版本。原稿的公开性能值是来源作者报告，非本项目实测；后续真实实验见[Laya 三轮报告](artifacts/decision_study/REPORT.md)。版本敏感事实按原调研日期阅读。

决策模型让程序直接获得一个选择、真假概率或评分分布，不必先生成一段文字再解析。例如输入“转账还没到账”，问题是“属于哪种业务”，允许答案是 77 个业务类别。返回类型正确，不代表业务判断正确。

## 输入和输出是什么

`state` 是本次判断的上下文，例如文档或工单，并非模型持久记忆。`questions` 说明判断任务与允许结果；`schema` 是这些输出类型、类别与顺序的约定。

| 输出 | 直觉 | 容易误解的地方 |
|---|---|---|
| Choice | 对候选类别给概率，并选一个 | 有限选项防止编造类别，仍可能选错 |
| Noul | 陈述成立的概率 | 不是保证事实为真 |
| Score | 有序等级分布及期望评分 | 不是随意生成连续数值 |
| Box，Jev 新版预览 | 图像上的区域输出 | 版本与实验接口条件需按官方说明核对 |

| 对象 | Jev | Laya |
|---|---|---|
| 交付方式 | TypeSafe 托管 API，权重闭源 | Convai 开放代码与权重 |
| 主干 | 所查公开资料未披露参数 / 结构 | 英文 ModernBERT-large，多语言 mmBERT-base |
| 客户适配 | 官方不提供客户数据微调 / LoRA | 有主干 / head 训练与校准入口 |
| 输入范围 | 原调研确认 1.14 与图像预览发布 | 所查权重家族为文本 |

不能从“并行决策”推测 Jev 的内部结构。Jev 当前没有本项目 API 结果，旧第三方分数不能补作新版本对照。

## Laya 怎样把文本变成选项概率

Encoder 是把整段输入变成 token 向量的主干，head 是额外读出模块。公开 Laya 路径大致为：

```text
问题＋各选项 marker / 描述＋state
  → 双向 Encoder
  → question type embedding＋额外 Transformer head
  → 读取每个选项 marker
  → 共享 scorer 给出各选项分数 z
  → 有效选项 softmax / 温度校准
  → 程序组装 Choice / Score / Noul
```

英文和 typed-decisions 作者标称约 421M 参数，多语言约 322M。原权重默认总序列预算为英文 512、多语言 1024、typed 1024；问题、候选和 state 共同占预算。单纯调高 max_len 不代表长文能力已经训练或验收。

一个请求里的多问题通常各自构造序列，放进 batch 一次 forward。这样能摊薄开销，但不是“主干只编码一次共享 state”。双向 attention 使 state 表征依赖问题和选项，不能直接跨任意问题复用完整 hidden states / KV 而保持数值等价。

tokenization 或固定 schema token 缓存仍可复用。若想真正只编码一次 state，再由不同问题读取，通常要改变结构并重新训练。

## 与普通 LLM 首 token 决策有什么不同

causal LLM 可以在问题前放 state，读取第一个回答位置的标签分数，并只在允许标签中归一化：

\[
p(k\mid\text{allowed})=\frac{\exp z_k}{\sum_{j\in\text{allowed}}\exp z_j}.
\]

它能省去长文本解码，也可能用 prefix KV cache；但如果所有标签在全词表中概率都很低，条件归一化仍会显得很自信。因此还要检查 `label_mass`、`argmax_is_label`。换一个接口不会自动获得 Jev 或 Laya 的能力。

原调研时 vLLM 已有 Structured Decisions 与 `/v1/systemone` 文档，所以同名 API wrapper 本身不是技术空白。ModernBERT pooling 支持也不自动覆盖 Laya 的额外 head、marker gather、可变候选 mask 和校准配置。

## 什么时候需要训练

| 目标 | 优先工作 |
|---|---|
| 调用 Jev | 输入 / schema、真实任务评测、阈值与 API 工程 |
| 正确部署已有 Laya | 权重、token 契约、输出映射、数值一致性 |
| 学会固定业务类别 | 监督线性头、领域微调 / 蒸馏，与简单基线比较 |
| 改善概率可信度 | 独立数据校准，绑定权重 / schema / dtype |
| 更低延迟或更高吞吐 | 在质量已经验证后优化 batch、bucket、kernel、图执行 |

有真实标签可用交叉熵 \(L=-\log p(y)\)；教师分布为 \(q\) 时可用 \(L=-\sum q_k\log p_k\) 蒸馏。教师分布可能有错，需要独立测试。

Laya RLCD 对分数加探索噪声，用分布评分规则与 policy-gradient 项，另含 soft-CE。严格适当评分规则在正确目标分布和理想优化条件下鼓励诚实概率，并不是任意神经网络已在分布外校准的证明。官方 / 社区所测领域中，额外 RL 项没有建立一致优势；所以先做 CE / 蒸馏对照，不能把 RL 当必要条件。

温度 \(p=\mathrm{softmax}(z/T)\) 调整尖锐程度，正温度不改 argmax。校准数据和测试数据要分开；多个问题来自同一工单时要按案例分组，防止关联泄漏。

## 已有公开分数怎样看

原稿保留作者 typed-decisions 和旧版第三方成对测试。Laya 领域微调的作者结果较基础权重更好，但某些行没有同一 harness 的结果文件，Jev 1.13 又是外部引用，因此不能称统一排行榜。

《Fast Models, Slow Evidence》用固定 state / questions 比较旧 Laya 0.3.21 与 Jev 1.13，指出大标签集合、干扰选项和顺序敏感性；它是预印本，旧运行时的结果不能直接当成所有新版 / 微调权重能力。其他 pilot 中 Jev 也出现真实标签零概率和阈值迁移问题，说明 API 同样需要领域验证。

Laya 作者已有 ONNX、TileLang、CUDA Graph、长度 bucket 路径，不是毫无运行时优化。作者 T4 延迟不等于本机测量；Jev 托管 API 包含网络和服务调度，不能直接对比本地 GPU forward。

实际决策成本应考虑：

\[
C_{\mathrm{decision}}=C_{\mathrm{local/API}}
+P(\mathrm{upgrade})C_{\mathrm{large\ model}}+C_{\mathrm{business\ error}}.
\]

自部署没有 API 单次收费，也不意味着设备常驻、CPU、运维和错误成本为零。

## 对 Infra 贡献，先从哪一处读代码

| 入口 | 要回答的问题 |
|---|---|
| `common.py` 的序列构造与 head | 选项截断、marker、类型 embedding 怎样进入输出 |
| `agent.py` 的 encode / infer / output | Request ID、问题 / 选项映射、batch 与校准 |
| `train.py` 的 soft-CE / RLCD | 目标、冻结参数、案例权重 |
| `calibrate.py` | 温度、拒答、与部署精度的关联 |
| vLLM ReadStrategy / decisions 文档 | 首 token、classification 与 joint schema 的不同路径 |

值得先验证的是原权重输出适配、可变候选 metadata、并发 ID 归属、长度分桶、图缓冲与概率一致性。真正共享 state 编码、任意新 schema 理解和中文能力可能涉及结构 / 数据训练。文本 Laya 不一定最适合 Omni 多阶段生成路径；多模态决策 head 则需要单独核对现有覆盖。

后续本项目已验证[Banking77 候选预算与监督头](artifacts/decision_study/REPORT_BANKING77-friendly.md)，但没有 Jev / 强微调 / 生成模型同协议对照，也没有 SOTA 证据。

主要来源：[TypeSafe 模型](https://docs.typesafe.ai/models)、[Jev 1.14](https://typesafe.ai/blog/jev-1-14)、[Laya 模型卡](https://huggingface.co/convaiinnovations/laya)、[Laya 代码](https://github.com/NandhaKishorM/laya)、[训练文档](https://nandhakishorm.github.io/laya/finetune/)、[vLLM decisions](https://docs.vllm.ai/en/latest/models/structured_decisions_models/)、[成对预印本](https://arxiv.org/abs/2610.02267)。原调研完整 20 项来源、旧数字、来源时效冲突保留；这里只重组阅读，不重新把旧来源当成最新事实。
