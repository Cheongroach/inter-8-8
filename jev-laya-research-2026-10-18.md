# Jev、Laya：决策模型、后训练与推理 Infra 调研

资料截止：2026-10-18。面向 AI Infra、vLLM / vLLM-Omni 的技术判断。

本报告依据官方文档、开放实现、公开实验和第三方成对评测。没有实际调用付费 Jev API、下载完整权重、开展训练或在 H20 上测性能。性能数字均为来源报告值，不能视作本机实测。

## 结论

Jev 与 Laya 的共同方向值得关注：把软件需要的分类、真假判断、评分和路由直接表示成模型输出，减少生成文本、解析文本及逐 token 解码的开销。它们适合放在 Agent 或业务系统的决策环节。[cite:1][cite:2][cite:3]

“需要训练、后训练，还是 Infra 适配”的答案取决于目标：

- **调用 Jev**：主要做 API、输入组织、问题分解、评测和阈值策略。目前官方不提供客户数据微调或 LoRA，同一套权重服务所有客户。[cite:4]
- **部署已有 Laya**：可以直接推理；需要正确的运行时、输入构造、模型选择和批处理。是否需要领域微调由任务评测决定，不能因为下载了权重就认为通用能力足够。[cite:2][cite:3]
- **让 Laya 学会业务判断**：通常应优先考虑领域微调、教师蒸馏和独立数据校准。已有实验中，领域微调带来的提升远大于是否额外加入 RLCD 项的差异。[cite:3][cite:6][cite:7]
- **给普通 LLM 提供类似接口**：可以只改 Infra，读取第一个回答 token 的候选标签概率；vLLM 上游已经有这条实现。它提供输出形式与推理优化，不自动赋予 Jev 的决策能力或概率校准。[cite:8]
- **研发新的高效决策模型**：模型读出方式、训练数据与目标、概率校准、Serving 都需要配合；不必从零预训练，也不能仅靠换接口获得新模型能力。

我的判断：Jev 值得作为决策质量与接口设计的参考；Laya 值得作为可复现、可微调的小模型研究平台。做开源贡献时，最有价值的组合是**有真实质量评测的决策推理能力、可验证的低延迟和概率一致性**。

## 1. 它们是什么

输入通常是 `state + questions`。`state` 是文档、邮件、工单、文本化程序状态或 JSON；`questions` 指定要判断什么，以及允许哪些结果。这里的 state 是本次判断的输入上下文，不等于模型自带持久记忆。

| 维度 | Jev | Laya |
|---|---|---|
| 提供方 | TypeSafe AI | Convai Innovations |
| 交付形态 | 托管 API，闭源权重 | Apache 2.0 开放权重和代码 |
| 典型输出 | Choice、Score、Noul；新版预览增加 Box | Choice、Score、Noul |
| 主干 | 所查公开资料没有披露具体主干、参数量与层数 | 英文使用 ModernBERT-large；多语言使用 mmBERT-base |
| 读出 | 官方称并行输出决策；内部实现不公开 | 双向 Encoder、额外 Transformer head、选项 marker scorer |
| 客户微调 | 官方明确不做客户数据微调或 LoRA | 提供完整微调入口，可训练主干和决策头 |
| 多模态 | 10 月发布的预览能力支持图像及 Box | 所查官方权重家族是文本决策模型 |

以上是截至调研日期的来源描述。[cite:1][cite:2][cite:3][cite:4][cite:5][cite:9]

三种主要文本决策：

- **Choice**：对调用时给定的候选选项输出概率分布，并选择一个选项。
- **Score**：对有序评分等级输出概率分布，返回期望评分；不是任意连续数值生成。
- **Noul**：返回某个陈述成立的概率，可理解成 `P(true)`。

候选选项可以在请求时定义，因此它们与“最后一层固定输出若干业务类别”的普通分类器存在区别。但新 schema 不要求重建模型，不代表模型必然能理解任意新任务。[cite:2][cite:3]

**“零幻觉”需要限定含义**：有限选项和程序化输出能避免模型编造不存在的类别、生成不符合 schema 的文本。它们仍会错误分类、漏检、过度自信。输出类型正确和判断事实正确是两种指标。[cite:1][cite:10][cite:11]

## 2. Laya 的实现与性能来源

官方家族描述如下。参数量为作者标称，没有在本次调研中重新统计。[cite:2][cite:3]

| 权重 | 主干 | 总参数 | 默认序列长度 | 使用侧重点 |
|---|---|---:|---:|---|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 tokens | 英文文本任务 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 tokens | 多语言；运行时可显式增至 8192 |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 tokens | 在 typed-decisions 任务上微调 |

默认长度是问题、选项和 state 共同使用的序列预算。提高 `max_len` 不等于经过了对应长文任务训练，也不能沿用短文本的性能数字。

核心序列：

```text
[CLS] question type + instructions [SEP]
[MASK] option 0 [MASK] option 1 ... [SEP]
state [SEP]
```

公开 `DecisionModel` 的核心路径：

```text
完整序列
  → 双向 Encoder 的 token hidden states
  → 加入 question type embedding
  → 额外 TransformerEncoder head（默认两层）
  → gather 每个 [MASK] 对应的 hidden state
  → 共享 scorer 输出各选项 logits
  → 按该问题的有效选项做 softmax / calibration
  → 由程序组装 choice、score、noul
```

这些结构在较早的完整代码与当前实现、官方模型卡中相互吻合。额外还有 `act_head`，读取 pooled hidden state 与分布特征；仓库目前明确记录其 `act_probability` 不适合直接作为行动门控。[cite:2][cite:3][cite:12][cite:13]

### 一次 forward 的准确含义

Laya 把多个问题构造成多条序列，放到 batch 中一起运行。**一次调用里的一次 batched forward，不等于主干仅编码一次共享 state。** 多个问题可以复用 state 的 tokenization，但 state hidden states 依然与各自的问题、选项共同编码。[cite:12][cite:13]

这解释了为什么“并行问题”仍增加计算量。作者在 T4 上报告，多语言权重单问题为 32.8 ms，十个问题一批为 72.3 ms；这支持批处理摊薄开销，不支持问题数增加完全免费。[cite:3]

性能收益主要来自：

1. 小型主干，比大生成模型少得多的模型计算。
2. 直接读出 logits，避免长文本生成与解析。
3. 多问题、多请求批处理，提高 GPU 使用率。
4. 算子融合、CUDA Graph、长度分桶等运行时优化。

Laya 已有 PyTorch / Transformers 路径、ONNX Runtime 和可选 TileLang fast path。TileLang 路径融合矩阵乘、激活、归一化、RoPE、attention，并针对形状 bucket 捕获 CUDA Graph；不是还没有任何优化的原型。[cite:3]

### 与 KV Cache 的区别

Laya 的主干是双向 attention，state 的表示受问题、选项以及整个序列影响。不能把 state 单独算一次，再直接跨任意问题复用完整 hidden states 或每层 KV，要求数值等价的情况下尤其如此。

Tokenizer 结果缓存、固定 schema 的 token 缓存、候选 embedding 缓存仍然有价值。若希望真正只编码一次 state，再用不同问题读取，需要考虑独立 state encoder 加 cross-attention / readout 等结构，通常涉及重新训练。

普通 causal LLM 的首 token 决策则不同：把 state 放在问题之前，可利用因果性复用 state prefix 的 KV。vLLM 当前决策接口正是这样描述其 prefix caching 的。[cite:8]

## 3. 训练、后训练和校准的边界

### 已有预训练主干足够作为起点

Laya 使用预训练语言 Encoder，再训练额外决策头，并微调 Encoder。该路线是“预训练主干上的适配 / 后训练”，不要求重做基础语言预训练。[cite:2][cite:3]

```mermaid
flowchart LR
    B["已有预训练主干"] --> T["领域微调或蒸馏"]
    T --> M["决策模型"]
    M --> C["独立数据校准"]
    M --> I["推理与服务适配"]
    C --> I
```

训练与 Infra 可以并行推进；最终上线的校准参数应匹配实际推理版本与数值精度。

### 有标签数据时，先建立 CE / 蒸馏基线

若每个样本有正确选项，使用交叉熵即可：

```text
L_CE = -log p(correct option)
```

若教师提供完整分布 `q`，使用 soft cross-entropy：

```text
L_soft_CE = -Σ_i q_i log p_i
```

这属于分布蒸馏。教师概率本身也可能错误或未校准；应保留来自真实结果、人工标注或可验证任务的独立评测。

固定业务分类可以先试冻结主干、只训练 head，减少训练开销；较复杂任务则比较全量微调。Laya 当前 `TrainConfig` 明确支持 `freeze_encoder`，默认关闭。[cite:6][cite:14]

### RLCD 做了什么

TypeSafe 把 RLCD 定位成一种后训练方法，目标是校准决策。其内部算法与数据细节没有在所查资料中完整公开，不能假设与 Laya 的实现相同。[cite:1][cite:15]

Laya 的公开 RLCD：

- 模型报告一个概率分布；
- 对 logits 添加高斯探索噪声，形成多个样本；
- 使用 log score、spherical score 和有序等级的 ranked probability score 评价分布；
- 用 GRPO 风格的 policy-gradient 项训练；
- 同时加入对教师目标分布的 soft cross-entropy。[cite:6][cite:12][cite:14]

“严格适当评分规则”的理论意思是：在正确的目标分布和理想优化条件下，诚实报告概率可最大化期望得分。它不是“神经网络在任意分布外任务上都已校准”的证明。有限数据、优化误差、标签偏差与分布变化都会影响校准。

### 不应把 RL 当成必要条件

Laya issue #741 的三路对照，在该任务上没有发现增加 RL 项优于 soft-CE 的可测收益。当前训练代码保留 `rlcd` 与 `soft-ce` 两种模式，并明确引用这一实验。[cite:6][cite:7]

官方微调文档另报告 typed-decisions 的单种子实验：RLCD accuracy 为 0.7715，soft-CE 为 0.790。这个结果支持先做简单基线，不足以证明所有任务上 CE 都更好。[cite:6]

RL 更值得尝试的情形包括：没有直接标签、奖励来自环境交互结果，或目标包含不可微的端到端业务效用，例如错选工具的代价、升级到大模型的成本。即便如此，也应与监督学习、蒸馏或直接可微目标比较。

### 校准不能省略

Softmax 输出和“概率能被当成真实频率使用”不同。常见后处理是温度缩放：

```text
p_i = softmax(z_i / T)
```

训练集用于学权重；校准集用于拟合温度与阈值；测试集用于衡量最终质量。多问题同属一个原始工单或对话时，建议按原始案例分组切分，避免相关问题泄漏。

Laya 仓库明确说明基础权重可能过度自信，多语言权重没有拟合好的温度。门控阈值应匹配任务、语言、选项数量和推理 dtype，而不是照搬示例里的数字。[cite:3][cite:6]

## 4. 公开效果证据与限制

### Laya 的优势主要由领域微调支撑

作者公布的 typed-decisions 测试包含 400 个案例、2000 个决策：[cite:3]

| 模型 | Accuracy | 证据与限制 |
|---|---:|---|
| Laya 英文基础权重 | 0.362 | 作者报告，引用已提交运行结果 |
| Laya 多语言基础权重 | 0.352 | 作者报告 |
| 多数类基线 | 0.461 | 作者报告 |
| Laya 领域微调权重 | 0.766 | 作者报告；README 说明该行尚无已提交结果文件 |
| Jev 1.13 | 0.727 | Laya 文档引用外部发布结果，不是在同一 harness 下重测 |

因此不能把这一表称为“Laya 全面击败 Jev”。它最能支撑的判断是：**Laya 基础权重在这些工作流上表现不足，领域微调明显有用。**

### 成对评测提供更强比较依据

2026-10-01 的预印本《Fast Models, Slow Evidence》用相同 state 和 questions 比较 Laya 0.3.21 与 Jev 1.13。以下是其中的代表结果：[cite:10]

| 决策任务 | Laya | Jev | 实际含义 |
|---|---:|---:|---|
| 工具选择，随机干扰项 | 84.0% | 99.5% | 在此数据与设置下 Jev 优势明显 |
| 大标签集合 | 35.0% | 81.0% | Laya 的输入预算和候选表示问题突出 |
| RAG relevance gate | 60.5% | 61.0% | 这组测试没有明显赢家 |
| 便宜 / 强模型路由 | 48.5% | 50.0% | 两者都接近平衡集随机基线 |

该论文还报告 Laya 在选项顺序反转时改变 30.3% 的答案，Jev 为 1.8%。这些是旧版运行时与所用 checkpoint 上的结果。Laya 后续已有 shuffle-options 训练和 parallel layout，不能把旧结果直接当成所有新版微调权重的表现。[cite:3][cite:6][cite:10]

这篇论文是预印本，任务标签、候选构造和措辞仍有限制。适合用于识别问题，不足以建立普遍排行榜。

### Jev 也需要领域验证

另一独立评测在 DAIR Emotion 上观察到，Jev 对部分真实标签给出零概率，显示“校准决策”也不能免除失效分析。[cite:11]

第三方完整 Banking77 / CLINC150 评测中，验证集选好的阈值到测试集可能超出原先误差目标。阈值的经验表现不会自动变成业务 SLA。[cite:16]

### 比较成本和延迟时应统一口径

Laya 的本地 GPU 时间不能直接作为 Jev 模型内部计算时间的比较：Jev API 耗时包含网络和服务调度。预印本中，同一 Jev 从不同客户端地点测得明显不同延迟。[cite:10]

“自部署无 API 费用”不等于运行成本为零。应计入 GPU 利用率、模型常驻、CPU tokenization、排队、服务运维和升级到大模型的调用。可用下式做真实流量核算，而不是宣称固定倍数：

```text
每次决策总成本
  = 本地或 API 决策成本
  + 升级概率 × 大模型成本
  + 错误决策的业务成本
```

## 5. 当前 vLLM 能做什么

截至调研日期，vLLM 上游已有专门的 Structured Decisions Models 文档与 `/v1/systemone`。因此“开发一个相同名称的接口”并不是空白机会。[cite:8]

### 普通 LLM 首 token 路径：可以仅做 Infra

现有实现将每个问题写成 state、指令、候选标签组成的 prompt，读取首个回答 token 的标签 logprobs，并在候选标签范围内归一化：

```text
p(option i | allowed labels)
  = exp(logit(label i)) / Σ_allowed exp(logit(label j))
```

这可以避免完整解释文本和长解码。但如果模型在整个词表中只给标签很少概率，候选集合内归一化仍会显得很自信。上游因此提供 `label_mass` 和 `argmax_is_label` 等诊断量。[cite:8]

这条路径：

- 不需要新建决策头，也不要求重新训练才能运行；
- 有机会使用 prefix caching 复用共同 state；
- 质量、选项顺序偏差、校准仍取决于基座和 prompt；
- 不是开放版 Jev 权重，也不是 Laya 原模型的等价推理实现。

文档当前对普通标签读出路径限制为每请求 64 个问题、Choice 26 个选项、Score 10 个等级；专用决策模型可能有不同上限。这些限制是当前实现描述，需按实际版本检查。[cite:8]

### Laya 原权重路径：需要专用适配

vLLM 已支持 ModernBERT 的 embedding / sequence-classification，但这不能自动覆盖 Laya 的完整模型：

- 额外 Transformer head；
- question type embedding；
- 每问题可变的 `[MASK]` marker；
- marker gather 和共享 scorer；
- 可变选项 mask；
- 校准配置与 typed output。[cite:12][cite:13][cite:17]

所查 vLLM Structured Decisions 支持表没有列出 Laya。实现时可以优先考察 pooling / token-classification 扩展与模型插件，而不是强行套用长文本生成循环。[cite:8]

本工作区目前未发现 Laya、`/v1/systemone` 或 ReadStrategy 的本地实现，当前 Python 环境也未安装 `vllm`。这里对上游的判断来自官方源码和文档，不是对工作区完成运行验证。

### 适合 AI Infra 的工作

| 工作 | 解决什么 | 能否只改 Infra |
|---|---|---|
| 原权重加载和输出适配 | 模型能正确运行 | 通常可以 |
| marker / variable options metadata | logits 与问题、选项正确对应 | 可以 |
| 请求和问题映射、异构 batch 拆装 | 并发正确性及结果归属 | 可以 |
| 长度分桶、microbatch、padding 优化 | 计算浪费与排队延迟 | 可以 |
| CUDA Graph buffer 管理 | launch overhead 与并发串数据 | 可以 |
| 算子融合与量化 | 延迟、吞吐、显存 | 可以，但要复测概率和决策边界 |
| 概率校准文件版本绑定 | 校准配置匹配权重及 dtype | Infra 加离线校准 |
| 候选 shortlist / tournament | 大标签预算不足 | 可以缓解；需衡量召回和整体质量 |
| 真正只编码一次共享 state | 避免多问题重复主干计算 | 对现有 Laya 通常需要结构变化与训练 |
| 消除权重的选项位置偏差 | 语义判断稳定性 | 训练增强或新的 attention / position layout |
| 中文任务、否定理解、零样本模型路由 | 模型能力不足 | 通常需要数据、训练和评测 |

对 Laya 的 shortlist / tournament，输出概率只覆盖保留选项或最终候选，不能当成原始全部标签的无条件概率。[cite:3][cite:18]

Laya 当前的 `option_layout="parallel"` 通过重复选项位置与 attention mask 隔离候选，追求选项顺序不变性。官方说明已发布权重仍是 sequential layout，parallel 需要专门训练；当前 ONNX、TileLang 和 compile 路径还不支持该布局。这是训练与 Infra 相互约束的具体例子。[cite:3][cite:13]

## 6. 对 vLLM-Omni 与个人贡献的建议

**文本 Laya 适配**优先考虑 vLLM pooling / 决策模型层，或 Laya 自身运行时。仅因为它是新模型，不能推出 vLLM-Omni 的多阶段生成流水线是最合适入口。

**多模态决策**更契合 Omni：从视觉、音频或 Omni 主干获取表示，以决策 head 输出路由、评分或是否继续推理，而非必须生成文字、图像或音频。上游已有多模态 JointSchema / classification 先例；Jev 最新预览也正在扩展输入和输出类型。[cite:5][cite:8]

对于你关注的 AI Infra，建议按以下顺序推进一个可衡量的贡献：

1. 选定真实的决策任务和已有基线，使用同一输入与候选 schema 比较 Laya、普通 LLM 首 token 路径、专用 classifier / reranker。
2. 判断质量瓶颈来自模型、token 截断还是运行时。模型能力不足时先领域训练；运行时瓶颈则做批处理、算子或 metadata 适配。
3. 如果采用 Laya，先用 soft-CE / 教师蒸馏建立基线，再对照 RLCD，不把 RL 作为先决条件。
4. 优化后同时报告端到端 p50 / p95、吞吐、显存、Accuracy / Macro-F1、NLL / Brier / ECE、固定误差下覆盖率和选项顺序稳定性。
5. 保留权重版本、schema、dtype、校准参数的关联，验证优化没有改变错误方向、概率阈值或请求归属。

推荐优先级：**可复用的决策 readout 与调度正确性 > 有质量约束的推理加速 > 单个 API wrapper**。多模态 head 值得探索，但应先确认已有上游模型和接口的覆盖范围。

## 7. 关键代码阅读入口

| 来源 | 阅读对象 | 关注问题 |
|---|---|---|
| Laya `laya/common.py` | `build_head` / `build_sequence` | state 与 schema 怎样拼接、怎样截断 |
| 同上 | `DecisionModel` | Encoder、额外 head、type embedding、marker scorer |
| 同上 | `collate_items` | 不同长度、不同选项数怎样组 batch |
| Laya `laya/agent.py` | `Agent` 的 encode / infer / output 路径 | 推理、校准、结果映射、并发控制 |
| Laya `laya/train.py` | `soft_ce_loss` / `rlcd_loss` | RL 项与 soft targets 的关系 |
| 同上 | `finetune` / `split_calibration` | 权重训练与校准数据切分 |
| Laya `laya/calibrate.py` | 温度拟合与 abstention 阈值 | 概率如何成为可验证的门控 |
| vLLM decisions 文档及 ReadStrategy | 首 token、JointSchema、classification | 不同决策主干如何共用服务接口 |

当前 main 文件较长，本次 WebFetch 对部分内容存在截断。完整基础结构通过 v0.3.7 的较短代码确认；当前行为以 main 文档及能读取的代码为依据，未声称逐行审查整个仓库。[cite:8][cite:12][cite:13][cite:14]

## Sources

所有下列网页在 2026-10-18 调研。官方结构与 API 事实属于第一手描述；作者 benchmark 与社区实验均注明来源，不能视为独立复现实验。

- [cite:1] TypeSafe：Introducing System One Models & Jev（官方产品、训练方向和并行读出描述）。https://typesafe.ai/blog/introducing-system-one-models-and-jev
- [cite:2] Laya 官方模型卡（权重家族、结构、接口）。https://huggingface.co/convaiinnovations/laya
- [cite:3] Laya 官方仓库 README（当前运行时、性能、已知限制及训练说明）。https://github.com/NandhaKishorM/laya
- [cite:4] TypeSafe Models（客户微调政策、版本与旧版模型描述）。https://docs.typesafe.ai/models
- [cite:5] TypeSafe：Jev 1.14 发布，2026-10-14。https://typesafe.ai/blog/jev-1-14
- [cite:6] Laya 官方微调文档（CE / RLCD 对照、数据、校准与入口）。https://nandhakishorm.github.io/laya/finetune/
- [cite:7] Laya issue #741：社区三路训练目标对照。https://github.com/NandhaKishorM/laya/issues/741
- [cite:8] vLLM Structured Decisions Models 官方文档；并核对其 main 源文档。https://docs.vllm.ai/en/latest/models/structured_decisions_models/ · https://raw.githubusercontent.com/vllm-project/vllm/main/docs/models/structured_decisions_models.md
- [cite:9] TypeSafe Images（图像输入、版本要求与实验性质）。https://docs.typesafe.ai/images
- [cite:10] Jiawei Li：Fast Models, Slow Evidence，2026-10-01，预印本及原始结果仓库。https://arxiv.org/html/2610.02267v1 · https://github.com/David-DL-Space/sys1-eval
- [cite:11] AbdelStark：jev-benchmarks，300 个样本的独立 pilot，概率质量与局限。https://github.com/AbdelStark/jev-benchmarks
- [cite:12] Laya v0.3.7 `common.py` 完整基础结构参考。https://raw.githubusercontent.com/NandhaKishorM/laya/v0.3.7/laya/common.py
- [cite:13] Laya main `common.py` 与 `agent.py`，当前输入和运行时实现。https://github.com/NandhaKishorM/laya/blob/main/laya/common.py · https://github.com/NandhaKishorM/laya/blob/main/laya/agent.py
- [cite:14] Laya main `train.py` 与官方 API reference。https://github.com/NandhaKishorM/laya/blob/main/laya/train.py · https://nandhakishorm.github.io/laya/reference/train/
- [cite:15] TypeSafe AI primer（RLCD 定位、校准含义）。https://docs.typesafe.ai/introduction/machine-learning-primer
- [cite:16] nibzard：decision-model-benchmark，完整测试与验证阈值迁移结果。https://github.com/nibzard/decision-model-benchmark
- [cite:17] vLLM Classification Usages（ModernBERT、pooling、分类能力）。https://docs.vllm.ai/en/latest/models/pooling_models/classify/
- [cite:18] Laya Helpers reference（shortlist、tournament、序列预算）。https://nandhakishorm.github.io/laya/reference/helpers/
- [cite:19] Vercel：2026-10-15 Jev 1.14 Gateway 上线公告。https://vercel.com/changelog/typesafe-ais-jev-1-14-is-now-available-on-ai-gateway
- [cite:20] TypeSafe Confidence（confidence 与 probability 的区分和公式）。https://docs.typesafe.ai/confidence

### 来源时效冲突

TypeSafe Models 页面在本次读取中仍将 Jev 1.13 列为当前版本，并表示文本输入；但 10 月 14 日的官方发布和 Images 页面已经说明 1.13 / 1.14 的实验图像能力，Vercel 10 月 15 日也确认 1.14 上线。因此本文按新发布确认图像能力，并不假设所有入口的别名或文档同步完成。[cite:4][cite:5][cite:9][cite:19]

Laya Hugging Face 模型卡和 GitHub README 的版本、部分 benchmark 行也存在不同步。本文把版本敏感行为优先归于当前 GitHub 文档，旧版数字仅在其明确的实验范围内使用，不拼成统一排行榜。
