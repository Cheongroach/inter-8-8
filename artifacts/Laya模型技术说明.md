# Laya 模型技术说明

本文面向已经理解 Transformer、embedding、softmax 和基本训练流程的读者，目标是建立一套完整但紧凑的 Laya 认知框架。

文中的“Laya 模型”指固定源码
[`7a77439b`](https://github.com/NandhaKishorM/laya/tree/7a77439b0923d6efe00052d50a7f787840bfce98)
与英文 checkpoint revision
[`7b928d82`](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148)。
“本项目”指该 checkpoint 上完成的 AG News 校准和 Banking77 领域路由实验。

## 1. 核心定位

Laya 不是缩小版 Qwen，也不是让 Qwen 少生成几个 token。它把任务重新定义为：

> 给定一个状态、一个问题和有限候选，直接输出候选概率。

其完整主线是：

```text
state + typed question + options
               ↓
       双向 Encoder
               ↓
  question type embedding
               ↓
    typed decision head
               ↓
  各 option marker 的 logit
               ↓
 temperature + softmax
               ↓
 Choice / Score / Noul
```

这种结构服务于分类、路由、真假判断和等级评分。它不做自回归文本生成，因此没有逐 token 解码、停止词和回答解析等生成链路。

## 2. Laya 与 Qwen3 的架构差异

| 维度 | Laya | Qwen3 等主流生成模型 |
|---|---|---|
| 主干 | 双向 Encoder；英文 checkpoint 使用 ModernBERT-large | decoder-only causal Transformer |
| 训练目标 | 对调用时给定的候选分布建模 | 预测下一个 token |
| 输入依赖 | state、问题和候选共同编码 | prompt 经过 causal mask 编码 |
| 输出位置 | 每个候选对应一个 marker；共享 scorer 给 marker 打分 | LM head 对回答位置输出全词表 logits |
| 推理方式 | 非自回归；一个问题对应一个决策序列 | 自回归生成一个或多个 token |
| 输出空间 | 当前请求定义的有限候选 | 固定全词表 |
| 多问题处理 | 多个问题构造成多行，在 batch 中一次 forward | 通常逐问题生成；可复用共同 causal prefix KV |
| state 复用 | 可复用序列化和 tokenization；不能等价复用完整双向 hidden states | 共同前缀可以复用 KV cache |
| 概率处理 | 内置按问题类型、候选数分桶的温度 | token 概率通常不是业务类别校准概率 |

### 2.1 英文 checkpoint 的实际结构

源码中的完整模块名为 `DecisionModel`。英文 checkpoint 约 421M 参数，其 Encoder 关键配置为：

| 配置 | 值 |
|---|---:|
| Encoder | `answerdotai/ModernBERT-large` |
| Encoder 层数 | 28 |
| hidden size | 1024 |
| attention heads | 16 |
| FFN intermediate size | 2624 |
| attention 模式 | 每 3 层一个 full attention，其余为 sliding attention |
| local attention window | 128 |
| checkpoint `max_len` | 512 |
| checkpoint `head_max_len` | 192 |

Encoder 后接一个 Laya 专用决策头：

```text
Encoder hidden states [B, L, 1024]
        + 3 类 question-type embedding
        ↓
2 层 TransformerEncoder
  - hidden = 1024
  - heads = 16
  - FFN = 4096
        ↓
gather 每个 [MASK] marker 的 hidden state
        ↓
LayerNorm → Linear(1024,1024) → GELU → Linear(1024,1)
        ↓
每个候选一个 logit
```

三种 question type 为：

- `choice`：离散候选中选择一项。
- `score`：对有序等级输出分布，并计算期望等级。
- `noul`：二元命题，输出 `P(true)`。

模型还有一个辅助 `act_head`，输入 `[CLS]` 表示及 top-1、margin、entropy、候选数等分布特征，输出 action logits。它不是三类 typed decision 的主分类路径，业务门控也不能用它替代经过验证的候选概率。

### 2.2 为什么候选 marker 是核心

每个候选前放一个 `[MASK]` marker。模型最终读取 marker hidden state，而不是读取固定分类 token 或生成标签词。

这带来两个性质：

1. 候选数可以随请求变化，不需要把最后一层固定为某个业务类别数。
2. 候选描述必须真实、完整且彼此可区分；候选被截成相同 token span 时，模型已经失去区分它们的输入信息。

默认已发布 checkpoint 使用 `sequential` option layout：候选依次排列并互相可见。`parallel` layout 会让每个候选从相同 position 开始，并阻止候选之间互相 attention，以减弱顺序偏差；但它改变了输入分布，需要用该 layout 重新训练，不能只在推理时切换。

## 3. 输入与输出规范

### 3.1 公共输入格式

推理接口接收：

```python
state: str | dict | list
questions: dict[question_id, question_definition]
```

一个完整示例：

```python
state = {
    "message": "My transfer has not reached the recipient."
}

questions = {
    "intent": {
        "type": "choice",
        "instructions": "Which banking intent does `message` express?",
        "criteria": {
            "transfer_not_received_by_recipient": "Recipient has not received a transfer",
            "pending_transfer": "Transfer is still pending",
            "failed_transfer": "Transfer failed"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["low", "medium", "high"]
    },
    "needs_manual_review": {
        "type": "noul",
        "instructions": "Does this request need manual review?"
    }
}
```

三种问题的约束如下：

| 类型 | `criteria` 形式 | 输出语义 |
|---|---|---|
| `choice` | `label -> description` 字典，或 label 列表 | 候选概率和 argmax label |
| `score` | 按等级从低到高排列的列表 | 等级概率与期望等级 |
| `noul` | 可省略；或仅含 `true` / `false` 描述的字典 | `P(true)` |

构造请求时必须满足：

- `question_id` 唯一且非空。
- 每个问题都必须提供非空 `instructions`。
- `choice` label 必须唯一，并使用字符串、数字或布尔标量；label 同时是输出 key。
- `score` 的顺序就是等级顺序，index 0 为最低等级。
- `noul` 始终是 false / true 两个语义位置；自定义显示文本使用 `labels`，不能改变内部语义顺序。
- 同一批 `states` 使用同一套 `questions`；返回结果按输入顺序对齐。

### 3.2 内部序列格式

每个 state-question 对被构造成一条序列：

```text
[CLS]
question_type + instructions
[SEP]
[MASK] option_0
[MASK] option_1
...
[SEP]
serialized_state
[SEP]
```

`dict` 或 `list` 类型的 state 先序列化为 JSON。多个问题会得到多条序列；它们可以进入同一个 batch，但 Encoder 并没有只计算一次共享 state。

两个长度参数共同决定输入：

- `max_len`：问题、候选和 state 的总 token 上限。
- `head_max_len`：问题与全部候选的 token 预算。

构造器先分配 question head，再把剩余位置给 state：

```text
state_room = max_len - actual_head_length - 1
```

每个候选描述首先最多保留 48 token。若全部候选超过 `head_max_len`，系统再按候选数平均压缩，并保证每项至少 4 token。候选多时，两个不同描述可能因此变成相同 token span；训练阶段会将其标记为 `options_collapsed`，推理阶段也应检查 `options_distinct`。

### 3.3 输出格式

每个问题都返回 typed answer。核心字段如下：

```json
{
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "transfer_not_received_by_recipient",
      "probabilities": {
        "transfer_not_received_by_recipient": 0.71,
        "pending_transfer": 0.24,
        "failed_transfer": 0.05
      },
      "confidence": 0.45,
      "answer_confidence": 0.71
    },
    "urgency": {
      "type": "score",
      "score": 1.62,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.08, "1": 0.22, "2": 0.70},
      "answer_confidence": 0.70
    },
    "needs_manual_review": {
      "type": "noul",
      "noul": 0.83,
      "answer_confidence": 0.83
    }
  }
}
```

上面的数值只是格式示例。字段含义必须区分：

- `probabilities`：温度缩放后的候选分布。
- `choice`：`choice` 分布的 argmax。
- `score`：等级 index 的概率期望，不是任意连续回归值。
- `noul`：`P(true)`。
- `answer_confidence`：所有问题类型统一使用的 `max(probabilities)`。
- `confidence`：`choice` / `score` 默认是归一化熵集中度，与 `answer_confidence` 不是同一量；跨类型门控应优先使用经过独立验证的 `answer_confidence`。

## 4. 核心超参数

### 4.1 架构与输入参数

| 参数 | 默认 / 本 checkpoint | 作用 | 配置原则 |
|---|---:|---|---|
| `encoder` | ModernBERT-large | 决定主干表示能力 | 更换主干等于更换模型，不是普通推理参数 |
| `head_layers` | 2 | Encoder 后的 typed-decision Transformer 层数 | 必须与 checkpoint 权重一致 |
| `max_len` | 512 | 完整序列总预算 | 增大只避免截断，不保证模型学过长上下文 |
| `head_max_len` | 192 | instructions + options 预算 | 大候选集应重点检查 option collapse |
| `option_layout` | `sequential` | 候选的位置和 attention 关系 | `parallel` 必须训练后再用于推理 |
| `temperature` | 每个问题类型一个 | 校准 logits | 必须绑定 checkpoint、任务和数据分布 |
| `temperature_by_options` | 按类型和候选数分桶 | 修正不同候选数的概率尺度 | bucket 优先于类型级温度 |
| `amp_dtype` | `bf16` | CUDA autocast 目标精度 | 影响速度、显存和概率数值，需要复测 |

当前运行时将温度限制在 `[0.5, 5.0]`。英文 checkpoint 的原始
`choice:11+` 温度为 `0.10058`，运行时会钳制到 `0.5` 并报告警告；钳制只限制过度锐化，不能把该 bucket 自动变成可靠校准概率。

### 4.2 训练参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `epochs` | 4 | 完整遍历训练 items 的次数 |
| `micro_batch` | 8 | 每次 forward / backward 的 item 数 |
| `grad_accum` | 8 | 累积多少个 micro-batch 再更新 |
| `encoder_lr` | `2.5e-5` | Encoder 参数学习率 |
| `head_lr` | `1e-4` | decision head 学习率 |
| `min_lr` | `1e-6` | cosine schedule 末端学习率 |
| `weight_decay` | `0.01` | AdamW 权重衰减 |
| `grad_clip` | `1.0` | 全局梯度范数上限 |
| `loss` | `rlcd` | `rlcd` 或 `soft-ce` |
| `freeze_encoder` | `False` | 只训练 decision head |
| `shuffle_options` | 空 | 指定哪些问题类型每 epoch 随机打乱候选 |
| `label_smoothing` | `0.0` | 将 hard label 的部分概率质量分配给其他候选 |
| `calib_frac` / `calib_max` | `0.1` / `400` | 训练前保留的校准集比例与上限 |
| `target_error` | `0.10` | 拟合 abstention threshold 的目标错误率 |
| `min_abstain_n` | `10` | 拟合每个拒答 bucket 所需的最少校准样本 |
| `eval_data` | 无 | 训练前后使用的独立评价集 |

有效 batch 上限为：

\[
B_{\text{effective}}
= B_{\text{micro}} \times N_{\text{accum}} \times N_{\text{devices}}.
\]

单设备默认是 `8 × 8 = 64`。每 epoch 最后一个不完整累积窗口仍执行一次 optimizer update，因此准确更新数为：

\[
N_{\text{updates}}
= E \times
\left\lceil
\frac{\left\lceil N_{\text{train}}/B_{\text{micro}}\right\rceil}
{N_{\text{accum}}}
\right\rceil .
\]

### 4.3 RLCD 参数

只有 `loss="rlcd"` 时，下列参数参与训练：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `rl_samples` | 4 | 每条 logit 采样的探索分布数 |
| `sigma_start` | 0.4 | 第一轮探索噪声 |
| `sigma_end` | 0.1 | 最后一轮探索噪声 |
| `w_sph` | 0.75 | spherical score 权重 |
| `w_rps` | 1.0 | 有序 score 问题的 RPS 权重 |

噪声 `sigma` 按 epoch 从 0.4 线性退火到 0.1。增加采样数会提高 reward 估计成本；这些参数影响的是 RLCD 项，不改变输入 schema。

## 5. 完整训练流程

Laya 的正式 fine-tune 不是“读 CSV 后直接跑交叉熵”，而是一个包含数据契约、优化、校准和独立评价的闭环：

```text
原始 CSV / JSONL
        ↓
校验 question schema 和 target
        ↓
构造 state-question items
        ↓
训练前切出 calibration items
        ↓
训练 Encoder + decision head
        ↓
在 calibration items 上拟合温度与拒答阈值
        ↓
在独立 eval_data 上比较训练前 / 训练后
        ↓
保存权重、tokenizer、config、questions、report
```

### 5.1 数据与目标

支持两种 JSONL 监督形式：

```json
{"state": "...", "questions": {...}, "expected": {"intent": "billing"}}
```

`expected` 给单个正确答案，可转为 one-hot 或 label-smoothed target。

```json
{"state": "...", "questions": {...},
 "gold": {"intent": {"probabilities": {"billing": 0.9, "technical": 0.1}}}}
```

`gold` 直接提供 teacher 分布，用于 soft-target 训练。每个 target 必须与候选一一对应、非负且归一化。

数据预处理会拒绝：

- 空 state 或空 label。
- 非法 question schema。
- target 与候选数不一致。
- 两个候选在预算内变成相同 token span。
- 部分候选 marker 超出 `max_len`。

### 5.2 训练目标

`soft-ce` 对每行真实候选计算分布交叉熵：

\[
\mathcal L_{\text{soft-CE}}
=-\frac{1}{N}\sum_{n=1}^{N}
\sum_{k=1}^{K_n} q_{nk}\log p_{nk}.
\]

其中 \(q\) 可以是 hard label 的 one-hot，也可以是 teacher 概率。

`rlcd` 在 soft-CE 之外增加一个 GRPO 风格的 logit-space policy-gradient 项：

1. 对 detached logits 加零均值 Gaussian noise。
2. 每条输入生成 `rl_samples` 个候选分布。
3. 用 proper scoring reward 评价分布：log score + spherical score；`score` 类型额外减 ranked probability score。
4. 对 reward 做样本内中心化和标准化，得到 advantage。
5. 用 advantage 加权噪声样本的 Gaussian log-density。
6. 最终损失为 policy-gradient 项加全权重 soft-CE。

简写为：

\[
\mathcal L_{\text{RLCD}}
=-\mathbb E[A\log p(\epsilon)]
+\mathcal L_{\text{soft-CE}}.
\]

RLCD 的作用是直接优化概率分布评分，但它不是所有任务都优于 soft-CE。上游已有任务中，soft-CE 得到过同等或更高准确率，因此两者应作为实验臂比较，而不是默认把 RL 项视为必需。

### 5.3 参数更新

训练使用两组 AdamW 参数：

- Encoder：`encoder_lr`。
- type embedding、decision head、scorer 等非 Encoder 参数：`head_lr`。

学习率采用 cosine annealing 到 `min_lr`。支持梯度累积、全局 norm clipping、CUDA mixed precision 和 Encoder / head gradient checkpointing。

若 `freeze_encoder=True`，Encoder 设为 eval 并 detach hidden states，只训练后部决策参数。该模式降低显存和计算成本，但表达能力受固定 Encoder 限制。

若启用 `shuffle_options`，每 epoch 重排候选，并同步重排 target，用于降低顺序先验。它与 `parallel` layout 不同：前者是训练增强，后者改变 position 和 attention mask。

### 5.4 校准、拒答与评价

校准样本在训练前切出，不能先用于更新权重。训练后按 question type 和候选数 bucket 拟合温度：

```text
choice:2
choice:3-5
choice:6-10
choice:11+
score:...
noul:2
```

每个温度通过最小化 target NLL 拟合。样本不足的 bucket 回退到类型级温度；温度文件与 checkpoint、schema、候选数及推理精度绑定。

同一 calibration slice 上的训练后指标只说明校准拟合，不能作为独立泛化结果。正式验收应提供无训练 / 校准重叠的 `eval_data`，并比较 accuracy、NLL、Brier、ECE、拒答覆盖和业务错误成本。

最终 checkpoint 保存：

```text
model.safetensors
encoder/config.json
tokenizer/
rl_agent_config.json
questions.json
train_report.json
checkpoint_latest/
```

其中 `max_len`、`head_max_len`、`option_layout` 和校准参数都写入 config，确保训练和推理使用相同输入契约。

## 6. 本项目解决的问题与实现方式

### 6.1 问题定义

本项目要把 Laya 用于 Banking77 的 77 类银行意图路由。直接使用英文 checkpoint 有两个实际问题：

1. 默认 `head_max_len=192` 无法充分表示 77 个候选，77 类压成 73 种不同 token span。
2. 即使扩大候选预算，zero-shot Laya 在去重测试集上仍只有 57.00% 准确率，说明问题不只来自输入截断。

前两轮 AG News 校准还说明：温度或仿射校准可以改善概率尺度，但不能替代领域分类能力。项目因此从“只调输出概率”转向“训练领域监督头”。

### 6.2 数据隔离

Banking77 官方源经精确规范化文本去重后划分为：

| split | 数量 | 用途 |
|---|---:|---|
| fit | 9,229 | 拟合词法和 dense 分类头 |
| validation | 770 | 选择正则强度和融合权重 |
| test | 3,072 | 只做最终评价 |

测试在 `selection.json` 落盘后才执行。精确去重去除了训练 4 条、测试 8 条，但不等于完成语义去重或预训练污染审计。

### 6.3 两条监督路径

词法路径：

```text
word TF-IDF (1,2)-gram，最多 50k
 + char_wb TF-IDF (3,5)-gram，最多 50k
 → 拼接并 L2 normalize
 → 77 类 Logistic Regression
```

共同配置为 `min_df=2`、`sublinear_tf=True`；验证候选
`C ∈ {1,4,16}`，最终选择 `C=16`。

Dense 路径：

```text
消息
→ 冻结 Laya Encoder
→ mean-pooled embedding，max_length=128
→ L2 normalize
→ 77 类 Logistic Regression
```

验证候选 `C ∈ {1,10,100}`，最终选择 `C=100`。这条路径没有更新 Laya Encoder，也没有更新原 marker-based typed decision head。

验证集最终选择概率融合：

\[
p_{\text{selected}}(y\mid x)
=0.75p_{\text{lexical}}(y\mid x)
+0.25p_{\text{dense}}(y\mid x).
\]

运行时恢复 TF-IDF 词表、IDF、两个 `.npz` 分类头和固定融合权重；新请求不会在线更新模型。

### 6.4 为什么还测原 Laya 与 top-k

项目保留以下对照，用于定位瓶颈：

- `laya_default`：默认 512 / 192 token 预算。
- `laya_wide`：扩大为 1024 / 768，隔离候选预算问题。
- `laya_top10`：监督分类器先召回十类，再让原 Laya 重排。
- `fusion_0.25` / `fusion_1`：保留分类器 prior，再叠加不同强度的 Laya 重排分数。

top-10 的测试召回率为 99.71%，纯 Laya 重排却只有 67.25% 准确率。这说明大部分正确类别已经进入候选，但原 typed-decision 路径仍会重排错误；因此当前选定方案直接返回监督融合概率，不再经过原 Laya scorer。

### 6.5 结果与边界

| 方法 | Accuracy | NLL ↓ |
|---|---:|---:|
| 原 Laya 默认预算 | 45.08% | 4.3379 |
| 原 Laya 扩大预算 | 57.00% | 3.2423 |
| 词法监督头 | 90.82% | 0.4145 |
| 冻结 Encoder 分类头 | 82.42% | 0.7610 |
| 75% 词法 + 25% dense | **91.67%** | 0.4438 |
| top-10 后由 Laya 纯重排 | 67.25% | 1.3474 |

选定融合相对词法基线为 44 胜、18 负，净多答对 26 / 3,072 条，即 +0.85 个百分点；配对 bootstrap 95% 区间为 `[+0.36,+1.33]` 个百分点。

准确率改善没有同时带来概率质量改善：融合的 NLL、Brier 和 ECE 均比词法基线差。因此当前产物适合“固定 77 类、准确率优先”的离线路由，不应未经重新校准就把概率当成生产拒答阈值。

必须明确，本项目第三轮做的是：

> 冻结 Laya Encoder 的领域表征复用 + 词法监督头 + 验证选择的概率融合。

它不是完整 Laya fine-tune，没有更新原 Encoder、type embedding、两层 decision head 或 marker scorer，也没有建立跨语言、动态 schema、Jev 对照或 SOTA 结论。

## 7. 最终认知框架

Laya 的技术逻辑可以压缩为三层：

1. **通用决策模型层**：双向 Encoder 将 state、问题和候选联合编码；typed decision head 从候选 marker 直接输出概率。
2. **领域训练层**：用 hard label 或 teacher 分布训练 Encoder / head，随后独立拟合校准和拒答参数。
3. **本项目适配层**：发现 77 类候选预算和 zero-shot 能力不足后，不修改原 Laya 决策权重，而是训练词法与冻结 Encoder 分类头，验证选择 75% / 25% 概率融合，形成可加载的固定 Banking77 路由器。

三层不能混称：官方 Laya 架构定义模型如何决策；官方 fine-tune 定义如何更新完整模型；本项目最终方案定义如何在固定领域复用 Laya 表征并获得更高准确率。

## 8. 实现与证据入口

- Laya 架构：
  [`common.py`](https://github.com/NandhaKishorM/laya/blob/7a77439b0923d6efe00052d50a7f787840bfce98/laya/common.py)
- 输入输出：
  [`agent.py`](https://github.com/NandhaKishorM/laya/blob/7a77439b0923d6efe00052d50a7f787840bfce98/laya/agent.py)
- 完整 fine-tune：
  [`train.py`](https://github.com/NandhaKishorM/laya/blob/7a77439b0923d6efe00052d50a7f787840bfce98/laya/train.py)
- 温度校准：
  [`calibrate.py`](https://github.com/NandhaKishorM/laya/blob/7a77439b0923d6efe00052d50a7f787840bfce98/laya/calibrate.py)
- 本项目训练与对照：
  [`decision_study/banking77.py`](decision_study/banking77.py)
- 本项目运行时：
  [`decision_study/banking77_runtime.py`](decision_study/banking77_runtime.py)
- 冻结选择与最终结果：
  [`selection.json`](decision_study/results_round3/selection.json)、
  [`results.json`](decision_study/results_round3/results.json)
- 复现入口：
  [`BANKING77_REPRODUCE.md`](decision_study/BANKING77_REPRODUCE.md)
