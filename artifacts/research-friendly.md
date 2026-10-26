# 决策模型训练 Infra：从输入、损失到多设备协作

这是[2026-10-19 完整调研](research.md)的阅读版本。原稿中的资源数量与“尚未训练”是当时状态，当前实验结果见[入口](README.md)。这里保留原方案的技术逻辑，不把设计写成已完成成果。

Infra 是让模型训练 / 推理正确、高效运行的基础设施。它解决张量布局、数据组织、内存、调度、通信和恢复；模型是否理解业务还需要质量评测。

## 决策模型与生成模型的训练差别

决策模型直接输出允许选项的概率，通常不需要生成一段答案文字。输入仍有文本，但损失的分母可以是“每个判断”，不一定是“每个 token”。

| 读出方式 | 输出位置 | Infra 需要保持的对应关系 |
|---|---|---|
| Laya | Encoder＋决策 head，从选项 marker 读分数 | 每题序列、有效候选、marker 与类别 |
| causal LLM 首 token | 第一个回答位置的候选标签分数 | 最后有效 read position、标签 token、候选概率与全词表概率 |
| 固定 code head | 专门分类头的类别 code | code 与请求 option ID |
| joint schema head | 共享状态表征上的多问题输出 | 问题 offsets、样本隔离、各问题权重 |

外部接口相似，内部未必共享同一次主干编码。Laya 为每题构造序列，批量 forward 不等于 state 只编码一次。优化时要先核对真实路径。

若有真实标签，交叉熵 \(L=-\log p(y)\) 是直接基线；若教师提供分布 \(q_k\)，可用 \(L=-\sum_k q_k\log p_k\) 做蒸馏。教师也可能错误，质量测试须独立。RLCD 的 logits 探索噪声不要求每个探索样本重跑整个 Transformer；额外 RL 项仍需与 soft-CE 对照。

## 为什么全局损失分母很重要

一个工单可以附五个问题，另一个只有一个。按问题平均会给前者五倍总权重；按案例平均先平均同一工单的问题，再平均工单；按 token 平均是另一种目标。原方案推荐决策任务按案例平均，须在 manifest 中声明。

设每题权重为 \(a_j\)，一次全局更新全部有效权重之和为 \(A\)。若通信后端对 \(W\) 个 rank 的梯度取平均，每 rank 累积的 backward 标量应为：

\[
L_{\mathrm{rank}}=\frac{W}{A}\sum_{j\in\mathrm{local}}a_j\ell_j.
\]

分母 \(A\) 必须覆盖所有 rank 和全部 microbatch。因为后端再除以 \(W\)，最终才得到所要的全局加权平均。若后端是 sum reduction 或混合分片，其 divide factor 不同，公式必须按实际实现调整。直接平均各 rank 的局部 mean 在负载不等时会改目标。

低频局部训练的岛内目标使用岛内 world size；外部模型平均属于另一套规则，不能冒充等价 global batch。

## 输入正确性先于算子加速

有效选项 mask 必须排除 padding；打乱选项时 target 和输出 ID 一起重排；截断要检查证据和完整类别描述；packing 要隔离样本 attention；随机数、dropout、选项顺序与数据游标需要一起恢复。未参与 loss 的参数应显式处理，不能假设全模型都得到梯度。

Laya parallel layout 和 shuffle 训练属于模型 / 训练增强，单纯换后端不会自动消除位置偏差。量化或融合可能保持 argmax 却改变概率，验收必须看分布和阈值，不能只检查类别。

## 先算状态内存，再讨论几张卡

按 FP32 参数、梯度及 Adam 两份状态，基础状态约 \(16P\) 字节：

| 原 Qwen3 参考参数 | 总基础状态 GiB | 4 路分片 GiB/rank | 8 路 | 16 路 |
|---|---:|---:|---:|---:|
| 8.2B | 122.2 | 30.5 | 15.3 | 7.6 |
| 14.8B | 220.5 | 55.1 | 27.6 | 13.8 |
| 32.8B | 488.8 | 122.2 | 61.1 | 30.5 |

这是规划，不是显存测量；未含激活、临时完整参数、额外实现副本、通信缓冲与 CPU 快照。改成新决策 head 后须重新统计实际 trainable 参数。即使分片后足够，分片前初始化仍可能耗尽显存（OOM）。

## 设备优化按真实热点排序

H20 和 NPU 都应先锁定版本，跑完整 forward、backward、optimizer 和保存 / 加载，再 profile。推理支持某个 attention 不代表训练 backward 可用。

| 热点 | 可尝试的优化 | 同时验证 |
|---|---|---|
| attention / norm / activation | 支持 backward 的融合、重计算 | mask、dQ/dK/dV、残差、dtype |
| 动态长度与布局 | 有限长度 bucket、减少 transpose / contiguous | padding 浪费、编译次数、样本隔离 |
| host 下发 | 预处理预取，减少 `.item()` 同步 | CPU 空洞、输入等待、端到端时间 |
| optimizer / 分片 | foreach / 融合更新，FSDP2 | 更新数值、状态布局、checkpoint |
| 通信 | 亲和性、布局与调度 | 实际物理机、payload、尾延迟 |

NPU 使用选定 TorchNPU / CANN 配套；H20 可使用 CUDA 生态的融合与图捕获，但都不能仅凭文档能力声明收益。FP8 还需硬件 / kernel 与训练精度验收。

## 低带宽下，先减少跨机关键路径

全局 FSDP 频繁收集层参数；HSDP 把这件事留在机内，机间仍同步梯度。TP、CP、EP 等每层或 token 调度通信也会受慢链路影响，不应只看总字节。

保持训练目标的优化包括布局、梯度累积的正确归一化、bucket、重叠和预取；改变轨迹的包括 Local SGD、DiLoCo、压缩。后者需要新的质量曲线。

理想传输 \(T=8S/B\) 只是下限模型，实际时间还含拷贝、校验、聚合和等待。是否有收益，应看质量达标时间、实际监督吞吐、暴露通信、内存、费用与恢复，而非只看 GPU 利用率。

相关基础来源：[Laya 训练](https://nandhakishorm.github.io/laya/finetune/)、[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)、[FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[TorchNPU](https://github.com/Ascend/pytorch)、[DiLoCo](https://arxiv.org/abs/2311.08105)、[TorchTitan](https://github.com/pytorch/torchtitan)。原稿的完整 S1–S38 来源与硬件优化表保留，外部或其他项目探针不计作本项目实测。
