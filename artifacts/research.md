# 决策模型训练 Infra：910B4、H20 与低带宽分布式训练调研

资料日期：2026-10-19。范围包括 Laya、Clef、Decision 3.0、Liquid d1，以及使用 Qwen 8B–32B 主干构建的同类决策模型。

初始调研完成资料研究、平台只读查询与方案设计。后续按用户要求申请 H20/910B4 各 1 卡，完成不同 HL 节点的 TCP、CPU Gloo 和设备写回探针，见 [实测报告](communication-results/report.md)；尚未进行真实模型训练。其他实验目录只读，原有归档不计作本项目实测。

## 1. 技术判断

推荐项目是：**面向决策模型后训练的、具有 CUDA/NPU 后端的分布式训练系统，重点研究节点内分片与低带宽跨节点同步，在相同质量约束下比较训练时间。**

核心训练负载选择 Qwen3-8B 全参数决策后训练，扩展到 Qwen3-14B、Qwen3-32B。Laya 322M/421M 用于输入、loss、mask、梯度及恢复逻辑的低成本验证。公开的 Clef-Flash 9B 和 Clef 27B 说明，直接输出选项概率的模型并不局限于小 Encoder。[S1][S2][S3][S4][S5]

项目应形成三种可分开验收的成果：

1. **算子与内存层**：优化 forward、backward、optimizer 和激活保存。
2. **分布式语义层**：在不同长度、不同问题数、不同 rank 负载下，保持声明的训练目标和更新正确。
3. **低通信优化层**：比较同步 HSDP、Local SGD、DiLoCo 与压缩通信，验证达到同等质量的墙钟时间。

DiLoCo 和压缩改变数值或优化轨迹，应单独建立质量证据；不能只把通信次数减少当成训练成功。[S6][S7]

## 2. 资源与已有证据

### 2.1 平台当前资源

只读调用：

```text
control-plane=cn
resource groups get
sid=2079
options.include_resources=true
```

返回组名 `search_douyin_life_nlp`；成功码 0；request_id：
`202610190105106F4C48AF81C305A8505B`。

| 集群 | 硬件 | total | used | total−used，仅算术差值 |
|---|---|---:|---:|---:|
| cloudnative-lf，sid 22 | NVIDIA-H20 | 136 | 80 | 56 |
| cloudnative-hl，sid 20 | NVIDIA-H20 | 32 | 19 | 13 |
| cloudnative-hl，sid 20 | Ascend910b4-64G | 16 | 2 | 14 |

这是当前控制面、当前账号的聚合资源快照，不证明一定可同时分配、节点拓扑、排队 ETA、抢占规则或跨集群连接。接口返回的 `allocatable` 在这些行与 total 相同，因此不能将该字段直接解释为空闲卡数。公平/优先队列及小任务优先来自用户提供的背景，本轮接口未核实这些调度政策。

这次仅完成 Resource 当前态/候选查询；后续训练提交应走 Merlin Job 的流程。

### 2.2 既有工作区

另一个 AI 正在使用 `training_infra_lab/` 与 `lowband_training/`。本项目只读参考，主要产出放在 `laya/`。

已读到的既有内容：

- `training_infra_lab/README.md` 记录 Qwen2.5-1.5B-Instruct/no_robots 全参数 SFT，以及双卡 H20、910B4 的小模型 BF16/collective/FSDP2 探针结果。
- `loss.py` 有监督位置筛选与分块投影重计算。
- `train.py` 有 FSDP2、全局监督 token 归一化和吞吐统计；文件由另一任务持续修改，本轮不将其某一快照当作最终实现。
- `jobs/h20-multinode-probe/inspection.txt` 的四个 rank 报告同一个 hostname `compute-node-3a931df8`。

最后一项只能证明多 worker 进程组运行，不能证明跨物理机通信。正式跨机实验必须同时记录平台 node identity、hostname、设备 UUID/PCI 地址和通信路径。不同 pod、不同 rank、不同容器 IP 均不足以单独证明不同物理节点。

平台知识库中的 Arnold FAQ 也记录过多个 worker 调度到同一机器的情况。`NCCL_HOSTID` 等配置即使解决建链，也不会把同机实验变成跨机实验。[S8]

## 3. 决策模型训练有哪些特殊工作

### 3.1 读出模式与损失

| 模型路线 | 训练读出 | 特殊 Infra 工作 |
|---|---|---|
| Laya | 双向 Encoder + 两层决策 head + marker scorer | 每题独立序列、可变选项 mask、局部/全局 attention |
| Causal LLM 首 token 决策 | 问题后最后有效 token 的 hidden state，读取答案标签 logits | 标签 token 验证、精确 read position、候选分布与全词表分布区分 |
| 答案 code classification | 专门训练的固定 code 输出头，再映射到请求候选 | code 与选项映射、校准温度、模型配置与服务读出一致 |
| Joint schema head | 共享 state 表示，读取多个问题并联合输出候选 logits | 多问题 offsets、head attention、跨样本隔离、问题权重 |

这些接口可以相似，内部并非同一种训练方法。Laya 公开代码逐题构造序列；Clef 是共享主干加 joint schema head；D3 文档说明每题各自 forward；d1 的公开文档强调共享 state 的一次调用，但对照其具体代码时仍须验证实现与训练目标。[S1][S2][S3][S9][S10]

核心监督目标可以是：

```text
L_hard = -log p(correct option)
L_soft = -Σ q_teacher(option) log p_student(option)
```

硬标签 CE、soft-target 蒸馏可作为先行基线；RLCD 是单独实验臂。Laya 文档与代码已经支持 soft-CE，并报告在所测领域任务上额外 RL 项没有带来一致收益。[S1][S11]

RLCD 的多个探索样本在 logits 上添加噪声，不代表每个探索样本必须重新运行整个 Transformer。把离线概率训练误做成在线长文本 rollout，会引入不必要的系统复杂度。

### 3.2 不能混淆三种归一化

必须在实验 manifest 中声明样本权重：

- 按决策平均：每道题权重相同，问题多的 state 总权重更大。
- 按案例平均：先对一个 state 的问题平均，再按案例平均。
- 按 token 平均：普通语言建模/SFT 的监督 token 目标。

推荐主任务采用**按案例平均**，避免一个 state 附带更多评分题而自动获得更大权重。设每题权重为 \(a_j\)，每个 optimizer step 的全局有效权重为 \(A\)，在梯度平均语义下：

```text
本 rank 每个 microbatch 的 backward 标量
  = world_size × Σ_local(a_j × loss_j) / A
```

累积窗口的分母必须覆盖所有 rank 和全部 microbatch。不能直接平均局部 mean loss；不能把 global token count 当作决策样本权重。自定义 sum-reduction hook、分片及混合并行还需按实际 divide factor 修正。[S12][S13]

在低频同步的计算岛内部使用 island world_size 与 island 目标；外部权重及数据配比属于 DiLoCo 优化规则，不可与同步全局大 batch 等价性混为一谈。

### 3.3 输入与随机性

决策任务需要处理：

1. **可变选项数**：padding 选项不得参与 softmax、target 或 loss。
2. **选项乱序**：输入 slots 变化时，target 必须同步重排；服务输出恢复原 option ID。
3. **截断可见性**：检查 state 证据、问题、选项描述以及 read markers 是否被截断。
4. **多问题独立性**：联合 head 或共享 causal prefix 不能让某题读取另一题的 gold answer。
5. **跨样本隔离**：packing 不能仅拼接 token；必须保持 attention 隔离和位置编号规则。
6. **随机数**：dropout、选项顺序、RL 噪声、负采样必须可恢复。
7. **未使用参数**：如 Laya `act_head` 不参与所选 loss，显式冻结或处理其 reducer 语义，不能假设全模型每个参数都获得梯度。

Laya 的 parallel option layout 和选项 shuffle 属于模型/训练增强；它们不同于单纯的后端优化，实验中应分开。[S1][S11]

### 3.4 后处理和质量

训练产物除权重外，还包括 tokenizer/schema/读出配置、校准数据版本、温度或其他 calibration 参数。报告 Accuracy、Macro-F1、NLL、Brier、ECE、Score MAE、按语言/选项数分组的结果，以及固定误差约束下的自动接受覆盖率。

不同 dtype 或优化 kernel 可改变概率但不改变 argmax；仅检查预测类别一致不足以验证决策训练 Infra。校准在独立数据上拟合，测试集只作最终评估；多个问题、同一 prompt 的多个回答、平行翻译按来源案例分组切分。

## 4. 910B4：训练类型与优化顺序

### 4.1 推荐负载

910B4 可以承担真实混合精度训练，不能仅凭某些营销资料把它归为“只能推理”。现有平台归档已有小模型反向及 FSDP2 探针；官方 TorchNPU 提供训练、collective、DTensor 和图模式能力。但这不证明本项目的完整 Qwen 或 Clef 训练已经通过。[S14]

推荐：

- 8B/14B dense Transformer 的全参数决策后训练、分类/评分模型训练。
- 中长序列文档判断，多问题评分，选项较多的工具选择。
- 后续多模态决策 head；先冻结视觉塔，再逐步比较解冻训练。
- Laya/mmBERT 用于 NPU attention/mask/backward 的正确性与算子验证。

LoRA 可作低通信业务基线，但主 Infra 项目采用全参数训练才能充分暴露参数、梯度、optimizer 分片和恢复问题。

### 4.2 环境闭环

锁定驱动、固件、CANN、TorchNPU package version、PyTorch、Python、Transformers、通信库和主干 revision。TorchNPU 产品版本与 pip package version 不是同一编号；官方兼容矩阵明确列出配套关系。[S15]

已有环境日志出现 PyTorch 2.7.1 / torch-npu 2.7.1，不应仅凭版本号猜测完整 CANN 配套，也不应为了“最新版”覆盖共享环境。后续每次实验使用隔离镜像或锁定环境，记录真实导入路径和版本。

### 4.3 优化动作

| 层面 | 优先动作 | 必须验证 |
|---|---|---|
| Attention | 使用具有训练 backward 的融合 attention | dQ/dK/dV、causal/非 causal、GQA、padding/window mask |
| Tensor layout | 减少 transpose、contiguous、格式转换；复用布局 | 实际复制与转换耗时、前后梯度 |
| Vector/Cube 利用 | 融合 RoPE、RMSNorm/LayerNorm、SwiGLU/GEGLU 等热点 | backward、recompute、残差与 dtype |
| Shape | 有限长度 bucket；必要时 varlen/packing | 编译次数、padding、样本隔离 |
| 内存 | BF16 compute、FP32敏感归约、选择性重计算、分片 | allocated/reserved 峰值、临时全参数 |
| Host | 避免逐层 `.item()` / CPU 同步；预处理与预取 | host 下发空洞、输入等待 |
| Optimizer | 对照可用融合/foreach 实现 | update 等价、optimizer 状态布局与 checkpoint |
| 通信 | HCCL、设备与 NIC 亲和性，分片/通信调度 | 多物理机大 payload、尾延迟 |
| 图模式 | 热点图捕获、固定 bucket | 动态形状、graph break、backward 可用性 |

NPU 的 `npu_incre_flash_attention` 等增量推理算子不能代替训练 backward。拟采用训练向 `npu_fusion_attention` / FlashAttentionScore 路径，具体支持矩阵由已锁定版本和完整正反向探针确认。本轮官方页面正文读取不完整，因此没有承诺某个 sparse_mode、head_dim、GQA 或 varlen 配置已支持。[S16]

ModernBERT 的 local-global 交替 attention 需要保留原语义，不能用普通 causal mask 替换。Qwen3 dense 主干先行，可以减少 NPU hybrid linear-attention 适配的不确定性。Clef/D3 基于更新 Qwen 主干的扩展放在后续阶段。

先用 `torch_npu.profiler` 与 msprof-analyze 判断热点，再选一个真实占比高的算子或调度点做改动。不能把底层某个 kernel 提升直接当作训练全链路提升。[S17]

### 4.4 大模型初始化

不要先把完整 FP32 32B 模型 `.to(npu)` 再分片。采用 meta 初始化、分片后加载/分布权重，检查共享 embedding、输出权重的 owner，以及非持久 RoPE buffer 的初始化。

优化器在分片之后构建。NPU 上 DTensor、global grad norm、DCP 全链路需要实测，不可把小型 Linear 探针外推为完整 32B 能力。[S13][S14]

## 5. H20：训练类型与优化顺序

H20 为 Hopper 系列，可用于训练；其官方资料列出不同显存版本，不能将队列名 `NVIDIA-H20` 自动解释成某个固定容量。规划先按 96GB 条件设计，正式运行读取设备实际容量、compute capability、MIG/完整卡和互联。[S18]

公开资料普遍描述 H20 相比完整 H100 的计算能力受限，而显存与带宽较充裕。本轮未取得直接支撑特定板卡 TFLOPS 的厂商数据表，不使用这些数值计算 MFU 或训练时长承诺。

### 5.1 推荐负载

- 8B/14B/32B 全参数后训练；必要时节点内 FSDP2/HSDP。
- 大教师离线标注、概率蒸馏数据生成，可使用其推理生态优势。
- 中长文档、多问题训练；以有效输入 token、决策数和实际质量作为指标。
- 后续 FP8 主干线性层训练与 BF16 决策 head 的对照。

### 5.2 优化动作

1. **先使用 BF16 建立稳定基线**：敏感 loss/softmax/统计使用 FP32；锁定梯度归约精度。
2. **改善 GEMM 工作量**：利用显存增加合适 microbatch、减少 padding，不无限增大全局 batch。
3. **检查 attention backend**：确认融合路径实际生效，支持 causal/GQA/mask 和 backward。
4. **融合真实热点**：optimizer、norm、activation、RoPE、候选 logits gather，不复制另一目录的实验代码。
5. **减少无必要重计算**：比较相同全局 batch 下不重计算、选择性重计算和全重计算；显存富余时不要默认全重计算。
6. **拓扑感知分片**：只在高带宽计算岛内部安排 TP/FSDP 高频通信；跨岛用复制维度或低频同步。
7. **条件性使用 FP8**：检查设备和 Transformer Engine 的运行时支持。保存 amax/scaling 状态；比较主干 FP8 与 BF16 head，不将推理量化直接移植为训练量化。

FP8 的缩放、amax 同步、forward/backward 格式与模块约束有额外成本；数据格式和 dtype 转换本身不保证提速。MXFP8/NVFP4 是其他硬件代际相关能力，不能仅凭 TE 文档就声称 H20 支持。[S19]

## 6. 大模型内存规划

以下是**根据模型卡参数量推导的状态内存账本，不是设备实测或可运行保证**。[S4][S5][S20]

采用保守的 16 bytes/parameter：

```text
FP32 主参数 4 + FP32 梯度 4 + Adam m/v 8 = 16 bytes
```

BF16 compute 的参数物化副本、激活、通信 buffer、head、workspace、allocator 碎片和 checkpoint staging 另外计算。若用 BF16 存储/梯度，账本会变化；实验必须报告实际表示。

| 主干 | 模型卡参数量 | 全部状态 GiB | 分片 4 路 GiB/rank | 分片 8 路 GiB/rank | 分片 16 路 GiB/rank |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 8.2B | 122.2 | 30.5 | 15.3 | 7.6 |
| Qwen3-14B | 14.8B | 220.5 | 55.1 | 27.6 | 13.8 |
| Qwen3-32B | 32.8B | 488.8 | 122.2 | 61.1 | 30.5 |

结论：

- 8B 适合以 4/8 卡岛启动，在两种硬件上比较。
- 14B 在 64GB NPU 上的 4 路分片余量很小，建议从 8 卡岛规划。
- 32B 的 8 路状态已经约 61.1 GiB/rank，910B4-64G 不能按“8卡肯定装下”规划；保守采用 16 路高带宽岛，或另外比较 offload/低精度状态的代价。
- 若 16 张 NPU 构成一个完整 32B 岛，就没有剩余 NPU 组成第二个同规模副本；因此 NPU 的低频同步主实验应先选 8B/14B。
- 32B H20 在 8 路分片条件下可进入试验计划，但仍需实测激活/物化峰值；8卡总显存不是可用内存之和的简单保证。

若一个计算岛必须跨两台机器，其内部也需要足够快的互联。DiLoCo 只减少岛与岛之间的通信，不能修复岛内部无法承受的 TP/FSDP 通信。

## 7. 低带宽会怎样影响实际训练

### 7.1 性能与正确性分开

对于梯度、参数和 optimizer 完全同步且 collective 顺序正确的 DDP/FSDP，低带宽通常首先增加等待时间，不会天然让梯度变成过期梯度。

实际次生问题包括：

- 计算完成后等待 all-reduce；设备空转。
- FSDP 层前参数 all-gather 无法隐藏，前向/反向断续执行。
- 更多预取 buffer 与暂存参数增加峰值显存。
- 慢 rank 由数据长度、tokenization、存储等待或网络造成，所有 rank 同步等待。
- collective 顺序/shape 不一致、某 rank 跳过 step、进程退出会 hang 或 timeout。
- 全局 batch 随卡数扩大而更新次数下降，收敛变化被误认为系统收益。
- 数据读取、checkpoint 与梯度同步争用链路。
- 极低带宽下，初始化权重广播和恢复状态也可能成为主要耗时。

NCCL/HCCL kernel 时间不能直接当作纯网络时间：某 rank 的 collective 也可能在等待其他 rank 到达。应联合查看 bucket ready time、算子 timeline 与 NIC 数据。[S12][S21][S22]

### 7.2 通信量账本

对于 R 个 rank 的 ring all-reduce，完整梯度 payload 为 \(S=P b_g\)，每 rank 的发送量约：

```text
V_send ≈ 2(R−1)/R × P × gradient_bytes
```

这是理想化算法流量公式，不能直接用作整个集群或每台 NIC 的流量。接收也有对应流量，不能把 send+receive 与单方向带宽混用。[S23]

按 Qwen3-8B 的 8.2B 参数和 FP32 梯度推导，完整梯度 payload 为 32.8 GB；Decision head 只输出少量类别，仍不能减少全参数主干梯度的体积。

低带宽情况下，先判断：

```text
通信需求 = 更新频率 × 每次 payload + 高频小消息/控制同步
可隐藏通信 = 反向计算允许的 overlap 窗口
暴露等待 = 完成所有必要通信后的关键路径空洞
```

多网卡不一定增加实际可用带宽；不同 rank 常共享同一 NIC/PCIe/CPU 路径。HSDP 跨节点每个 rank 只同步一个 shard，但每节点全部 shard 的总 payload 仍相当于完整梯度，不能再凭空除一次节点卡数。

### 7.3 各种并行方式

| 方式 | 高频通信 | 低带宽影响 |
|---|---|---|
| DDP | 每次 optimizer update 的梯度 all-reduce | 可部分与 backward 重叠；小 microbatch 时尾部明显 |
| 全局 FSDP/ZeRO-3 | 层前参数 all-gather、反向再物化、梯度 reduce-scatter | 通信次数多；参数通信位于计算关键路径 |
| HSDP | 岛内分片通信 + 岛间梯度 shard 同步 | 将参数物化留在快网络，但仍每次更新跨岛同步 |
| TP | 每层 activation collective | 小消息频率高；优先放岛内 |
| PP | stage 间 activation/gradient send/recv | 带宽、延迟与流水线气泡一起影响 |
| CP/SP | attention/sequence 相关通信 | 长序列受益取决于通信是否可隐藏 |
| EP/MoE | token dispatch/combine all-to-all | 不均衡、路由偏斜与网络拥塞相互放大 |
| Local SGD/DiLoCo | 每 H 次本地更新交换参数 delta | 同步少，优化轨迹变；外部同步仍可能阻塞 |

主项目优先 dense 模型。MoE 大词表表、EP 或复杂多模态生成的训练会引入额外问题，适合作为后续负载扩展，而非首个基线。

### 7.4 保持目标的优化

优先次序：

1. 检查是否真实跨机，实际 transport 是否 RDMA、GPUDirect 是否生效。
2. 确认单机、同机多卡、跨机 collective 的 payload 曲线和尾延迟。
3. 增加合理 microbatch、减少 padding，缩小 rank 的工作量差。
4. DDP 梯度累积的前 A−1 个 microbatch 使用 `no_sync`，forward 也须在 context 内。
5. FSDP2 将梯度累积与参数 reshard 策略分别调优；HSDP 可以仅延后复制维度 all-reduce，保留岛内 reduce-scatter。
6. 合并合理 bucket、prefetch/overlap，并控制暂存内存。
7. 数据、checkpoint staging 分离，日志批量上报。

**梯度累积不是 Local SGD**：累积期间权重不更新，窗口末更新一次；Local SGD 每个本地 step 都更新权重。

**FSDP 的 no-sync 类机制不取消前向需要的参数 all-gather**。HSDP 的 `set_requires_all_reduce`、FSDP2 的 `set_requires_gradient_sync` 等接口语义应按已安装版本确认，不能套用其他版本的 API。[S13][S24]

### 7.5 改变优化轨迹的方案

DiLoCo：

```text
所有岛从 θ_t 开始
每岛用 AdamW 做 H 次本地更新 → θ_i
Δ_i = θ_t − θ_i
聚合 Δ_i
外部 Nesterov optimizer 更新 θ_{t+1}
新 θ_{t+1} 成为下轮所有岛的起点
```

原论文的低通信结果主要来自语言模型/C4 实验，不能直接保证决策后训练、长文评分或非 IID 中文数据也成立。[S6][S7]

需要比较的因素：

- H 太大导致分叉，特别是 head、少数类和语言分布。
- 内部 Adam m/v 的保留/重置策略；外部 momentum 状态。
- 不同 island 的学习率、更新次数与数据配比。
- 增加样本权重不自动消除不同 H、不同 Adam 轨迹造成的偏差。
- BF16/INT8 delta 压缩的误差、error feedback、量化与 CPU staging 成本。
- 更新基版本和 round ID；不接受旧 θ_t 的 delta 进入新一轮。
- 失败/超时后的原子提交、重试去重以及 data cursor 恢复。

先用相同步数、均衡任务配比的同构岛验证；异构 H20/NPU 岛与异步 gossip 放在扩展阶段。

### 7.6 研究与工程已有工作

- **DiLoCo**：算法定义与训练结果；不重复声称发明。[S6]
- **OpenDiLoCo**：复现研究，旧仓库标记不再维护。[S7][S25]
- **prime-diloco**：CPU 外部同步、分块 INT8、FSDP2、弹性通信等参考；当前 `PrimeIntellect-ai/prime` 已是另一类 CLI/SDK 仓库，不按旧 README 链接混用。[S26]
- **torchft**：容错 DDP/HSDP、LocalSGD/DiLoCo；后两者官方标为 experimental。[S27]
- **TorchTitan**：训练主干、FSDP2/HSDP、数据、checkpoint 和 profiling 的参考；最新版依赖可能不同于当前 NPU 镜像，不盲目使用 nightly。[S28]
- **Factored Gossip DiLoCo，ICML 2026**：指出低频通信的 outer synchronization 仍存在阻塞和长尾，适合后续研究而非第一阶段依赖。[S29]

贡献点应落在**决策后训练负载、可变问题数/长度的训练语义、NPU 后端、质量与校准约束、跨岛状态提交与恢复**。

## 8. 当前网络未知项

内部知识库查询到《NCCL常见故障排查方法》，包含交换机亲和性、NUMA、TCP fallback、版本不匹配、数据预取争用、PFC/ECN 等案例。这些是一般排查依据，未证明当前 HL/LF 训练链路存在这些问题。[S21]

关键词 `H20 Ascend910B4 训练 RDMA HCCL 带宽` 的第一次明确返回只有 GPU Sandbox 枚举；不能支撑这两种卡的训练性能。之后用 `训练任务 跨机 RDMA NCCL HCCL 网络带宽 优化` 获取了有效 NCCL 一般依据。

**知识库未找到可直接支撑的依据**来确认当前队列的 910B4 HCCL 带宽、HL↔LF 有效带宽、GPUDirect 条件和调度拓扑。下一步需要在正式实验中进行节点身份、NIC/设备拓扑、P2P 与真实 collective 探测。

后续通信专项核对确认：CPU Gloo 有官方通信原语，prime-diloco 有 CPU FP32 更新同步及写回设备的代码先例；其 local mesh 使用 CUDA，尚不能当作现成异构支持。NCCL/HCCL 不提供标准 CUDA/NPU 设备混合 collective。共享 HDFS/ByteNAS 若是唯一可达路径，需要单独的更新文件与轮次提交协议。具体支持矩阵、源码引用、数据量及验收条件见 [communication-feasibility.md](communication-feasibility.md)。

后续运行证据已确认本次 HL 的 H20/910B4 两个不同节点直接 IPv6 TCP 与 CPU Gloo 可通，4 KiB–256 MiB 每档 3 次平均及设备写回逐元素一致，双方正常退出。该证据未覆盖 HL↔LF、FSDP 分片、完整训练、RDMA 路径与长期稳定性。

读取 `/sys/class/infiniband` 只证明设备可见；`iperf` TCP 结果不等于 NCCL/HCCL 的实际 transport；`tc` 对 TCP 限速不一定作用于 RoCE/RDMA。不能把 `sleep` 增加的 step 时间冒充真实低带宽网络实验。

## Sources

本报告 [S#] 为本轮实际读取的来源索引；平台快照及本地文件单独标注。实验设计参数在方案文件中标记为建议，不是实测结果。

- [S1] Laya 官方微调文档：https://nandhakishorm.github.io/laya/finetune/
- [S2] Clef-Flash 官方模型卡，9B joint schema：https://huggingface.co/Cloudflare/clef-flash
- [S3] Clef 官方模型卡，27B joint schema：https://huggingface.co/Cloudflare/clef
- [S4] Qwen3-8B 官方模型卡：https://huggingface.co/Qwen/Qwen3-8B
- [S5] Qwen3-14B 官方模型卡：https://huggingface.co/Qwen/Qwen3-14B
- [S6] DiLoCo 原论文：https://arxiv.org/html/2311.08105v3
- [S7] OpenDiLoCo 论文：https://arxiv.org/abs/2407.07852
- [S8] Arnold FAQs 知识库命中：[internal-reference-omitted]
- [S9] Decision 3.0 d3-mini 官方模型卡：https://huggingface.co/vllm-sr/d3-mini
- [S10] Liquid d1 官方模型卡：https://huggingface.co/LiquidAI/d1-3B
- [S11] Laya 公开训练与基础结构：https://raw.githubusercontent.com/NandhaKishorM/laya/main/laya/train.py · https://raw.githubusercontent.com/NandhaKishorM/laya/v0.3.7/laya/common.py
- [S12] PyTorch DDP 官方 API：https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- [S13] PyTorch FSDP2 教程与 API：https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html · https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- [S14] TorchNPU 官方仓库：https://github.com/Ascend/pytorch
- [S15] TorchNPU 配套版本：https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md
- [S16] Ascend 训练 attention 文档入口，正文读取有限：https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_fusion_attention.md · https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/appdevgapi/context/FlashAttentionVarLenScore.md
- [S17] msprof-analyze 官方仓库：https://github.com/Ascend/msprof-analyze
- [S18] NVIDIA H20 官方产品版本文档：https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h20.html
- [S19] Transformer Engine 低精度训练说明：https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- [S20] Qwen3-32B 官方模型卡：https://huggingface.co/Qwen/Qwen3-32B
- [S21] 《NCCL常见故障排查方法》知识库命中：[internal-reference-omitted] ，查询成功 request_id=redacted
- [S22] NVIDIA NCCL 网络诊断：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html
- [S23] NCCL 流量与带宽定义：https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- [S24] TorchTitan FSDP2/HSDP 梯度累积说明：https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/FSDP.md
- [S25] OpenDiLoCo 代码与维护说明：https://github.com/PrimeIntellect-ai/OpenDiLoCo
- [S26] prime-diloco：https://github.com/PrimeIntellect-ai/prime-diloco
- [S27] torchft：https://github.com/pytorch/torchft
- [S28] TorchTitan：https://github.com/pytorch/torchtitan
- [S29] Factored Gossip DiLoCo：https://openreview.net/forum?id=1MynM9hkFH
- [S30] MASSIVE 官方来源数据卡：https://huggingface.co/datasets/AmazonScience/massive
- [S31] HelpSteer2 官方数据卡：https://huggingface.co/datasets/nvidia/HelpSteer2
- [S32] ContractNLI 原始数据说明：https://stanfordnlp.github.io/contract-nli/
- [S33] ModernBERT 官方模型卡：https://huggingface.co/answerdotai/ModernBERT-large
- [S34] 既有大词表 loss 方法 CCE：https://github.com/apple/ml-cross-entropy
- [S35] vLLM 决策读出与服务接口：https://docs.vllm.ai/en/latest/models/structured_decisions_models/
- [S36] NCCL correctness/performance tests：https://github.com/NVIDIA/nccl-tests
- [S37] PyTorch DDP 压缩 hooks 与 PowerSGD：https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html
- [S38] Laya Ascend 社区移植，主要为推理参考：https://github.com/zzhdbw/laya-Ascend

以上论文/模型卡结果属于其具体任务和版本；本报告没有将作者排行榜数据拼成 H20/NPU 训练性能比较。
