# H20×8＋910B4×8 的训练收益：文献与立项判断

调研日期：2026-10-19。原文最初讨论固定16卡；用户随后将当前目标改为先做4＋4真实预验。实验已经实际执行，最新结论见[多组合实测与选择](experiment-conclusions.md)。本文件以下保留文献判断，实验状态和规模以最新报告为准。

## 1. 证据结论

公开研究支持异构训练的资源整合，也支持在有硬件互补性的工作负载中获得效率收益。但是，本轮检索和原文核对没有找到 **H20×8＋Ascend910B4×8 在相同模型、数据、训练预算下，同时超过 H20×16 和 Ascend910B4×16** 的直接实验证据。

研究中最接近这一目标的路线有两条：

- **异构流水线训练**：将模型的不同阶段放到不同设备，利用算力、显存和通信的互补性。H2 有收益先例，但模型、网络、芯片和对照口径与本项目不同。<a href="#source-1">[cite:1]</a>
- **生成与训练分工的 RL**：将自回归生成与梯度更新分别放到适合的设备。AReaL-Hex、StreamRL 和 AReaL 的结果支持这种系统方向，但没有证明本项目的固定 8＋8 配置。<a href="#source-2">[cite:2]</a><a href="#source-3">[cite:3]</a><a href="#source-4">[cite:4]</a>

**立项判断：不按原来的“双训练副本定期平均”方案直接扩成 16 卡主项目。异构 RL 分工值得进行真实负载性能预验；通过预验后再立项，不预先承诺混合硬件必胜。**

这里的“值得预验”与“已证明可立项”是不同结论。若必须在投入前确定混合 16 卡能胜过两种同构 16 卡，当前证据不足。

## 2. 通信探针的用途与局限

本项目已验证两个不同 HL 节点上的 H20 与 910B4，CPU 中转后 TCP/Gloo 聚合及设备写回正确，4 KiB–256 MiB 每档三次通过，双方退出码为 0。探针解决了“两个后端能否交换张量”这个前置问题，不证明训练吞吐、梯度一致性、完整模型容量或收敛收益。<a href="#source-11">[cite:11]</a>

256 MiB CPU Gloo all-reduce 的中位耗时为 344.19 ms。它不是完整模型传输实验，不能线性外推为 8B/32B 权重同步时间，也不是链路标称带宽。TCP 路径累计双向 payload 约 1.97 GiB，仍没有覆盖长期竞争、分片映射和故障恢复。<a href="#source-11">[cite:11]</a>

通信成功使工程方案有实施可能；立项价值要靠减少真实训练的耗时、费用或资源等待来证明。“能同时使用两种卡”本身不足以满足本项目的收益目标。

## 3. “训练效果更好”的比较口径

| 比较问题 | 可接受证据 | 不能替代的指标 |
|---|---|---|
| 相同训练数据曝光量，最终质量更高 | 固定评测、多个 seed、准确率/损失/概率质量 | 不同 token 数的端点 loss |
| 达到同一质量更快 | 质量阈值固定，真实墙钟时间更短 | 单个 step 更快、卡利用率更高 |
| 相同运行时长，质量更高 | 同一初始模型、同一任务、固定时长的质量曲线 | 只报告生成 token/s |
| 相同花费，质量或吞吐更好 | 实际设备计价、资源小时和同一质量目标 | 固定 16 卡数量 |
| 固定资源池，能更快完成 | 计入可审计的排队与执行时间 | 用“同构卡申请不到”代替同规模性能对照 |

本项目采用 **达到同一质量的墙钟时间** 为主指标，同时报告固定数据预算下的质量。混合 16 卡必须与经过调优的两种同构 16 卡比较。租用费用和排队时间可单独报告，但不能偷偷替换用户要求的固定卡数比较。

在同步梯度、相同全局 batch 和相同优化器等理想条件下，换设备不会额外引入模型能力。浮点与 kernel 差异需要控制，不能把数值误差当成可靠的泛化机制。Local SGD 改变优化轨迹，有质量收益的可能，但同构硬件也能采用同一算法。<a href="#source-5">[cite:5]</a><a href="#source-6">[cite:6]</a>

## 4. 原文核对

### 4.1 H2：确有互补性，但不是固定同卡数胜过两种同构的证明

H2（2025）提出 DiTorch、DiComm、HeteroPP/HeteroAuto。在 100B LLM 上，异构芯片分别承担不同流水线阶段；高显存设备承接需要保存更多激活的早期阶段，其他阶段采用不同层数、TP 度数与重计算策略。跨芯片连接有专门的设备直连 RDMA 优化。<a href="#source-1">[cite:1]</a>

论文表 5 将硬件匿名为 Chip-A/B/C/D，没有标明 H20 和 910B4。表 6 的同构基线是每类 256 卡；表 7 的混合实验覆盖数百至上千卡，部分实验改变全局 batch。它不支持对本项目硬件进行型号映射或性能承诺。

第 6.2 节定义的 HeteroSpeedupRatio 是：

```text
混合集群吞吐 / Σ(该类芯片数量 × 同构基线每卡吞吐)
```

Exp-A/Exp-B 在全局 batch 取各组之和时报告 109.03%/104.29%；在保持全局 batch 时为 89.56%/77.45%。这些数值说明实验条件和 batch 口径很重要，不能把摘要中的“超线性”解释成任意异构组合都更快。<a href="#source-1">[cite:1]</a>

对本项目的启示是寻找 **可测量的内存/计算互补性与不同流水线布局**。这条路线不是两个完整模型训练副本定期平均。当前 CPU Gloo 探针也不等同于论文的设备直连 RDMA 能力。

### 4.2 HETHUB：负载均衡收益，未超过更强同构方案

HETHUB（2024）建立跨厂商通信、性能预测与自动并行规划。Llama2-70B 的一个异构设置中，非均匀流水线划分用时 412.49 ms，均匀划分为 507.3 ms；改善来自同一个异构系统内的工作分配。<a href="#source-7">[cite:7]</a>

论文还报告了异构吞吐可接近混合资源的理论组合能力，但部分比较仍低于更强的同构硬件。达到理论组合能力的 97.49% 不是比同构更快 97.49%。这类工作证明“把异构池用好”，不证明“混合比最佳同构池更强”。

### 4.3 FlashFlex 与 HAP：异构并行有价值，也有明确反例

FlashFlex（2024）的主要比较使用相近总峰值 FLOPS，而不是相同 GPU 数。它用 3080Ti/3090/4090/A800 混合训练 7B–30B；与同构 A800 的 RDMA 配置相比，异构 MFU 仍有差距。论文表 1 的 7B 实测值为同构 RDMA 53.72%、两个异构设置 31.19% 和 33.47%。这些是不同配置下的利用率，不能直接换算为本项目吞吐。<a href="#source-8">[cite:8]</a>

HAP（EuroSys 2024）优化异构张量分片比例、维度和通信方法。它进一步说明要解决的是不对称工作分配，而不是给不同类型卡分相同数量的工作。其 GPU 实验也不提供 H20/910B4 的直接证据。<a href="#source-9">[cite:9]</a>

### 4.4 DiLoCo 与 EDiT：优化算法收益不等于异构芯片收益

DiLoCo 在 C4、60M/150M/400M 模型上研究本地 AdamW、多步局部训练和外部 Nesterov 更新。表 2 的一个比较中，大 batch 同步基线 PPL 为 15.30，DiLoCo 为 15.02；两者具有相应的 compute/data 口径。论文也明确使用 step 数作为墙钟时间的代理。不能将这些结果当作本项目 8B 模型的真实加速保证。<a href="#source-5">[cite:5]</a>

EDiT/A-EDiT（2025）结合分片、逐层同步和异步更新。其主要质量实验在 64 张 A100 上完成；1B FineWeb-Edu 的评测平均分为基线 44.60、EDiT 44.82、A-EDiT 43.90，异步并非所有指标都更好。慢节点/低带宽实验通过暂停进程、重复通信人为引入，不能当作实际 GPU/NPU 混合网络的证据。<a href="#source-6">[cite:6]</a>

如果把混合 DiLoCo 与同构同步训练比较，可能测到算法变化而不是硬件互补。要判断异构本身的贡献，必须补同构 DiLoCo/EDiT 对照。

### 4.5 AReaL-Hex：最接近 H20 分工的先例，主对照是等费用

AReaL-Hex（2025）将 H20 用于生成、H800 用于反向训练，规划并行度、资源分配和数据过期约束。论文主实验按相同租用预算，对比混合部署与同构 H800/H20 的 AReaL，报告相对 H800 基线最高 1.50 倍吞吐。<a href="#source-2">[cite:2]</a>

第 3 节案例对照为 32 H800、88 H20、24 H800＋24 H20。它们卡数不同，因此不能用该结果回答固定 8＋8 与 16/16 的问题，更不能把 H800 的训练效率替换为 910B4。

论文对硬件互补的机制较明确：生成和反向训练偏好不同设备。但主要评价是 30 个 RL step 的系统吞吐，不能据此声称固定卡数下最终准确率更高。其表 2 中 7B 混合权重更新为 58.34 秒，且采用 NCCL，提示权重同步的真实成本和跨 CUDA/NPU 的适配仍要测量。<a href="#source-2">[cite:2]</a>

### 4.6 StreamRL 与 AReaL：支持分工与质量约束

StreamRL（2025）将生成与训练解耦，做 streaming、阶段均衡和长尾调度。在第 7.3 节，生成池放到 H20、训练留在 H800，报告按费用归一化的吞吐改善 1.23–1.31 倍。论文另外在 Qwen2.5-32B 的特定任务上展示同步与 one-step asynchronous 的 reward 曲线相近；没有证明本项目 8＋8 的同卡数优势。<a href="#source-3">[cite:3]</a>

AReaL 的 NeurIPS 2025 论文报告相同 GPU 数下，相对同步系统最高 2.77 倍训练加速并保持相近或更好质量。但这是异步系统收益，同构设备也可以使用。项目不能通过给同构基线关闭异步、packing 或长度均衡来制造优势。<a href="#source-4">[cite:4]</a>

### 4.7 CadenceRL：最新的互补调度研究，评价范围仍有限

CadenceRL（2026-09）研究异构 rollout 池的长短轨迹重排，报告最高 48% decode 吞吐提升、P95 轨迹延迟减少最高 64%。论文测试 A910X、BI-V150、H800，trainer 固定在 H800 且预留容量，让 rollout 成为瓶颈。该结果属于生成调度，不能作为 H20/910B4 联合反向训练或同卡数 time-to-quality 的证明。<a href="#source-10">[cite:10]</a>

它也提示简单“静态分配完就不管”不足以覆盖负载变化，但项目首版不需要复制其轨迹迁移系统。

## 5. 两种可能的收益机制

### 5.1 相同训练工作：混合通常是折中

以下是分析模型，不是设备实测或性能预测。令 H20 和 910B4 在同一种训练任务上的有效每卡计算速率分别为 r_H、r_N，并暂时假设完全线性扩展、无通信、相同优化语义，则：

```text
混合 8＋8 的速率 = 8r_H + 8r_N
最佳同构 16 的速率 = 16 × max(r_H, r_N)
因此混合速率不超过最佳同构速率。
```

现实系统可能突破这个简化比较，因为 16 卡扩展不是线性的，显存/重计算/网络布局会改变。H2 给出了这种可能性，但要通过真实瓶颈证明，不能从“硬件不同”直接得出。<a href="#source-1">[cite:1]</a>

原双岛方案每岛都执行完整模型的前向、反向和 Adam 更新，没有把两类设备的不同长处分工使用。两岛固定 H 的同步还会等待较慢的一组；减少同步次数不自动解除持续慢组问题。<a href="#source-5">[cite:5]</a><a href="#source-6">[cite:6]</a>

### 5.2 两种不同工作：互补可以超过任一种同构

RL 有生成和参数更新两个阶段。若真实测量得到：

```text
H20 的生成效率更高
910B4 的反向/优化器更新效率更高
```

则混合系统有机制上的收益空间：H20 生成训练轨迹，910B4 更新唯一的 policy，再把新权重发送给 H20。不需要跨硬件逐步同步所有梯度，也不需要两个 Adam 副本做模型平均。文献支持这一机制，但本项目的设备排序仍是未知项。<a href="#source-2">[cite:2]</a><a href="#source-3">[cite:3]</a>

判别时采用如下粗略的阶段模型，并用实际 timeline 校正：

```text
T_mix ≈ max(T_rollout_H20×8, T_update_NPU×8) + T_未隐藏同步及其他开销
T_H16 = H20×16 下调优后的端到端 RL 时间
T_N16 = NPU×16 下调优后的端到端 RL 时间
```

只有 `T_mix < min(T_H16, T_N16)` 且模型质量符合要求，才能满足用户的强目标。生成量与训练输入 token 量不同，应按同一批有效轨迹及相同更新次数测量，不能简单用两个 tokens/s 取较小值。

如果 H20 在生成和训练两阶段都更快，或者 NPU 训练虽快却无法覆盖同步开销，那么固定 8＋8 没有足够理由优于 H20×16。

## 6. 项目范围选择

| 候选项目 | 满足固定 16 卡强目标的依据 | 当前判断 |
|---|---|---|
| 两个完整训练副本＋周期平均 | 算法可能改善质量，但同构也可复用；硬件互补弱 | 不作为主项目直接扩卡 |
| H20/910B4 异构流水线 | 内存/计算互补有论文先例；需要训练阶段 profile 与跨设备激活/梯度传输 | 技术难度高，作为坚持 SFT/预训练时的候选 |
| H20 rollout＋910B4 policy update | 阶段互补有多个来源；跨组低频权重传输与轨迹传输 | 优先做性能预验，通过才进入主项目 |

第三条符合用户已有 RL/rollout 经验，更容易把工程贡献落在性能可解释的环节：异构后端解耦、版本化权重发布、过期数据控制、生成与训练的动态均衡。生成和更新的框架仍应复用，不能把现有异步 RL 能力重新包装为原创。

这会改变原 Laya 项目的工作负载。单步决策模型的候选输出很短，离线 CE/蒸馏没有长期 decode 瓶颈；不能为了异构收益而给 Laya 人为添加长文本 rollout。若切到 Qwen 8B/14B reasoning 或 tool-use RL，应明确说明负载已扩展为长轨迹后训练，不再声称只是 Laya 训练优化。

AReaL 与 verl 已有 Ascend 训练支持，但当前文档与代码存在分支、镜像和依赖差异，支持分别部署 GPU/NPU 不等于同一系统混用已可直接运行。AReaL 的 NPU 安装文档明确指向独立 `ascend-v1.0.4` 分支；跨后端协议和兼容性仍需验证。<a href="#source-12">[cite:12]</a><a href="#source-13">[cite:13]</a>

## 7. 进入主实验前的验收

### 7.1 真实负载性能预验

以固定 Qwen3-8B dense 配置、同一初始权重、同一公开数据与 BF16 为主。小模型只用于正确性检查，14B/32B 是后续规模扩展。不要用 LoRA 结果代替全参数训练验收。

需要测量四条性能曲线：

1. H20 的真实生成性能：不同并发、真实 prompt/response 长度分布、有效轨迹/s、尾延迟。
2. 910B4 的相同生成工作：用于构建同构 NPU 的完整基线，不能省略。
3. H20 的 forward/backward/optimizer 性能：固定有效训练 token 和优化器语义。
4. 910B4 的相同更新工作：确认其是否在本任务上有足够优势。

单卡结果不能决定 8/16 卡布局。随后按可运行的并行度测 4/8 卡，并补齐同构 16 卡曲线；实际拓扑、collective 和最慢 rank 必须记录。测试真实模型权重发布与加载，而不只传 256 MiB 常量张量。

若设备优势没有交叉，或者 CPU staging/发布/加载的暴露时间吞掉收益，停止“混合优于两种同构”的项目假设。可保留有价值的后端优化成果，但不继续靠扩大模型、改变 batch 或限制基线寻找胜利。

### 7.2 固定 16 卡的公平对照

| 实验臂 | 硬件 | 要求 |
|---|---|---|
| H | H20×16 | 调优同构 colocated、disaggregated/async 方案，取可行最优 |
| N | 910B4×16 | 相同训练语义，采用可用后端的调优方案 |
| M | H20×8＋910B4×8 | 固定 8＋8，比较角色分配与可行并行度 |
| M-static | 同 M | 固定分工与同步策略，用于判断新增调度是否必要 |

每个臂可按显存和性能选择不同 microbatch、packing、重计算与并行布局，但全局训练目标、有效数据预算、reward、模型配置和允许的数据过期范围保持一致。基线应有同等调参预算，不能只比较默认 8/8 资源切分。

两个同构系统也允许针对各阶段优化，不能要求它们只能把 16 张卡一起交替使用。Mixed-vs-async-homogeneous 是必要对照，否则会把异步带来的收益错算给异构。

主结论使用多个 seed，建议 3 个；评测与质量阈值在测试前固定。记录训练目标值、held-out 准确率、生成长度、截断、KL、行为策略版本、importance ratio、有效/丢弃轨迹量及故障重试。系统快但只留下短轨迹或容易题目，会改变训练分布。

### 7.3 明确的继续与停止规则

下列阈值属于建议验收要求，不是实测或收益预测：

- 达到固定质量目标的时间，相对 **两个同构系统中更快者** 至少缩短 10%。
- 固定数据预算下 held-out accuracy 降幅不超过 1 个百分点，关键任务组没有未解释的明显退化。
- 如果要声称“最终质量更好”，需另行证明质量提升超过 seed 波动，不能仅凭更快达到阈值。
- 同步、丢弃、重试和恢复全部计时；扣除后的 kernel-only 结果只能帮助归因。
- 若只能胜过同构 NPU、不能胜过同构 H20，则只能报告这一较弱结果，不满足当前强目标。
- 若只在人为限速网络或故意弱化同构配置下胜出，只能作为对应条件的结论。

预验和实验能否在 1–2 个晚上的闲置窗口内完成，需要用实际生成/更新速度估算。目前不承诺完成时长。

## 8. 最终决策

**原方案不直接立项扩成 H20×8＋910B4×8。** 文献已足以排除“混合设备或平均模型就自然更好”的理由。

**异构 RL 分工为有条件推荐。** 其系统机制有论文依据，但 910B4 能否承担比 H20 更高效的反向更新、固定 8＋8 的均衡点和权重发布成本尚未知。性能预验通过并能公平胜过两种同构 16 卡后，才把它确立为主项目；否则停止该强收益假设。

如果必须保持离线 Laya/决策模型训练范围，优先选择能以 profile 证明的 NPU 算子、内存或负载均衡优化；异构流水线需独立立项，不能沿用已完成通信探针就推定性能优势。

## Sources

<a id="source-1"></a>
[cite:1] H2: Towards Efficient Large-Scale LLM Training on Hyper-Heterogeneous Cluster over 1,000 Chips（2025），重点核对 §3.2、§4.1、§6.1–6.2、表 5–7。https://arxiv.org/html/2505.17548v1/

<a id="source-2"></a>
[cite:2] AReaL-Hex: Accommodating Asynchronous RL Training over Heterogeneous GPUs（2025），重点核对 §3、§4.4–4.7、表 2。https://arxiv.org/html/2511.00796v1

<a id="source-3"></a>
[cite:3] StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation（2025），重点核对 §7.3–7.4。https://arxiv.org/html/2504.15930

<a id="source-4"></a>
[cite:4] AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning，NeurIPS 2025 官方论文页。https://proceedings.neurips.cc/paper_files/paper/2025/hash/33c00862bfa29ac72ecf630a41e19352-Abstract-Conference.html

<a id="source-5"></a>
[cite:5] DiLoCo: Distributed Low-Communication Training of Language Models（2023，读取 2024 v3），重点核对 §3 与表 1–2。https://arxiv.org/html/2311.08105v3

<a id="source-6"></a>
[cite:6] EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models（读取 2025 v2），重点核对 §4.1–4.3、表 1–2。https://arxiv.org/html/2412.07210

<a id="source-7"></a>
[cite:7] HETHUB: A Distributed Training System with Heterogeneous Cluster for Large-Scale Models（2024），重点核对 §4.4。https://arxiv.org/html/2405.16256

<a id="source-8"></a>
[cite:8] FlashFlex: Accommodating Large Language Model Training over Heterogeneous Environment（2024 v1，后续引用亦使用 HexiScale 名称），重点核对 §5.1 与表 1。https://arxiv.org/html/2409.01143v1

<a id="source-9"></a>
[cite:9] HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis，EuroSys 2024。https://arxiv.org/abs/2401.05965

<a id="source-10"></a>
[cite:10] Reshaping Rollout Workloads for Asynchronous RL Post-Training on Heterogeneous Accelerators / CadenceRL（2026-09），重点核对 §6.1 的硬件和 trainer 配置。https://arxiv.org/html/2609.36899v1

<a id="source-11"></a>
[cite:11] 本项目 H20/910B4 两节点通信报告与原始事件。[报告](communication-results/report.md)，[manifest](communication-results/manifest.json)。

<a id="source-12"></a>
[cite:12] AReaL NPU 安装与分支文档。https://areal-project.github.io/AReaL/en/tutorial/installation_npu.html

<a id="source-13"></a>
[cite:13] verl Ascend Quickstart，支持配置和环境要求。https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html

定量结果为各论文自己的报告，部分来自预印本，未由本项目复现。对固定 H20/910B4 16 卡强比较仍为证据不足，不将不同论文的数值拼成设备性能预测。
