# 决策模型训练 Infra

**2026-10-25 更新：** 原NPU×16与混合8＋8实验已停止，不再继续。
Jev/Laya的独立决策质量进展见[两轮报告](decision_study/REPORT.md)，
其中第一轮负结果和第二轮新保留集结果均完整披露。尚无SOTA证据。
下文保留为先前训练系统背景。


项目范围：Laya 与同类决策模型，包含 Qwen 8B–32B 主干的全参数后训练、多问题决策 head、H20/CUDA、910B4/NPU，以及低带宽分布式同步。

当前状态：两模型、4＋4混合与两种同构8卡已执行全参数训练和生成/训练分工预验，覆盖batch配比、Local SGD、发布间隔、发布实现和固定教师。**暂不推荐为了加速强行绑定H20/910B4。** 先读[实测后的选择](experiment-conclusions.md)，完整数字见[组合对照](combination-comparison.md)。

阅读顺序：

1. [异构训练收益与立项判断](heterogeneous-training-decision.md)：比较 8＋8 与两种同构 16 卡，论文对照口径、条件和停止标准。
2. [一段话版本与核心价值](project-pitches.md)：原候选项目的表述与面试模板，立项状态以最新决策报告为准。
3. [通信可行性核对](communication-feasibility.md)：NCCL/HCCL、CPU Gloo/TCP、共享存储的支持范围，数据量与待验证条件。
4. [调研](research.md)：训练特殊工作、硬件优化、通信瓶颈、模型与平台证据。
5. [原项目方案](proposal.md)：8B 主实验、14B/32B 扩展、系统模块、实验基线与验收。
6. [H20/910B4 通信实测](communication-results/report.md)：不同节点证据、4 KiB–256 MiB 聚合正确性、操作耗时与退出状态。

最新判断：同步训练明显慢于同构；生成/训练分工尚未胜过两种同构中较快者。优先考虑各资源池独立运行、固定教师离线产数，或可由profile验证的设备内优化。短程CE实验不构成完整RL或同质量收敛证据。

原候选贡献是跨 CUDA/Ascend 更新同步与一致恢复。最新方向需要证明真实训练收益，再选择后端解耦、权重发布与阶段均衡的实际缺口；现成 FSDP、DiLoCo 或异步 RL 本身不作为原创能力。

异构通信采用 CPU 中转；本项目已验证本次 HL H20/910B4 两节点的 TCP 与 CPU Gloo 路径。标准 NCCL/HCCL 不提供设备混合 collective；HL↔LF、完整训练和恢复尚未验证。实际网络只允许共享存储时需另建轮次交换协议。

多个 AI 正在共享仓库。本项目产出集中在 `laya/`，对 `training_infra_lab/`、`lowband_training/` 只读参考。目录中的方案参数是建议；既有其他目录的探针结果不算本项目训练验收。

主要验证边界：

- 同步优化要验证 loss、梯度、更新和数据进度。
- Local SGD、DiLoCo 与压缩要验证收敛、校准和同质量时间。
- NPU 适配要覆盖 backward，不能仅验证推理。
- 32B 内存规划要包含 optimizer、激活和参数物化，不能只数 BF16 权重。
- 多 pod 不等于多物理机；网络实验要证明节点与 transport。

当前按用户授权执行 4＋4 性能预验，模型为 Qwen2.5-7B-Instruct 和 Qwen3.5-4B 文本主干。固定 8＋8 或同构 16 卡不再作为当前必须达到的规模。
