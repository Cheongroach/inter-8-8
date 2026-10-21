# 新架构对照矩阵

2026-10-19。按用户要求，主测从Qwen2.5改为Qwen3.5-9B与Qwen3.8-27B，Qwen3.5-4B保留作校验。三个最终任务已经提交，当前均started/未分配worker，尚无新模型的16卡性能结果。

| 模型 | 文本主干参数 | 层数 | DeltaNet / 完整注意力层 | 当前用途 |
|---|---:|---:|---:|---|
| Qwen3.5-4B | 4,205,751,296 | 32 | 24 / 8 | 已有实测、小规模校验 |
| Qwen3.5-9B | 8,953,803,264 | 32 | 24 / 8 | 新架构主测 |
| Qwen3.8-27B | 26,895,998,464 | 64 | 48 / 16 | 新版本、大模型主测 |

参数量由官方配置构建meta模型计数，未包括视觉/MTP。Qwen3.6-27B也已核实有开放权重；本轮选较新的3.8-27B。官方3.8模型卡明确说明沿用3.5的架构基础，因此不将3.6/3.8版本提升当作一种完全独立的架构。

任务：

- [H20×16]([internal-reference-omitted])：两个8卡worker。
- [910B4×16]([internal-reference-omitted])：两个8卡worker。
- [H20×8＋910B4×8]([internal-reference-omitted])：CUDA/NPU各一个8卡角色。

对照涵盖全局FSDP16、机内分片/机间复制、机间CPU Gloo/TCP逐步梯度同步及每4/16步参数平均，以及8卡生成/8卡训练。NVIDIA原生通信分默认与强制Socket；NPU的无机间RDMA场景使用机内HCCL、机间CPU/TCP，不能称为原生HCCL关闭RDMA。

同一模型的各组使用相同数据与更新次数，27B microbatch为1，其他模型为2。训练BF16计算、FP32 master及AdamW；周期参数平均保留本岛优化器状态，不是DiLoCo，不具备同质量收敛证据。生成/训练分工是实际回答的CE系统预验，不是完整RL。

为避免27B完整FP32模型在单卡分片前OOM，先在CPU加载BF16权重，再逐层转FP32并FSDP分片；最后处理embedding、LM head与norm。实际设备峰值及forward/backward是否通过仍需worker验证。

新模型聊天文本前缀一致，但完整文本与prompt分别分词会在答案边界发生BPE合并。训练按聊天文本边界拆分后分别分词，保持实际生成prompt与监督答案的契约，避免把token级前缀假设用于这些模型。

元数据、tokenizer、权重清单已逐文件锁定哈希，3.5-9B和3.8-27B的ModelScope权重哈希均与官方Hugging Face清单一致。meta参数名覆盖核对与小尺寸CPU前向/反向均产生通过输出；本地沙箱在进程收尾报告/proc访问限制，不能作为完整runtime验收。原模型尺寸、GPU/NPU训练与性能验证尚待任务启动。

三个任务的最终预检均success，保留式合并request_context.precheck_sid后30分钟内提交；error停止、warn/hint展示后继续。平台修改排队入口需要stop/retry，排队等待可能重新计算；本次未修改其他人的任务。

模型与代码：[模型哈希清单](new-model-manifest.json)、[代码哈希](code-sha256.json)。公开来源：[Qwen3.8官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[Qwen3.6官方发布](https://qwen.ai/blog?id=qwen3.6-27b&lang=en)、[Qwen3.5-9B官方仓库](https://huggingface.co/Qwen/Qwen3.5-9B)。
