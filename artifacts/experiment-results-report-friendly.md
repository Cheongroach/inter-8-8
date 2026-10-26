# rank 级性能记录：它解释局部开销，不直接决定组合胜负

[原报告](experiment-results-report.md)由 worker JSONL 生成，展示主要 rank 0 记录。跨组是否更快，应使用[组合表说明](combination-comparison-friendly.md)中的所有 rank 最慢时间；单看 rank 0 会漏掉其他进程等待。

`rank` 是分布式进程编号；一张卡通常对应一个进程。`cuda` / `npu` 表示混合中的各四卡侧，`cuda8` / `npu8` 是同构八卡，`q25` / `q35` 分别为 Qwen2.5-7B / Qwen3.5-4B 文本主干。

## 三张表的分母不同

| 原表 | 如何计时 | 正确读法 |
|---|---|---|
| 生成 profile | 每配置三次，排除首次；其余输入 / prompt 长度不同 | 组 token/s 是吞吐观测，不是单请求延迟；不能无条件稳定排序 |
| 训练 profile / paired | 排除首个更新；paired 显示 7 个有效计时更新，profile 3 个 | 观察热身后局部计算；不能替代完整八更新组合时间 |
| 生成 / 更新分工 | 各 producer / trainer rank 0 的总时间 | 最终要取所有 rank 最慢者，包含最后发布 |

例如 q35 混合 `local-h8`：CUDA rank 0 组墙钟为 20.268 秒，NPU rank 0 为 22.597 秒；完整八更新组合表为 24.58 秒。三者计时范围不同，不能认定它们互相矛盾。

同构 NPU q35 的逐参数 overlap 分工，trainer rank 0 为 40.696 秒，producer rank 0 为 40.695 秒；完整组合最慢值约 40.70 秒。这里差小也不能据此默认所有其他行都能用 rank 0 替代。

## 哪些差别不能用于质量比较

profile-b1 / b2 / b4 使用不同 global batch，端点 loss 不能直接说明哪个训练方法更好。paired 固定全局每步 16 样本；单岛 token/s 仅累计该岛监督 token，组合要用两岛总数与整体时间。

Local SGD 保留独立 Adam 状态，参数平均不是全局同步 AdamW，也不是完整 DiLoCo。生成回答的 CE 预验没有 reward、KL 或 importance correction，不是 PPO / GRPO。

共同实现为 Transformers 5.2 SDPA / FSDP，FP32 master / AdamW、BF16 计算；没有生产连续批处理或专用 FLA。生成 batch 增大可以提高吞吐，但这些记录不能证明硬件的最佳实现性能。

## 完成记录与缺失项

四个最终历史任务均完成、worker 退出 0；历史失败或缺失不能当作零耗时。原报告的 suite 名称用于对应[原始 JSONL](experiment-results/)，不是新的独立实验样本。

本报告证明当前工作负载的运行和计时数据回收；多个训练 seed、固定质量门槛、压缩发布、故障恢复没有验收。最终系统判断见[结论说明](experiment-conclusions-friendly.md)，新模型 H20×16 使用另一套完整 case 计时，见[主报告](16card-experiments/paper/report-friendly.md)。
