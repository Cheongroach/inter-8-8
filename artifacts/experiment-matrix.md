# H20＋910B4 组合实测

用户已将目标从固定 8＋8 放宽为先观察 4＋4 的实际收益。实验同时覆盖成熟模型 Qwen2.5-7B-Instruct 和新架构 Qwen3.5-4B 的文本主干。Qwen3.5 不训练视觉分支或 MTP。

| 组合 | 实际工作 | 比较目的 |
|---|---|---|
| 两种硬件各自 4 卡 | 生成 batch 1/4/8、输出上限 64/128；全参数训练 microbatch 1/2/4 | 找出阶段优势是否交叉 |
| 4＋4 逐步同步 | H:N microbatch 2:2、3:1、1:3，全局 16 样本/step | 不同算力配比能否减少慢组等待 |
| 4＋4 Local SGD | 每 4/8 个局部 AdamW step 平均参数 | 节省通信是否抵消等待及优化轨迹变化 |
| H20 生成、910B4 更新 | 真实生成轨迹、全参数 CE 更新、完整 BF16 权重回传 | 生成/训练分工的端到端成本 |
| 910B4 生成、H20 更新 | 相同工作反向分工 | 不能预先假定设备排序 |
| 顺序 vs 重叠，发布间隔 1/2/4 | 每次发布校验参数名称、形状、版本和 SHA256 | 同步频率与可隐藏时间的关系 |
| 同构 8 卡 | 需实际执行相同数据和系统策略 | 判断混合能否胜过固定总卡数的最佳同构配置 |

输入是 GSM8K 的公开问题和答案，模型文件使用不可变清单并逐文件核对 SHA256。训练 master 参数与 AdamW 状态为 FP32，FSDP 计算为 BF16，岛内梯度归约为 FP32。跨岛按相同 FSDP rank 和参数清单交换对应分片，不能将不同分片直接归约。

分工测试使用**生成回答的 CE**，目的是验证吞吐、流水线、数据返回和权重发布；没有 reward、importance ratio 或 PPO/GRPO loss。它不是完整 RL，更不能据短程 CE 下降声称最终质量改善。

比较必须同时记录：

- 实际有效样本、监督 token、输出长度与 EOS。
- 最慢 rank 墙钟时间、计算时间、跨岛传输时间、权重加载时间。
- 每卡资源时与任务整体时间；“多申请了卡”带来的增加单独说明。
- 梯度/模型同步的优化语义；Local SGD 参数平均并不等同于逐步同步 AdamW。
- 可用后端与 kernel。相同 Transformers SDPA 只能作为共同实现的初测，不能代表两类硬件的最优推理服务。

四卡profile、4＋4同步配比和Local SGD均完成。两个模型两个分工方向的间隔1/2/4、顺序/重叠均完成；另测固定教师和分块发布，并实际执行两种同构对照。最新结果与边界见[实测后的选择](experiment-conclusions.md)。

任务：

- [H20×4]([internal-reference-omitted])
- [910B4×4]([internal-reference-omitted])
- [H20×8 同构对照]([internal-reference-omitted])
- [910B4×8 同构对照]([internal-reference-omitted])

重试预检两边均 `success`。门禁规则为 `data.status=error` 停止、`warn/hint` 展示后可继续；本次保留式合并 `request_context.precheck_sid`，并在 30 分钟内提交。
