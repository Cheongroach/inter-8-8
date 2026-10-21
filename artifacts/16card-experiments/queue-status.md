# 16卡跨机对照排队记录

更新：按用户要求已切换到Qwen3.5-9B与Qwen3.8-27B，旧队列trial已由新配置替换。最新模型、任务及验证边界见[新架构矩阵](new-architecture-matrix.md)。下文14:38状态作为历史记录保留。

2026-10-19 14:38（UTC+8）重新查询。两个同构16卡任务仍为started，当前trial均未分配worker；尚无同构16卡或关闭RDMA对照性能结论。两任务本轮提交时间约为10:45，已等待近4小时，不承诺排到时间。

- [H20×16]([internal-reference-omitted])：2个worker，每个8卡。
- [910B4×16]([internal-reference-omitted])：2个worker，每个8卡。

使用HL原队列、组2079、集群20，非弹性配额。两个worker按gang调度。运行上限为3小时，不包含排队；模型为Qwen2.5-7B-Instruct与Qwen3.5-4B文本主干，模型文件沿用已锁定清单。

已准备可自动执行的入口：先确认两个不同物理host，然后测试全局FSDP16、机内8卡分片＋机间复制、CPU/Gloo/TCP同步（间隔1/4/16），以及8卡生成＋8卡训练。CUDA原生通信分别测默认网络和NCCL_IB_DISABLE=1、NCCL_NET=Socket；NPU的无RDMA条件明确采用机内HCCL、机间CPU/TCP，不能称为原生HCCL关闭RDMA。

训练每配置16步、全局每步32样本，重复2次，记录完整case墙钟和所有rank完成事件。Local SGD保留本岛Adam状态，质量验证尚未完成。分工为真实生成回答的CE系统预验，不是完整RL。原生和CPU/TCP的算法区别必须在结论中明确，不能仅靠弱化同构基线制造异构收益。

8＋8混合组已完成：H20任务402c16fd53c4708a与NPU任务813188d499d39221均done，worker退出码0，阶段0至11全部code=0，日志包含LAYA16_COMPLETE。双方分别在11:22:46和11:22:53退出，已释放本次实验占用资源。其他任务未修改。

本地已完整回收的Qwen2.5混合训练计时：每配置16个更新、全局每步32样本，共64,103监督token；逐步同步两次为170.10/200.35秒，每4步平均参数59.51/93.04秒，每16步平均参数33.17/54.92秒。时间取双方全部16个rank的最大case墙钟。低频同步明显减少开销，但重复波动较大，且Local SGD保留独立Adam状态；不能据此判定胜过尚未运行的同构16卡或同质量收益。

最终run/pod状态和阶段完成证据来自本轮Merlin cn查询及log-proxy stdout。完整后半程计时没有全部导出到本地；成功完成事件不能替代计时数据。当前报告仅列已回收数字。

两个16卡预检均success。门禁为error停止、warn/hint展示后继续；保留式合并request_context.precheck_sid，并在30分钟内提交。当前状态只证明任务已入队，不承诺排到时间，也不说明16卡性能。

结构化资源与任务：[queued-jobs.json](queued-jobs.json)；复现代码与哈希位于code/及code-sha256.json。原始bootstrap、缓存、排障日志和临时操作脚本留在/tmp/laya16/。
