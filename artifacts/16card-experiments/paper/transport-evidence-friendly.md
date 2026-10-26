# 通信日志怎样证明实际用了 RDMA 或 Socket

本文件解释[原始传输摘录](transport-evidence.txt)，不改动日志。它回答“本次数据通道是什么”，不能单独回答“哪个 kernel 最慢”或“通信占总耗时多少”。

H20×16 获得两台不同物理 host，各 8 卡。跨机对照必须先证明不是同机多容器，再核对 NCCL 实际选择的通道。

| 日志内容 | 能说明什么 | 不能说明什么 |
|---|---|---|
| 两个不同 host 的运行记录 | 本次确实跨物理机器 | 所有后来任务都跨同样机器 |
| `via NET/IB/.../GDRDMA` | 对应 NCCL 通道使用本次 RDMA 数据路径 | 所有阶段均无 CPU 开销 |
| `Using network Socket` 与 Socket 通道 | 强制配置实际进入 Socket 路径 | CPU/Gloo 参数平均也是同一条 NCCL 路径 |
| RoCE / IB 设备列举 | 设备在环境中可见 | 这些设备一定承担本次通信 |
| 两个 pod 退出 0 | 任务正常结束 | 梯度与官方参考数值等价、达到质量目标 |

CUDA 对照通过 `NCCL_IB_DISABLE=1` 与 `NCCL_NET=Socket` 强制网络。CPU islands 机内仍用 NCCL，机间交换主机张量，包含设备到 CPU 搬运、Gloo / TCP 与写回，不是简单把同一个 collective 换一种网络。

NPU 的 CPU/TCP 路径也不能称为已经验证“原生 HCCL 关闭 RDMA”。本次缺少新版 NPU×16 结果，见[终态](../cancellation-20261025-friendly.md)。

对应耗时来自[training-metrics.json](training-metrics.json)，对应事件来自[events.jsonl](events.jsonl)。读法见[主报告](report-friendly.md)：42 次短程计时支持本次设置下的系统差距，不提供跨日期统计、质量曲线或原生 NCCL/HCCL 混合互操作证据。
