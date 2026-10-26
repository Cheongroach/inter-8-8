# H20 和昇腾怎样交换张量：已经实测到哪一步

2026-10-19 的通信探针使用一张 H20、一张 910B4，分配在两个不同节点。设备张量搬到 CPU，经 TCP 或 Gloo 聚合，再写回各自设备，全部测试通过。完整平台标识、退出与原始事件保留在[原报告](report.md)。

它回答“这对节点能不能正确交换数据”，不回答“完整模型混训更快吗”或“训练中断能恢复吗”。标准 NCCL / HCCL 路径不提供混合 GPU / NPU 设备 collective，本次采用主机中转。

## 正确性怎样验证

两侧 FP32 输入分别为 `arange % 127` 和 `arange % 127 + 1`，独立预期平均为 `arange % 127 + 0.5`。这些值可由 FP32 精确表示，检查逐元素相等，没有放宽容差。

TCP 路径由 NPU 主机发送到 H20 主机，平均后回传，检查 sequence、长度、SHA256、CPU 结果和设备写回。Gloo 路径在 CPU 上 `all_reduce(SUM)` 后除以二，再写回 CUDA / NPU，双方检查并 barrier。

各测 4 KiB、1 MiB、16 MiB、64 MiB、256 MiB，每档三次。每侧 15 次 TCP、15 次 Gloo 成功，双方输出 `probe_passed`，pod 退出 0。节点不同由调度 Bind 事件确认，[node-bindings.json](node-bindings.json)保留证据；不只依赖容器 IP。

## 两列时间为什么不能当成纯网络加速比

| 单侧张量量 | TCP 请求、平均、回传，中位 ms | CPU Gloo all-reduce，中位 ms |
|---|---:|---:|
| 4 KiB | 2.67 | 1.94 |
| 1 MiB | 16.71 | 3.80 |
| 16 MiB | 85.76 | 23.81 |
| 64 MiB | 354.65 | 66.47 |
| 256 MiB | 1,421.25 | 344.19 |

TCP 计时包括校验、序列化、缓冲、平均与回传；Gloo 只计 collective，取双方较慢者后作三次中位。设备拷贝另外记录。操作范围不同，不能把两列比值写成 Gloo 的纯网络加速。

256 MiB Gloo 三次为 432.55、344.19、255.44 ms，波动明显。按单侧 payload/time 算约 6.24 Gbit/s，但这不是网卡总流量或物理带宽，更不是稳定 P95。TCP 请求与回传合计 2,120,245,248 字节，约 1.97 GiB，不含协议头。

## 尚未覆盖的边界

没有测试直接 NCCL/HCCL 互操作、跨机房、长期竞争、完整模型更新、分片重排、训练反向、收敛或故障恢复。探针由 worker 直接传 payload，没有经过 Agent 或共享存储。

大模型更新还需要参数区间、版本、完整快照、优化器和数据进度一致。后续历史训练确已运行，见[组合结论](../experiment-conclusions-friendly.md)；不能沿用原探针日期的“尚未训练”作为当前状态。

证据：[manifest](manifest.json)、[CUDA 事件](cuda.events.jsonl)、[NPU 事件](npu.events.jsonl)、[探针源码](../communication_probe.py)。源码原 SHA256 为 `01e7c3c4be30b6b09c86d9f6dae9e84453f4ec92c6414bc7a02d07bffa240175`。公共导出中的原代码指纹记录的是脱敏前快照，导出文件另有公共校验清单。
