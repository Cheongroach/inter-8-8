# H20 与 910B4 跨硬件通信实测

实验日期：2026-10-19，约 01:47–01:52（北京时间）。控制面 `cn`，资源组 `search_douyin_life_nlp / 2079`，两侧均在 `cloudnative-hl / 20`。

**结论：本次分配的 H20 和 910B4 worker 位于两个不同 Kubernetes 节点，直接 IPv6 TCP 传输及 CPU Gloo all-reduce 均通过。设备张量搬到 CPU、聚合、写回各自设备的闭环已实测可行。**

本实验没有验证直接 NCCL/HCCL 互操作、跨 HL/LF、FSDP 分片重排、训练反向、DiLoCo 收敛或故障恢复。

## 资源与节点证据

| 项目 | H20 | 910B4 |
|---|---|---|
| 任务 | [695eeb0b1a5d04f6]([internal-reference-omitted]) | [bd1ddb781aa1010a]([internal-reference-omitted]) |
| Trial | 434380397 | 434380419 |
| Pod SID | 5368142285 | 5368147679 |
| 申请卡数 | 1 | 1 |
| 实际设备名 | NVIDIA H20 | Ascend910B4-1 |
| PyTorch | 2.9.1 | 2.7.1+cpu |
| TorchNPU | 无 | 2.7.1 |
| 镜像 VID | d7iu6hbc77u4ebjdshsg | d6nakm3c77u7jl17ea90 |
| Kubernetes node | compute-node-0d54834c | compute-node-30425be1 |
| Pod 退出码 | 0 | 0 |

两侧最终 run 状态为 `done`、pod 状态为 3，正常结束并释放资源。实际测试持续数十秒；资源占用窗口另包含平台初始化与等待配对。

不同 node 由 Kelemetry 完整 trace 中的 `godel-binder / Bind` 事件确认，并非仅依据 hostname 或容器 IP：

- [H20 调度记录]([internal-reference-omitted])
- [910B4 调度记录]([internal-reference-omitted])
- 已筛选保存的证据：[node-bindings.json](node-bindings.json)。

## 做了哪些验证

每侧使用实际设备 FP32 张量，已知输入分别为 `arange % 127` 与 `arange % 127 + 1`，独立参考平均值为 `arange % 127 + 0.5`。全部值可由 FP32 精确表示，直接逐元素相等检查，无容差放宽。

1. **独立 TCP 数据面**：NPU → CPU → IPv6 TCP → H20 主机 CPU，与 H20 本地输入求平均，返回 NPU 主机，双方写回设备。每次检查 sequence、长度、SHA256、CPU 聚合结果和设备写回结果。
2. **CPU Gloo**：同一对 worker 通过另一分配端口创建 Gloo 进程组，CPU FP32 `all_reduce(SUM)` 后除以 2，再写回 CUDA/NPU。双方检查结果并完成 barrier。
3. 两条路径各测 4 KiB、1 MiB、16 MiB、64 MiB、256 MiB，每档 3 次；每侧 15 次 TCP 和 15 次 Gloo 均成功，最终双方输出 `probe_passed`。

TCP 累计请求与回传 payload 为 **2,120,245,248 bytes，约 1.97 GiB**，不含协议头。数据经 worker 间直接连接传输；Agent 只下发对端地址，没有代理 payload，也未使用共享存储。

源码：[communication_probe.py](../communication_probe.py)；提交的源码 SHA256：
`01e7c3c4be30b6b09c86d9f6dae9e84453f4ec92c6414bc7a02d07bffa240175`。

## 耗时

下表取每档三次的中位数。Gloo 每次取两个 rank 中较慢者，再计算中位数。

| 单侧张量 payload | TCP 请求、平均与回传 | CPU Gloo all-reduce |
|---|---:|---:|
| 4 KiB | 2.67 ms | 1.94 ms |
| 1 MiB | 16.71 ms | 3.80 ms |
| 16 MiB | 85.76 ms | 23.81 ms |
| 64 MiB | 354.65 ms | 66.47 ms |
| 256 MiB | 1,421.25 ms | 344.19 ms |

256 MiB 的 Gloo 三次耗时为 432.55、344.19、255.44 ms，变化较明显，不能将三次结果当作稳定 p95 或链路容量。Gloo 仅按 payload/time 计算的中位有效速率约 6.24 Gbit/s，这不是 NIC 实际总流量或物理带宽。

TCP 计时包含 SHA256、序列化、接收缓冲区、CPU 平均及回传；Gloo 计时仅覆盖 `all_reduce`。两列用于证明路径和报告本实现成本，不能据此宣称 Gloo 相对 TCP 的纯网络加速倍数。设备搬运在 TCP 事件中单独记录；上述耗时没有将设备搬运混进网络计时。

当前未采集 NIC 总计数、CPU profiling、RDMA 路径、多连接竞争或长时间稳定性。也未直接发送完整 8B 模型约 32.8 GB 的 FP32 更新，不能据本表承诺大模型同步时长。

## 提交与证据链

两份最终 payload 通过 schema/dry-run 后分别预检，顶层 `data.status=success`，没有 warn/hint；保留式合并 `request_context.precheck_sid` 后，在 30 分钟内提交。预检 error 会禁止提交。

| 操作 | H20 | 910B4 |
|---|---|---|
| precheck SID | 8780255a-f0fc-4954-8a10-31ff83301821 | 1dc8c2f5-341b-49ce-a213-4e73ed49292e |
| create request ID | 20261019014738FC24C9F2F9222124CAE3 | 20261019014741EC972404BC32CB1FB5C5 |

运行证据按 `runs get → run-trial-pods list → run-trial-local-logs get` 获取 log-proxy stdout/stderr，并由 `run-trials exit-info` 确认 pod 正常退出；机读记录如下：

- [manifest.json](manifest.json)：资源、版本、任务、预检、退出状态。
- [cuda.events.jsonl](cuda.events.jsonl)、[npu.events.jsonl](npu.events.jsonl)：从实际 stdout 提取的完整探针事件。

平台初始化日志有 pip/HDFS 权限提示，但之后探针成功运行且双方退出码为 0。这些提示没有作为网络失败证据。额外身份查询到达时任务已经退出，未得到结果；节点身份使用上述调度绑定事件独立补齐。

## 对方案的影响

“CPU 中转后跨 H20/昇腾通信”已从源码推断推进到本环境实测，选定这对 worker 的 PyTorch 版本差异也未阻碍 CPU Gloo 的本次操作。

下一阶段的工程风险集中在真实模型训练与分片：保持模型和逻辑参数区间一致、控制 CPU 状态内存、构造完整更新快照、按轮次提交、保存/恢复优化器和数据进度。通信可达性与基础张量聚合已通过；这些训练能力仍需另行验证。
