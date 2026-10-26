# 通信可行性：连得上，不等于训练有收益

这是 2026-10-19 [技术原稿](communication-feasibility.md)的阅读说明，结合后续结果说明当前边界。

两个设备共同训练需要三件事：张量后端支持操作、实际节点能连接、交换的训练状态具有同一含义。本次已验证一对 H20 / 910B4 节点的 CPU 中转 TCP / Gloo 聚合与设备写回。标准 NCCL / HCCL 没有混合设备 collective，不能直接把 GPU 与 NPU 张量放入同一个原生 all-reduce。

## 实际路径

```text
H20 参数分片 → H20 主机 CPU
                       ↕ TCP / CPU Gloo
NPU 参数分片 ← NPU 主机 CPU
```

机内各自使用 NCCL / HCCL。机间发送约定的 CPU 表示，并检查参数区间、版本、形状和类型，然后聚合、写回、确认整轮完成。CUDA 和 NPU 计算轨迹不保证逐 bit 一致，传输正确不能代替梯度与质量验收。

Gloo 是主机集合通信库；独立 TCP 是字节传输协议，两者部署和异常处理不同。FSDP rank 相同也不保证分片含义相同，需要参数清单、逻辑偏移、长度一致；岛大小不同时还需要重分片。

| 路径 | 已有证据 | 当前不能推断 |
|---|---|---|
| CPU Gloo / TCP | 两节点 4 KiB–256 MiB、各三次、聚合写回正确 | 任意节点、完整模型、长时间稳定 |
| 各设备组内 NCCL / HCCL | 后续同构 / 混合历史训练完成 | 原生跨 CUDA/NPU collective |
| 共享存储交换 | 可设计提交文件协议 | 两侧实际吞吐、原子提交与恢复已通过 |

## 如果只允许共享存储

可以把每轮更新写成文件，再由聚合者发布新版本。但必须有不可变 round / base version、分片 manifest、校验、完整提交标记和重试去重，读者只读完整版本。恢复还要一起保存模型、优化器、随机数和数据游标。这是原方案设计，当前没有这条共享存储训练恢复验收。

异步重叠也需要独立快照，不能一边传旧参数一边继续修改同一缓冲。第一版阻塞同步是为明确状态边界，并非一定最快。

## 大更新的最小账本

FP32 每参数四字节。下表是理想单向 10 Gbit/s 满速传输估算，未测量：

| 参数数 | 一份 FP32 更新 | \(8S/B\) 理想秒 |
|---|---:|---:|
| 8.2B | 32.8 GB | 26.24 |
| 14.8B | 59.2 GB | 47.36 |
| 32.8B | 131.2 GB | 104.96 |

没有计入双向聚合、设备拷贝、哈希、缓冲、慢组等待和写回，不能据它承诺实际同步时长。CPU 快照还消耗主机内存，可能比网络先成为限制。

prime-diloco 有 CPU 外部同步与设备写回代码；torchft 提供可复用组件，但它们不等于已在本 NPU 环境验收。来源：[PyTorch 后端表](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/docs/source/distributed.rst)、[TorchNPU 注册](https://github.com/Ascend/pytorch/blob/master/torch_npu/_init/registry/distributed.py)、[prime-diloco](https://github.com/PrimeIntellect-ai/prime-diloco/blob/main/src/zeroband/diloco.py)、[torchft](https://github.com/pytorch/torchft)。

实测细节见[通信报告](communication-results/report-friendly.md)。后续[4＋4 实验](experiment-conclusions-friendly.md)说明 CPU 中转能运行，却未证明混合训练胜过更快同构；训练性能、质量和恢复应分别验收。
