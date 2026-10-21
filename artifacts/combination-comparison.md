# 混合组合对照

从双方完成的原始事件聚合，空白表示没有完成证据。

## 固定数据的全参数训练

每个 case 为 8 个更新、全局每步 16 样本。时间含每步等待、拷贝与交换；取两岛各自累计值的较大者。不是完整 case 墙钟，未计入 case 间加载、评测与日志。

| 模型 | 方案 | 累计步秒 | H20×8 秒 | 910B4×8 秒 | 相对更快同构 |
|---|---|---:|---:|---:|---:|
| q25 | sync-equal | 182.09 | 14.16 | 14.97 | 12.86× |
| q25 | sync-h3-n1 | 188.96 | 14.16 | 14.97 | 13.34× |
| q25 | sync-h1-n3 | 182.58 | 14.16 | 14.97 | 12.89× |
| q25 | local-h4 | 76.93 | 14.16 | 14.97 | 5.43× |
| q25 | local-h8 | 63.36 | 14.16 | 14.97 | 4.47× |
| q35 | sync-equal | 101.63 | 19.20 | 18.70 | 5.44× |
| q35 | sync-h3-n1 | 85.53 | 19.20 | 18.70 | 4.57× |
| q35 | sync-h1-n3 | 88.44 | 19.20 | 18.70 | 4.73× |
| q35 | local-h4 | 32.27 | 19.20 | 18.70 | 1.73× |
| q35 | local-h8 | 24.58 | 19.20 | 18.70 | 1.31× |

Local SGD 保留各岛 AdamW 状态，改变优化轨迹，不能据短程验证 loss 声称同质量收益。

## 真实生成与 CE 更新分工

每个 case 为 4 个更新，4 卡生成、4 卡训练，每生成卡 batch 2，输出上限 128。整体时间取双方最慢 rank 的较大者；包含最后一次完整权重发布。

| 模型 | 布局 | 发布实现 | 执行 | 间隔 | 秒 | 监督 token | token/s |
|---|---|---|---|---:|---:|---:|---:|
| q25 | H生成/N训练 | pipeline | overlap | 1 | 231.38 | 4096 | 17.70 |
| q25 | H生成/N训练 | pipeline | overlap | 2 | 127.18 | 4091 | 32.17 |
| q25 | H生成/N训练 | pipeline | overlap | 4 | 77.59 | 4096 | 52.79 |
| q25 | H生成/N训练 | pipeline | serial | 1 | 239.38 | 4096 | 17.11 |
| q25 | H生成/N训练 | pipeline | serial | 2 | 130.79 | 4091 | 31.28 |
| q25 | H生成/N训练 | pipeline | serial | 4 | 76.82 | 4096 | 53.32 |
| q25 | H生成/N训练 | pipeline-buckets | frozen | 0 | 20.36 | 4096 | 201.14 |
| q25 | H生成/N训练 | pipeline-buckets | overlap | 4 | 84.78 | 4096 | 48.31 |
| q25 | H生成/N训练 | pipeline-buckets | serial | 4 | 95.50 | 4096 | 42.89 |
| q25 | N生成/H训练 | pipeline | overlap | 1 | 241.46 | 4096 | 16.96 |
| q25 | N生成/H训练 | pipeline | overlap | 2 | 123.04 | 4096 | 33.29 |
| q25 | N生成/H训练 | pipeline | overlap | 4 | 68.16 | 4096 | 60.09 |
| q25 | N生成/H训练 | pipeline | serial | 1 | 252.37 | 4096 | 16.23 |
| q25 | N生成/H训练 | pipeline | serial | 2 | 124.79 | 4095 | 32.82 |
| q25 | N生成/H训练 | pipeline | serial | 4 | 67.44 | 4096 | 60.74 |
| q25 | N生成/H训练 | pipeline-buckets | frozen | 0 | 11.50 | 4096 | 356.27 |
| q25 | N生成/H训练 | pipeline-buckets | overlap | 4 | 75.24 | 4096 | 54.44 |
| q25 | N生成/H训练 | pipeline-buckets | serial | 4 | 77.97 | 4096 | 52.53 |
| q25 | H20同构 | pipeline-shared-parameters | frozen | 0 | 10.92 | 4096 | 374.93 |
| q25 | H20同构 | pipeline-shared-parameters | overlap | 4 | 64.73 | 4096 | 63.27 |
| q25 | H20同构 | pipeline-shared-parameters | serial | 4 | 87.56 | 4096 | 46.78 |
| q25 | H20同构 | pipeline-homogeneous | overlap | 1 | 236.08 | 4096 | 17.35 |
| q25 | H20同构 | pipeline-homogeneous | overlap | 2 | 124.34 | 4096 | 32.94 |
| q25 | H20同构 | pipeline-homogeneous | overlap | 4 | 68.19 | 4096 | 60.07 |
| q25 | H20同构 | pipeline-homogeneous | serial | 1 | 246.33 | 4096 | 16.63 |
| q25 | H20同构 | pipeline-homogeneous | serial | 2 | 126.99 | 4091 | 32.22 |
| q25 | H20同构 | pipeline-homogeneous | serial | 4 | 66.87 | 4096 | 61.25 |
| q25 | H20同构 | pipeline-shared-buckets | frozen | 0 | 11.38 | 4096 | 359.91 |
| q25 | H20同构 | pipeline-shared-buckets | overlap | 4 | 81.15 | 4096 | 50.47 |
| q25 | H20同构 | pipeline-shared-buckets | serial | 4 | 95.56 | 4096 | 42.86 |
| q25 | 910B4同构 | pipeline-shared-parameters | frozen | 0 | 23.01 | 4096 | 177.97 |
| q25 | 910B4同构 | pipeline-shared-parameters | overlap | 4 | 70.77 | 4096 | 57.88 |
| q25 | 910B4同构 | pipeline-shared-parameters | serial | 4 | 74.57 | 4096 | 54.93 |
| q25 | 910B4同构 | pipeline-shared-buckets | frozen | 0 | 24.22 | 4096 | 169.12 |
| q25 | 910B4同构 | pipeline-shared-buckets | overlap | 4 | 71.96 | 4096 | 56.92 |
| q25 | 910B4同构 | pipeline-shared-buckets | serial | 4 | 72.37 | 4096 | 56.60 |
| q35 | H生成/N训练 | pipeline | overlap | 1 | 117.04 | 3975 | 33.96 |
| q35 | H生成/N训练 | pipeline | overlap | 2 | 70.45 | 4047 | 57.44 |
| q35 | H生成/N训练 | pipeline | overlap | 4 | 44.24 | 4096 | 92.58 |
| q35 | H生成/N训练 | pipeline | serial | 1 | 143.84 | 3975 | 27.63 |
| q35 | H生成/N训练 | pipeline | serial | 2 | 70.23 | 4047 | 57.62 |
| q35 | H生成/N训练 | pipeline | serial | 4 | 48.36 | 4096 | 84.69 |
| q35 | H生成/N训练 | pipeline-buckets | frozen | 0 | 21.95 | 4096 | 186.60 |
| q35 | H生成/N训练 | pipeline-buckets | overlap | 4 | 52.79 | 4096 | 77.59 |
| q35 | H生成/N训练 | pipeline-buckets | serial | 4 | 84.61 | 4096 | 48.41 |
| q35 | N生成/H训练 | pipeline | frozen | 0 | 19.39 | 4096 | 211.28 |
| q35 | N生成/H训练 | pipeline | overlap | 1 | 139.43 | 3933 | 28.21 |
| q35 | N生成/H训练 | pipeline | overlap | 2 | 79.29 | 4050 | 51.08 |
| q35 | N生成/H训练 | pipeline | overlap | 4 | 49.27 | 4096 | 83.14 |
| q35 | N生成/H训练 | pipeline | serial | 1 | 150.42 | 3933 | 26.15 |
| q35 | N生成/H训练 | pipeline | serial | 2 | 82.40 | 4050 | 49.15 |
| q35 | N生成/H训练 | pipeline | serial | 4 | 54.14 | 4096 | 75.66 |
| q35 | N生成/H训练 | pipeline-buckets | frozen | 0 | 19.31 | 4096 | 212.11 |
| q35 | N生成/H训练 | pipeline-buckets | overlap | 4 | 54.84 | 4096 | 74.69 |
| q35 | N生成/H训练 | pipeline-buckets | serial | 4 | 76.11 | 4096 | 53.81 |
| q35 | H20同构 | pipeline-shared-parameters | frozen | 0 | 21.78 | 4096 | 188.02 |
| q35 | H20同构 | pipeline-shared-parameters | overlap | 4 | 50.65 | 4096 | 80.87 |
| q35 | H20同构 | pipeline-shared-parameters | serial | 4 | 73.85 | 4096 | 55.46 |
| q35 | H20同构 | pipeline-homogeneous | overlap | 1 | 141.28 | 3948 | 27.94 |
| q35 | H20同构 | pipeline-homogeneous | overlap | 2 | 84.26 | 4096 | 48.61 |
| q35 | H20同构 | pipeline-homogeneous | overlap | 4 | 52.76 | 4096 | 77.63 |
| q35 | H20同构 | pipeline-homogeneous | serial | 1 | 147.28 | 3948 | 26.81 |
| q35 | H20同构 | pipeline-homogeneous | serial | 2 | 77.70 | 4096 | 52.72 |
| q35 | H20同构 | pipeline-homogeneous | serial | 4 | 56.44 | 4096 | 72.57 |
| q35 | H20同构 | pipeline-shared-buckets | frozen | 0 | 22.50 | 4096 | 182.03 |
| q35 | H20同构 | pipeline-shared-buckets | overlap | 4 | 62.28 | 4096 | 65.76 |
| q35 | H20同构 | pipeline-shared-buckets | serial | 4 | 75.15 | 4096 | 54.50 |
| q35 | 910B4同构 | pipeline-shared-parameters | frozen | 0 | 19.33 | 4096 | 211.85 |
| q35 | 910B4同构 | pipeline-shared-parameters | overlap | 4 | 40.70 | 4096 | 100.65 |
| q35 | 910B4同构 | pipeline-shared-parameters | serial | 4 | 61.95 | 4096 | 66.12 |
| q35 | 910B4同构 | pipeline-shared-buckets | frozen | 0 | 20.07 | 4096 | 204.13 |
| q35 | 910B4同构 | pipeline-shared-buckets | overlap | 4 | 48.71 | 4096 | 84.10 |
| q35 | 910B4同构 | pipeline-shared-buckets | serial | 4 | 71.78 | 4096 | 57.06 |

frozen/间隔0表示固定教师，只传轨迹，不回传权重。它适用于离线数据生成，训练语义不同于周期刷新策略。

每个配置只有一次短程实验；同构采用同样共享 SDPA/FSDP 实现，未用 vLLM/FLA。这里验证系统可运行和开销，不能代表硬件最优、最终模型质量或完整 PPO/GRPO。

参数发布包含版本、形状及 SHA256 校验。原始逐参数实现与 CPU 分块发布分别记录，不能跨实现择优拼出异构胜利。
