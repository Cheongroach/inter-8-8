# H20＋910B4 实测记录

本报告由已导出的 worker JSONL 生成。四个最终任务均完成；本表按rank0记录展示，跨组比较应使用combination-comparison.md中全部八个rank最慢时间。历史失败/缺失项不能当作零耗时。

模型为 Qwen2.5-7B-Instruct 与 Qwen3.5-4B 文本主干，GSM8K 输入。共同实现采用 Transformers 5.2 SDPA 和 FSDP，全参数 FP32 master/AdamW、BF16 计算。
生成/训练分工使用真实生成回答的 CE 更新，是系统预验，不是 PPO/GRPO 或最终质量验证。

## 生成

| 记录 | 每卡 batch | 输出上限 | 组 token/s |
|---|---:|---:|---:|
| npu8/q25/baseline | 1 | 64 | 440.43 |
| npu8/q25/baseline | 1 | 128 | 443.01 |
| npu8/q25/baseline | 4 | 64 | 1381.89 |
| npu8/q25/baseline | 4 | 128 | 1389.25 |
| npu8/q25/baseline | 8 | 64 | 2491.34 |
| npu8/q25/baseline | 8 | 128 | 2553.76 |
| npu8/q35/baseline | 1 | 64 | 121.76 |
| npu8/q35/baseline | 1 | 128 | 259.77 |
| npu8/q35/baseline | 4 | 64 | 845.15 |
| npu8/q35/baseline | 4 | 128 | 872.88 |
| npu8/q35/baseline | 8 | 64 | 1434.70 |
| npu8/q35/baseline | 8 | 128 | 1491.53 |
| cuda/q25/profile | 1 | 64 | 92.69 |
| cuda/q25/profile | 1 | 128 | 189.76 |
| cuda/q25/profile | 4 | 64 | 374.58 |
| cuda/q25/profile | 4 | 128 | 577.86 |
| cuda/q25/profile | 8 | 64 | 936.78 |
| cuda/q25/profile | 8 | 128 | 1281.10 |
| cuda/q35/profile | 1 | 64 | 65.07 |
| cuda/q35/profile | 1 | 128 | 97.29 |
| cuda/q35/profile | 4 | 64 | 250.19 |
| cuda/q35/profile | 4 | 128 | 351.43 |
| cuda/q35/profile | 8 | 64 | 591.01 |
| cuda/q35/profile | 8 | 128 | 712.56 |
| npu/q25/profile | 1 | 64 | 220.46 |
| npu/q25/profile | 1 | 128 | 222.90 |
| npu/q25/profile | 4 | 64 | 686.51 |
| npu/q25/profile | 4 | 128 | 695.26 |
| npu/q25/profile | 8 | 64 | 1267.23 |
| npu/q25/profile | 8 | 128 | 1297.28 |
| npu/q35/profile | 1 | 64 | 115.67 |
| npu/q35/profile | 1 | 128 | 129.74 |
| npu/q35/profile | 4 | 64 | 403.15 |
| npu/q35/profile | 4 | 128 | 431.30 |
| npu/q35/profile | 8 | 64 | 724.34 |
| npu/q35/profile | 8 | 128 | 755.40 |
| cuda8/q25/baseline | 1 | 64 | 214.08 |
| cuda8/q25/baseline | 1 | 128 | 310.26 |
| cuda8/q25/baseline | 4 | 64 | 884.70 |
| cuda8/q25/baseline | 4 | 128 | 1141.42 |
| cuda8/q25/baseline | 8 | 64 | 1833.26 |
| cuda8/q25/baseline | 8 | 128 | 2436.52 |
| cuda8/q35/baseline | 1 | 64 | 135.59 |
| cuda8/q35/baseline | 1 | 128 | 189.29 |
| cuda8/q35/baseline | 4 | 64 | 509.38 |
| cuda8/q35/baseline | 4 | 128 | 770.76 |
| cuda8/q35/baseline | 8 | 64 | 1130.18 |
| cuda8/q35/baseline | 8 | 128 | 1320.33 |

每个配置三次运行，排除第一次；剩余两次样本不同且 prompt 长度不同。未增加专用推理服务、连续批处理或 Qwen3.5 FLA kernel，因此不是硬件最优值。

## 全参数训练

| 记录 | case | 有效计时步数 | 监督 token/s | 组墙钟秒 | 跨岛交换秒（rank 0） |
|---|---|---:|---:|---:|---:|
| cuda/q25/paired | local-h4 | 7 | 92.01 | 73.392 | 67.674 |
| cuda/q25/paired | local-h8 | 7 | 113.33 | 59.587 | 54.322 |
| cuda/q25/paired | sync-equal | 7 | 47.16 | 143.198 | 127.306 |
| cuda/q25/paired | sync-h1-n3 | 7 | 23.53 | 158.969 | 147.490 |
| cuda/q25/paired | sync-h3-n1 | 7 | 64.33 | 163.542 | 147.400 |
| cuda/q25/profile | profile-b1 | 3 | 935.17 | 1.725 | 0.000 |
| cuda/q25/profile | profile-b2 | 3 | 874.98 | 3.823 | 0.000 |
| cuda/q25/profile | profile-b4 | 3 | 1084.32 | 5.623 | 0.000 |
| cuda/q35/paired | local-h4 | 7 | 246.87 | 27.779 | 14.490 |
| cuda/q35/paired | local-h8 | 7 | 338.37 | 20.268 | 6.698 |
| cuda/q35/paired | sync-equal | 7 | 95.27 | 71.983 | 54.604 |
| cuda/q35/paired | sync-h1-n3 | 7 | 50.09 | 75.630 | 50.643 |
| cuda/q35/paired | sync-h3-n1 | 7 | 147.58 | 72.434 | 47.184 |
| cuda/q35/profile | profile-b1 | 3 | 324.42 | 5.052 | 0.000 |
| cuda/q35/profile | profile-b2 | 3 | 445.38 | 7.618 | 0.000 |
| cuda/q35/profile | profile-b4 | 3 | 1036.79 | 5.984 | 0.000 |
| cuda8/q25/baseline | baseline-b2 | 7 | 1497.32 | 9.351 | 0.000 |
| cuda8/q35/baseline | baseline-b2 | 7 | 1060.09 | 13.419 | 0.000 |
| npu/q25/paired | local-h4 | 7 | 98.96 | 73.244 | 33.125 |
| npu/q25/paired | local-h8 | 7 | 120.83 | 59.985 | 16.261 |
| npu/q25/paired | sync-equal | 7 | 50.62 | 143.189 | 110.852 |
| npu/q25/paired | sync-h1-n3 | 7 | 64.54 | 158.984 | 111.171 |
| npu/q25/paired | sync-h3-n1 | 7 | 21.28 | 163.557 | 111.010 |
| npu/q25/profile | profile-b1 | 3 | 130.52 | 12.358 | 0.000 |
| npu/q25/profile | profile-b2 | 3 | 164.07 | 20.387 | 0.000 |
| npu/q25/profile | profile-b4 | 3 | 384.37 | 15.863 | 0.000 |
| npu/q35/paired | local-h4 | 7 | 243.38 | 30.270 | 16.536 |
| npu/q35/paired | local-h8 | 7 | 326.02 | 22.597 | 9.496 |
| npu/q35/paired | sync-equal | 7 | 102.27 | 72.035 | 54.824 |
| npu/q35/paired | sync-h1-n3 | 7 | 138.06 | 75.595 | 47.753 |
| npu/q35/paired | sync-h3-n1 | 7 | 48.79 | 72.458 | 48.829 |
| npu/q35/profile | profile-b1 | 3 | 357.94 | 4.579 | 0.000 |
| npu/q35/profile | profile-b2 | 3 | 646.52 | 5.248 | 0.000 |
| npu/q35/profile | profile-b4 | 3 | 927.86 | 6.686 | 0.000 |
| npu8/q25/baseline | baseline-b2 | 7 | 1230.40 | 11.379 | 0.000 |
| npu8/q35/baseline | baseline-b2 | 7 | 1052.68 | 13.513 | 0.000 |

profile 的 microbatch 1/2/4 使用不同的全局 batch，不能比较端点 loss 的高低。paired 的同一步固定 16 样本；岛内 token/s 表示该岛监督 token，组合需使用两岛 token 总和与整体时间。
上表排除首个更新便于观察热身后计算；判断整体收益还需完整事件，包括首步、等待、传输及权重发布。Local SGD 的 Adam 状态留在本岛，参数平均不是 DiLoCo。

## 生成/训练分工

| 记录 | 执行 | 发布间隔 | 总秒 | 监督 token |
|---|---|---:|---:|---:|
| npu8/q25/pipeline-shared-parameters/trainer | serial | 4 | 74.515 | 4096 |
| npu8/q25/pipeline-shared-parameters/trainer | overlap | 4 | 69.989 | 4096 |
| npu8/q25/pipeline-shared-parameters/trainer | frozen | 0 | 22.874 | 4096 |
| npu8/q25/pipeline-shared-parameters/producer | serial | 4 | 74.571 | 4096 |
| npu8/q25/pipeline-shared-parameters/producer | overlap | 4 | 70.766 | 4096 |
| npu8/q25/pipeline-shared-parameters/producer | frozen | 0 | 23.015 | 4096 |
| npu8/q25/pipeline-shared-buckets/trainer | serial | 4 | 71.993 | 4096 |
| npu8/q25/pipeline-shared-buckets/trainer | overlap | 4 | 71.608 | 4096 |
| npu8/q25/pipeline-shared-buckets/trainer | frozen | 0 | 23.532 | 4096 |
| npu8/q25/pipeline-shared-buckets/producer | serial | 4 | 72.366 | 4096 |
| npu8/q25/pipeline-shared-buckets/producer | overlap | 4 | 71.961 | 4096 |
| npu8/q25/pipeline-shared-buckets/producer | frozen | 0 | 24.219 | 4096 |
| npu8/q35/pipeline-shared-parameters/trainer | serial | 4 | 61.948 | 4096 |
| npu8/q35/pipeline-shared-parameters/trainer | overlap | 4 | 40.696 | 4096 |
| npu8/q35/pipeline-shared-parameters/trainer | frozen | 0 | 19.306 | 4096 |
| npu8/q35/pipeline-shared-parameters/producer | serial | 4 | 61.922 | 4096 |
| npu8/q35/pipeline-shared-parameters/producer | overlap | 4 | 40.695 | 4096 |
| npu8/q35/pipeline-shared-parameters/producer | frozen | 0 | 19.334 | 4096 |
| npu8/q35/pipeline-shared-buckets/trainer | serial | 4 | 71.781 | 4096 |
| npu8/q35/pipeline-shared-buckets/trainer | overlap | 4 | 48.588 | 4096 |
| npu8/q35/pipeline-shared-buckets/trainer | frozen | 0 | 20.055 | 4096 |
| npu8/q35/pipeline-shared-buckets/producer | serial | 4 | 71.742 | 4096 |
| npu8/q35/pipeline-shared-buckets/producer | overlap | 4 | 48.706 | 4096 |
| npu8/q35/pipeline-shared-buckets/producer | frozen | 0 | 20.066 | 4096 |
| cuda/q25/pipeline-buckets-cuda | serial | 4 | 95.011 | 4096 |
| cuda/q25/pipeline-buckets-cuda | overlap | 4 | 84.785 | 4096 |
| cuda/q25/pipeline-buckets-cuda | frozen | 0 | 20.364 | 4096 |
| cuda/q25/pipeline-npu | serial | 1 | 251.615 | 4096 |
| cuda/q25/pipeline-npu | overlap | 1 | 240.157 | 4096 |
| cuda/q25/pipeline-npu | serial | 2 | 123.361 | 4095 |
| cuda/q25/pipeline-npu | overlap | 2 | 121.824 | 4096 |
| cuda/q25/pipeline-npu | serial | 4 | 66.893 | 4096 |
| cuda/q25/pipeline-npu | overlap | 4 | 68.025 | 4096 |
| cuda/q25/pipeline-cuda | serial | 1 | 238.548 | 4096 |
| cuda/q25/pipeline-cuda | overlap | 1 | 231.385 | 4096 |
| cuda/q25/pipeline-cuda | serial | 2 | 130.786 | 4091 |
| cuda/q25/pipeline-cuda | overlap | 2 | 127.185 | 4091 |
| cuda/q25/pipeline-cuda | serial | 4 | 76.824 | 4096 |
| cuda/q25/pipeline-cuda | overlap | 4 | 77.594 | 4096 |
| cuda/q25/pipeline-buckets-npu | serial | 4 | 77.104 | 4096 |
| cuda/q25/pipeline-buckets-npu | overlap | 4 | 73.846 | 4096 |
| cuda/q25/pipeline-buckets-npu | frozen | 0 | 10.182 | 4096 |
| cuda/q35/pipeline-buckets-cuda | serial | 4 | 84.297 | 4096 |
| cuda/q35/pipeline-buckets-cuda | overlap | 4 | 52.792 | 4096 |
| cuda/q35/pipeline-buckets-cuda | frozen | 0 | 21.950 | 4096 |
| cuda/q35/pipeline-npu | serial | 1 | 149.741 | 3933 |
| cuda/q35/pipeline-npu | overlap | 1 | 139.301 | 3933 |
| cuda/q35/pipeline-npu | serial | 2 | 81.397 | 4050 |
| cuda/q35/pipeline-npu | overlap | 2 | 79.196 | 4050 |
| cuda/q35/pipeline-npu | serial | 4 | 53.224 | 4096 |
| cuda/q35/pipeline-npu | overlap | 4 | 49.226 | 4096 |
| cuda/q35/pipeline-npu | frozen | 0 | 19.264 | 4096 |
| cuda/q35/pipeline-cuda | serial | 1 | 143.100 | 3975 |
| cuda/q35/pipeline-cuda | overlap | 1 | 117.035 | 3975 |
| cuda/q35/pipeline-cuda | serial | 2 | 70.230 | 4047 |
| cuda/q35/pipeline-cuda | overlap | 2 | 70.452 | 4047 |
| cuda/q35/pipeline-cuda | serial | 4 | 48.364 | 4096 |
| cuda/q35/pipeline-cuda | overlap | 4 | 44.242 | 4096 |
| cuda/q35/pipeline-buckets-npu | serial | 4 | 75.165 | 4096 |
| cuda/q35/pipeline-buckets-npu | overlap | 4 | 54.175 | 4096 |
| cuda/q35/pipeline-buckets-npu | frozen | 0 | 18.638 | 4096 |
| npu/q25/pipeline-buckets-cuda | serial | 4 | 95.497 | 4096 |
| npu/q25/pipeline-buckets-cuda | overlap | 4 | 84.623 | 4096 |
| npu/q25/pipeline-buckets-cuda | frozen | 0 | 20.320 | 4096 |
| npu/q25/pipeline-npu | serial | 1 | 252.368 | 4096 |
| npu/q25/pipeline-npu | overlap | 1 | 241.459 | 4096 |
| npu/q25/pipeline-npu | serial | 2 | 124.786 | 4095 |
| npu/q25/pipeline-npu | overlap | 2 | 123.042 | 4096 |
| npu/q25/pipeline-npu | serial | 4 | 67.435 | 4096 |
| npu/q25/pipeline-npu | overlap | 4 | 68.161 | 4096 |
| npu/q25/pipeline-cuda | serial | 1 | 239.378 | 4096 |
| npu/q25/pipeline-cuda | overlap | 1 | 230.476 | 4096 |
| npu/q25/pipeline-cuda | serial | 2 | 130.080 | 4091 |
| npu/q25/pipeline-cuda | overlap | 2 | 126.989 | 4091 |
| npu/q25/pipeline-cuda | serial | 4 | 76.409 | 4096 |
| npu/q25/pipeline-cuda | overlap | 4 | 77.593 | 4096 |
| npu/q25/pipeline-buckets-npu | serial | 4 | 77.975 | 4096 |
| npu/q25/pipeline-buckets-npu | overlap | 4 | 75.235 | 4096 |
| npu/q25/pipeline-buckets-npu | frozen | 0 | 11.497 | 4096 |
| npu/q35/pipeline-buckets-cuda | serial | 4 | 84.609 | 4096 |
| npu/q35/pipeline-buckets-cuda | overlap | 4 | 52.791 | 4096 |
| npu/q35/pipeline-buckets-cuda | frozen | 0 | 21.584 | 4096 |
| npu/q35/pipeline-npu | serial | 1 | 150.421 | 3933 |
| npu/q35/pipeline-npu | overlap | 1 | 139.434 | 3933 |
| npu/q35/pipeline-npu | serial | 2 | 82.395 | 4050 |
| npu/q35/pipeline-npu | overlap | 2 | 79.286 | 4050 |
| npu/q35/pipeline-npu | serial | 4 | 54.138 | 4096 |
| npu/q35/pipeline-npu | overlap | 4 | 49.265 | 4096 |
| npu/q35/pipeline-npu | frozen | 0 | 19.387 | 4096 |
| npu/q35/pipeline-cuda | serial | 1 | 143.842 | 3975 |
| npu/q35/pipeline-cuda | overlap | 1 | 116.913 | 3975 |
| npu/q35/pipeline-cuda | serial | 2 | 70.130 | 4047 |
| npu/q35/pipeline-cuda | overlap | 2 | 70.254 | 4047 |
| npu/q35/pipeline-cuda | serial | 4 | 48.086 | 4096 |
| npu/q35/pipeline-cuda | overlap | 4 | 44.241 | 4096 |
| npu/q35/pipeline-buckets-npu | serial | 4 | 76.113 | 4096 |
| npu/q35/pipeline-buckets-npu | overlap | 4 | 54.837 | 4096 |
| npu/q35/pipeline-buckets-npu | frozen | 0 | 19.311 | 4096 |
| cuda8/q25/pipeline-shared-parameters/trainer | serial | 4 | 86.679 | 4096 |
| cuda8/q25/pipeline-shared-parameters/trainer | overlap | 4 | 64.083 | 4096 |
| cuda8/q25/pipeline-shared-parameters/trainer | frozen | 0 | 9.731 | 4096 |
| cuda8/q25/pipeline-shared-parameters/producer | serial | 4 | 87.559 | 4096 |
| cuda8/q25/pipeline-shared-parameters/producer | overlap | 4 | 64.734 | 4096 |
| cuda8/q25/pipeline-shared-parameters/producer | frozen | 0 | 10.925 | 4096 |
| cuda8/q25/pipeline-homogeneous/trainer | serial | 1 | 246.327 | 4096 |
| cuda8/q25/pipeline-homogeneous/trainer | overlap | 1 | 236.001 | 4096 |
| cuda8/q25/pipeline-homogeneous/trainer | serial | 2 | 126.990 | 4091 |
| cuda8/q25/pipeline-homogeneous/trainer | overlap | 2 | 122.586 | 4096 |
| cuda8/q25/pipeline-homogeneous/trainer | serial | 4 | 65.987 | 4096 |
| cuda8/q25/pipeline-homogeneous/trainer | overlap | 4 | 67.504 | 4096 |
| cuda8/q25/pipeline-homogeneous/producer | serial | 1 | 246.251 | 4096 |
| cuda8/q25/pipeline-homogeneous/producer | overlap | 1 | 236.085 | 4096 |
| cuda8/q25/pipeline-homogeneous/producer | serial | 2 | 126.990 | 4091 |
| cuda8/q25/pipeline-homogeneous/producer | overlap | 2 | 124.342 | 4096 |
| cuda8/q25/pipeline-homogeneous/producer | serial | 4 | 66.874 | 4096 |
| cuda8/q25/pipeline-homogeneous/producer | overlap | 4 | 68.192 | 4096 |
| cuda8/q25/pipeline-shared-buckets/trainer | serial | 4 | 95.072 | 4096 |
| cuda8/q25/pipeline-shared-buckets/trainer | overlap | 4 | 80.025 | 4096 |
| cuda8/q25/pipeline-shared-buckets/trainer | frozen | 0 | 10.487 | 4096 |
| cuda8/q25/pipeline-shared-buckets/producer | serial | 4 | 95.562 | 4096 |
| cuda8/q25/pipeline-shared-buckets/producer | overlap | 4 | 81.150 | 4096 |
| cuda8/q25/pipeline-shared-buckets/producer | frozen | 0 | 11.380 | 4096 |
| cuda8/q35/pipeline-shared-parameters/trainer | serial | 4 | 73.227 | 4096 |
| cuda8/q35/pipeline-shared-parameters/trainer | overlap | 4 | 49.998 | 4096 |
| cuda8/q35/pipeline-shared-parameters/trainer | frozen | 0 | 21.158 | 4096 |
| cuda8/q35/pipeline-shared-parameters/producer | serial | 4 | 73.850 | 4096 |
| cuda8/q35/pipeline-shared-parameters/producer | overlap | 4 | 50.648 | 4096 |
| cuda8/q35/pipeline-shared-parameters/producer | frozen | 0 | 21.785 | 4096 |
| cuda8/q35/pipeline-homogeneous/trainer | serial | 1 | 147.163 | 3948 |
| cuda8/q35/pipeline-homogeneous/trainer | overlap | 1 | 140.565 | 3948 |
| cuda8/q35/pipeline-homogeneous/trainer | serial | 2 | 77.494 | 4096 |
| cuda8/q35/pipeline-homogeneous/trainer | overlap | 2 | 83.464 | 4096 |
| cuda8/q35/pipeline-homogeneous/trainer | serial | 4 | 56.444 | 4096 |
| cuda8/q35/pipeline-homogeneous/trainer | overlap | 4 | 52.764 | 4096 |
| cuda8/q35/pipeline-homogeneous/producer | serial | 1 | 147.284 | 3948 |
| cuda8/q35/pipeline-homogeneous/producer | overlap | 1 | 141.281 | 3948 |
| cuda8/q35/pipeline-homogeneous/producer | serial | 2 | 77.698 | 4096 |
| cuda8/q35/pipeline-homogeneous/producer | overlap | 2 | 84.259 | 4096 |
| cuda8/q35/pipeline-homogeneous/producer | serial | 4 | 56.444 | 4096 |
| cuda8/q35/pipeline-homogeneous/producer | overlap | 4 | 52.763 | 4096 |
| cuda8/q35/pipeline-shared-buckets/trainer | serial | 4 | 74.626 | 4096 |
| cuda8/q35/pipeline-shared-buckets/trainer | overlap | 4 | 62.019 | 4096 |
| cuda8/q35/pipeline-shared-buckets/trainer | frozen | 0 | 22.502 | 4096 |
| cuda8/q35/pipeline-shared-buckets/producer | serial | 4 | 75.153 | 4096 |
| cuda8/q35/pipeline-shared-buckets/producer | overlap | 4 | 62.284 | 4096 |
| cuda8/q35/pipeline-shared-buckets/producer | frozen | 0 | 22.475 | 4096 |

## 已完成的 suite

- npu8/q25/pipeline-shared-parameters/trainer
- npu8/q25/pipeline-shared-parameters/producer
- npu8/q25/baseline
- npu8/q25/pipeline-shared-buckets/trainer
- npu8/q25/pipeline-shared-buckets/producer
- npu8/q35/pipeline-shared-parameters/trainer
- npu8/q35/pipeline-shared-parameters/producer
- npu8/q35/baseline
- npu8/q35/pipeline-shared-buckets/trainer
- npu8/q35/pipeline-shared-buckets/producer
- cuda/q25/pipeline-buckets-cuda
- cuda/q25/paired
- cuda/q25/pipeline-npu
- cuda/q25/profile
- cuda/q25/pipeline-cuda
- cuda/q25/pipeline-buckets-npu
- cuda/q35/pipeline-buckets-cuda
- cuda/q35/paired
- cuda/q35/pipeline-npu
- cuda/q35/profile
- cuda/q35/pipeline-cuda
- cuda/q35/pipeline-buckets-npu
- npu/q25/pipeline-buckets-cuda
- npu/q25/paired
- npu/q25/pipeline-npu
- npu/q25/profile
- npu/q25/pipeline-cuda
- npu/q25/pipeline-buckets-npu
- npu/q35/pipeline-buckets-cuda
- npu/q35/paired
- npu/q35/pipeline-npu
- npu/q35/profile
- npu/q35/pipeline-cuda
- npu/q35/pipeline-buckets-npu
- cuda8/q25/pipeline-shared-parameters/trainer
- cuda8/q25/pipeline-shared-parameters/producer
- cuda8/q25/pipeline-homogeneous/trainer
- cuda8/q25/pipeline-homogeneous/producer
- cuda8/q25/baseline
- cuda8/q25/pipeline-shared-buckets/trainer
- cuda8/q25/pipeline-shared-buckets/producer
- cuda8/q35/pipeline-shared-parameters/trainer
- cuda8/q35/pipeline-shared-parameters/producer
- cuda8/q35/pipeline-homogeneous/trainer
- cuda8/q35/pipeline-homogeneous/producer
- cuda8/q35/baseline
- cuda8/q35/pipeline-shared-buckets/trainer
- cuda8/q35/pipeline-shared-buckets/producer

## 原始记录与任务

- 原始 JSONL 位于本目录 `experiment-results/`。
- 4＋4：`cf7c09e7515c8aaa` / `0cce0a79afb9fb2b`。
- 同构 8 卡：`352c9047eb604b07` / `10dd7ccc41b9c220`。
- 固定模型质量、多个 seed、完整 RL loss、压缩发布和恢复机制尚未验收。
