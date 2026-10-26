# 第二轮协议：保持选择不变，检查概率能否改善

[原协议](PROTOCOL_ROUND2.md)在推理前冻结，保留不动。这里解释它为什么这样设计。

第一轮仿射校准在 256 条测试上只净多答对 1 条，准确率区间跨零，NLL 和 Brier 退化。第二轮没有按旧测试挑另一个赢家，而是设立新保留集，尝试更稳妥的概率调整。

## 新保留集怎样得到

沿用相同权重、源码、四类问题和候选描述，仍用完整 AG News 测试源。先按原去重规则和 ID 哈希排序，排除已看过的 256 条，然后划出 1,024 条拟合、512 条验证、512 条新保留集，三组互斥。

这把官方测试源中的一部分用于自定义目标分布校准，所以最终 512 条成绩不是官方 7,600 条完整测试成绩。上游已将 AG News 标为训练内任务，也不能据此声称领域外泛化。

## 概率调整的两个方式

温度 \(T\) 用 \(p=\mathrm{softmax}(z/T)\) 调整分布尖锐程度；正温度不改变最大分数位置。候选仍是 0.25–16 的 161 个对数等距温度。

仿射候选则先除以 checkpoint 的原温度 \(T_0\)，再做：

\[
p=\mathrm{softmax}\big(A(z/T_0)+b\big).
\]

\(A\) 为 4×4、\(b\) 为 4 维，正则化到 \(A=I,b=0\)，强度为 0.1 / 1 / 10。零更新时恰好回到原有概率；第一轮直接映射原始 \(z\)，零更新等效温度 1。两者先验不同，不能只按正则名称当成同一配置。

共五个候选：原温度、新温度、三种仿射。Encoder 和原模型参数全部固定。只运行原始选项顺序；第一轮循环翻转较低，验证集又没有选中四排列集成，所以本轮没有增加四倍推理成本。

## 选择规则先保护准确率

在 512 条验证集上，先排除准确率低于原模型的配置，再选择 NLL 最低者；平局优先原模型。新保留集披露五种候选，不允许按测试改选。

准确率差和每条 NLL 差均作 2,000 次配对 bootstrap，seed 20261025。NLL 差为“候选减基线”，负值表示改善。本轮选中温度 1.582823，保留集准确率不变、NLL 改善，但 Brier 略差，见[结果](REPORT.md)。

第二轮同时改变校准来源、数据量、仿射先验和选优目标，所以不能把改善单独归因于其中一个因素。第一轮前缀带来主题 / 时期偏差仍只是可能原因，没有被独立验证。

## 运行与证据

样本 ID、排除项、数据 / 代码 SHA256 在首次推理前保存。执行为 CPU eager（直接执行 PyTorch 算子）、8 线程、batch 8，上限 1,800 秒。从项目根目录使用新的输出目录：

```bash
python laya/decision_study/iterate.py \
  --upstream /path/to/pinned/laya-checkout \
  --model /path/to/pinned/laya-checkpoint \
  --data /path/to/prepared/ag_news \
  --previous laya/decision_study/results/protocol.json \
  --output /path/to/new-round2
```

证据：[冻结样本](results_round2/protocol.json)、[全部指标](results_round2/results.json)、[逐样本 logits](results_round2/predictions.jsonl)。温度作用于固定 checkpoint 和新闻 schema，不能直接迁移到 77 类银行路由或其他语言。
