# 第二轮目标分布校准协议

第一轮验证集选择的 `affine_1.0` 在独立256条测试样本上，准确率仅增加
0.39个百分点，配对区间跨零，NLL和Brier退化。因此不能推广该配置。
该负结果完整保留，不覆盖第一轮文件，也不按第一轮测试集选另一个赢家。

第二轮沿用相同权重、源码、问题、候选描述及完整AG News测试源文件。
从按原规则去除精确重复、再按样本ID哈希排序的官方测试源池中，先排除
第一轮已经查看的256条测试数据，再固定1024条拟合、512条验证、512条新测试。
三组互斥，旧测试集不用于任何第二轮拟合、选择或评分。

这是目标分布的自定义校准/验证/保留集划分，**不是AG News官方完整测试集跑分**。
AG News已被上游标为训练内任务，依然不宣称领域外泛化或SOTA。
第一轮训练来源前缀的时期或主题偏差只是待验证的原因；本轮同时增大样本量、
更换校准来源和改变正则化基准，不能独立归因任何收益。

仅运行候选原始排列。第一轮排列翻转较低，四次forward的集成没有在验证集上
建立选择优势，因此本轮不再增加四倍推理成本。

固定五种配置：checkpoint温度、拟合温度，以及正则强度0.1、1、10的20参数
仿射校准。温度仍在0.25–16的161个对数等距值上选择。
仿射校准先将原始logits除以checkpoint温度，再对单位映射和零偏置正则化，
因此先验确实对应现有的校准输出。第一轮则对原始logits的单位映射正则化，
其先验对应温度1，两者不相同。Encoder及原模型权重均保持固定。

在512条验证集上，先排除准确率低于原模型的配置，再按最低NLL选择配置；
平局优先保持原模型。独立512条保留集披露全部五种配置，禁止事后改选择。
报告配对准确率bootstrap，以及每例NLL差值的2000次配对bootstrap区间，
seed 20261025。NLL差值为候选减基线，负值更好。

实际样本ID、排除项、数据及源码SHA256在首次推理前保存到
`results_round2/protocol.json`。CPU eager、8线程、batch 8，执行限制1800秒。

```bash
python laya/decision_study/iterate.py \
  --upstream /path/to/pinned/laya-checkout \
  --model /path/to/pinned/laya-checkpoint \
  --data /path/to/prepared/ag_news \
  --previous laya/decision_study/results/protocol.json \
  --output laya/decision_study/results_round2
```
