# 决策模型质量迭代协议

冻结日期：2026-10-25。该协议与 Qwen 异构训练吞吐实验独立。

## 目标与声明范围

在真实公开分类数据上，检验 Laya 的决策准确率、概率质量和选项顺序敏感性。
Jev 没有本轮 API 凭据，因此不填入推测结果。没有同一数据与输入下的
Jev、微调分类器及生成模型对照前，不作 SOTA 声明。

## 数据和版本

- 数据：AG News，原始 train 120,000 条、test 7,600 条。
  本轮使用原始 train 的固定前 20,000 条，test 使用完整 7,600 条。
- 数据副本：`mhjabreel/CharCnn_Keras`，commit
  `555590db4219b1243abb1918effd6a7425a2d75f`，路径
  `data/ag_news_csv/{train,test}.csv`。
- Laya 源码：`NandhaKishorM/laya`，commit
  `7a77439b0923d6efe00052d50a7f787840bfce98`。
- Laya 英文权重：`convaiinnovations/laya`，revision
  `7b928d828b7b0e022f929d9bd2e44165aa270148`。
- 每个文件的实际 SHA256、样本 ID 和本实验源码 SHA256 在推理前写入
  `protocol.json`。

训练 CSV 请求中途截断，因此在任何模型推理之前将来源范围固定为前 20,000 条，
单独导出完整前缀文件。该调整没有查看预测结果或按标签选数据；完整 120,000 条
下载与全源文件哈希验证仍未完成。模型、前缀或测试 CSV 不完整，数据量不符，
或源码 revision 不符即退出。
按规范化全文去掉精确重复，先处理 train，再处理 test，禁止精确重复跨 split。
按 `SHA256("20261025:" + source_row_id)` 排序：train 前 128 条拟合，
后 128 条用于配置选择；test 前 256 条保留测试。
这不是语义去重，也不能消除预训练污染。上游已经将 AG News 标为训练内任务，
本实验仅评价分布内决策与校准，不称为领域外泛化。
前缀样本可能带来时期或主题偏差，不能代表整个训练集。

## 推理与迭代

英文 checkpoint、CPU eager 路径，每 batch 8 条、8 个 Torch CPU 线程。
所有输入使用相同问题和四类描述。对四类候选做四次循环排列，
每次恢复到原始类别顺序后保存未经舍入的 logits。
每条输入的长度使用信息随原始输出保留。

预先固定七个配置：

1. checkpoint 自带的 `choice:3-5` 温度。
2. 在拟合集上选温度，候选为 0.25–16 的 161 个对数等距值。
3. logit 的 4×4 仿射映射，正则强度分别为 0.01、0.1、1。
4. 四次排列的概率均值，分别使用自带温度与拟合温度。

仿射映射正则化到单位矩阵和零偏置，400 次优化、学习率 0.01。
它只是输出分布的领域重校准，**没有微调 Encoder 权重**。
排列集成使用四倍 forward，必须披露额外执行成本，不能称为等延迟加速。

仅在配置选择集按准确率、NLL、较少 forward、配置名依次排序选配置。
保留测试集披露全部七配置，禁止按测试表现重新选择赢家或修改超参数。
无改善为有效负结果；之后扩展数据或训练方法必须另建协议与保留集。

## 评价

报告 accuracy、NLL（真实标签概率最低裁剪到 1e-12）、多类 Brier 求和、
15 等宽置信度桶 ECE、平均最大类别概率、原排列与旋转两位的选择翻转率。
主比较使用配置选择集选出的配置与 checkpoint 基线，在同一保留测试样本上
配对 bootstrap 2,000 次，seed 20261025，披露胜/负样本数与 95% 区间。

执行时间包含分词、拼接、forward 和解码，排除模型下载及加载。
CPU 结果不能作为 H20 性能或 Jev 托管 API 时延对照。

## 下一轮准入

SOTA 比较至少需要同一冻结任务集、相同候选 schema、任务外数据、
Jev 实际 API 返回、强分类器/生成模型基线、多个训练种子与独立校准集，
同时测量准确率、校准、拒答覆盖、P95/P99 和实际服务成本。
在完成这些证据前只能报告本轮具体任务的改进或负结果。

## 执行

```bash
python laya/decision_study/evaluate.py \
  --upstream /path/to/pinned/laya-checkout \
  --model /path/to/pinned/laya-checkpoint \
  --data /path/to/ag_news_csv \
  --output laya/decision_study/results
```

已有预测文件禁止覆盖。`python -m unittest discover -s laya/decision_study`
只验证数值与协议逻辑，不验证模型质量。


## 数据准备与 Jev 对照入口

`prepare_data.py --source /path/to/original/csv --output /path/to/prepared`
默认验证原始 train/test 的固定 Git blob，再导出20,000条训练前缀。
本轮因为原始 train 下载截断，使用显式 `--allow-training-prefix-only`；
测试源文件仍需通过完整 blob 校验。重复导出已核对与本轮输入逐字节一致。

Jev入口只在 `TYPESAFE_API_KEY` 已配置的环境执行；不通过命令行传密钥。
版本固定 `jev-1.14.0`，发送相同样本、问题和候选，保存完整回答及usage；
当前入口实际记录 `not_run`、`api_calls=0`，不是模型测试结果。

```bash
python laya/decision_study/collect_jev.py \
  --data /path/to/prepared \
  --protocol laya/decision_study/results/protocol.json \
  --output /path/to/jev-observations
```

Jev只测原始候选顺序；它的托管API时间包括网络，不能和CPU forward时间
直接比较。完整三方质量与服务性能验收仍需强分类器/生成模型对照。
第二轮协议见[目标分布校准协议](PROTOCOL_ROUND2.md)。
