# 第三轮：Banking77 领域路由（推理前冻结）

日期：2026-10-25。目标是检验领域监督与候选预算能否提高真实决策准确率。
本轮不再使用 AG News。Jev 无凭据，仍不参与实测。

## 数据和隔离

- 官方来源：PolyAI-LDN/task-specific-datasets，commit
  `57ec275d8078af65b7731c2a98be812d844a6d6b` 的 `banking_data/`。
- 校验 train/test/categories 的 Git blob；保留原始行号与规范化文本 SHA256。
- 先按规范化文本去重：训练源优先；跨 train/test 重复的 test 行排除并记录。
- 官方训练源每类按 `SHA256("banking77-r3:" + id)` 排序，前 10 条用于验证，
  其余用于监督拟合。官方测试源剩余全部样本只在选择锁定后评价。
- 官方测试集不参与拟合、词表、IDF、温度、超参数或方案选择。
- 不声称已审计上游预训练污染。上游脚本将 Banking77 标为非训练内任务；
  该声明不构成本项目对所有上游训练数据的独立审计。

## 固定候选

1. Laya 原始 77 类决策：checkpoint 默认 `max_len=512, head_max_len=192`。
2. Laya 扩大候选预算：`max_len=1024, head_max_len=768`，权重不变。
3. 词法监督基线：word(1,2)+char_wb(3,5) TF-IDF，各自最多 50000 特征，
   min_df=2、sublinear_tf，拼接后 L2 归一化；多项逻辑回归 C∈{1,4,16}。
4. Laya 冻结 Encoder + 新领域分类头：复用上游
   `embed_fn_from_agent` 的 mean pooling，max_length=128，L2 归一化，
   77 类多项逻辑回归 C∈{1,10,100}。只训练新线性头，不微调 Encoder 或 typed head。
5. 最优词法/稠密分类器概率混合：稠密权重∈{0,.25,.5,.75,1}。
6. 由上述监督候选中的验证赢家提供 top-10，按原类别顺序交给 Laya；
   max_len=512、head_max_len=256。不得用 gold 强行插入候选。
   比较纯重排，以及 `log p_classifier + α × logits/T_10`，
   α∈{.25,1}；候选外概率置零。另保留不重排的原分类器候选。

每一层只按验证准确率、NLL、固定名称顺序选择。最终从原始、扩大预算、
监督分类赢家和三种重排方案中选取一个；选择结果和参数写入不可覆盖文件后
才计算测试指标。披露全部预先注册的最终候选，不能根据测试结果换赢家。
分类器 C 的选择指标也全部保留。

## 质量、成本与适用范围

- 报告 accuracy、NLL（概率下限1e-12）、multiclass Brier、ECE15、
  paired bootstrap accuracy delta（2000次，seed=20261025）。
- top-10 recall 单独报告；截断和候选语义折叠单独记录。候选外置零导致的
  NLL 惩罚保留，不以条件准确率冒充完整系统准确率。
- 保存输入 ID、标签、候选、logits/概率、源码与数据指纹、线性头参数。
- 计时包括分词/编码/决策或分类器拟合，分阶段披露；CPU fp32、8线程。
  不将离线 batch 计时称为 API P95/P99，不跨硬件推导加速。
- 监督方案用了目标领域标签，不能称为与零样本模型等信息量比较。
  新分类头限定 Banking77 固定标签，不能称为通用 typed-decision 模型升级。
- 单数据集、单划分和冻结 Encoder 不满足 SOTA 证据要求。
- 原型复用上游 pooling、encoding、collation 和决策逻辑；候选筛选也已有上游实现，
  本轮贡献是监督适配、可部署参数与独立对照，而非发明 coarse-to-fine。
