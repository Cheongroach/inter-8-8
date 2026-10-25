# 第三轮：Banking77 领域适配与候选预算

验证集选定 `supervised`。下表由完整逐样本结果直接生成；未按测试集替换赢家。
本轮固定 77 类 schema，监督方法使用目标领域训练标签，不是零样本公平排名。
选定方案为75%词法分类器概率＋25%冻结Laya Encoder分类头概率；相对词法基线准确率提升，但NLL、Brier和ECE均退化。原Laya权重未微调，不能称通用模型升级。

| 方法 | 测试准确率 | NLL ↓ | Brier ↓ | ECE15 ↓ |
|---|---:|---:|---:|---:|
| Laya 默认 77 类 | 45.08% | 4.3379 | 0.8832 | 0.3537 |
| Laya 扩大候选预算 | 57.00% | 3.2423 | 0.7077 | 0.2743 |
| TF-IDF + 监督线性头 | 90.82% | 0.4145 | 0.1564 | 0.1026 |
| 冻结 Laya Encoder + 监督线性头 | 82.42% | 0.7610 | 0.2922 | 0.1519 |
| 监督概率混合 **（验证选定）** | 91.67% | 0.4438 | 0.1639 | 0.1550 |
| 监督 top-10 → Laya | 67.25% | 1.3474 | 0.4891 | 0.1414 |
| top-10 + prior，α=0.25 | 91.57% | 0.3775 | 0.1316 | 0.0462 |
| top-10 + prior，α=1 | 83.59% | 0.6881 | 0.2525 | 0.0828 |

## 配对证据

| 选定配置相对 | 准确率变化 | 95%配对bootstrap区间 | 胜 / 负 |
|---|---:|---:|---:|
| 默认 Laya | +46.58pp | [+44.73, +48.44]pp | 1481 / 50 |
| 词法监督基线 | +0.85pp | [+0.36, +1.33]pp | 44 / 18 |
| 监督混合 | +0.00pp | [+0.00, +0.00]pp | 0 / 0 |

## 数据与实施

- 拟合 9229 条、验证 770 条、测试 3072 条；原始训练10003、测试3080。
- 规范化全文去重排除训练4条、测试8条，其中测试7条与训练重复、1条为测试内部重复；无标签冲突。
- 测试仅在 selection.json 写入后推理。测试源经过排重，不冒充原版3080条官方完整分数。
- 验证选择词法 `lexical_C16`、稠密 `dense_C100`，混合的稠密权重 0.25。
- top-10测试召回率 99.71%；漏召回按系统错误处理。
- 原始checkpoint的choice:11+温度0.10058被固定上游运行时钳制到0.5；按运行时实际值计算概率。
- Encoder和typed decision head均未更新；新训练的是固定77类线性分类头。
- 初次内部批推理缺少inference_mode，在零验证预测时中断；补齐上下文后恢复已保存的分类头，复核验证指标逐项一致。原代码快照、原协议哈希与恢复说明均保留，候选和数据不变。
- 8种方案的测试指标已从逐样本概率完整复算；16例公共API回放验证了选定方案、默认/扩大预算及top-10路径。全部选择一致，最大概率差低于3.3e-6；回放不增加质量样本数。

## 诊断与限制

各路径的首例 token 使用情况（精确值来自编码器统计）：

```json
{
  "default": {
    "state": {
      "state_tokens": 12,
      "state_tokens_used": 12,
      "state_tokens_dropped": 0,
      "truncated": false
    },
    "options": {
      "options": 77,
      "options_distinct": 73,
      "tokens_per_option": 4
    },
    "input_tokens": 313
  },
  "wide": {
    "state": {
      "state_tokens": 12,
      "state_tokens_used": 12,
      "state_tokens_dropped": 0,
      "truncated": false
    },
    "options": {
      "options": 77,
      "options_distinct": 77,
      "tokens_per_option": null
    },
    "input_tokens": 370
  },
  "top10": {
    "state": {
      "state_tokens": 12,
      "state_tokens_used": 12,
      "state_tokens_dropped": 0,
      "truncated": false
    },
    "options": {
      "options": 10,
      "options_distinct": 10,
      "tokens_per_option": null
    },
    "input_tokens": 68
  }
}
```

测试决策阶段墙钟时间（秒）：

```json
{
  "default": 1687.894250612706,
  "wide": 1285.124773915857,
  "top10": 256.48575666733086
}
```

上述计时为共享CPU、8线程、batch 8，包含编码、forward及逐条结果落盘，不包含模型加载；top-10时间还不包含候选分类器开销。不能当成完整API延迟或GPU速度。
冻结表征、分类头拟合和各batch原始耗时另见`*_timing.json`与`*_selection.json`。

该结果不验证跨语言、新增工具、schema变化、分布漂移或真实业务收益。上游称Banking77未用于其训练，但本项目没有独立审计全部预训练数据。精确去重不能排除相近表述相关性；bootstrap只覆盖本次case采样，不覆盖重新划分或训练随机性。没有Jev API、强全参数微调模型或生成模型同条件对照，不能称SOTA。

## 剩余错误

以下是测试集上的描述性错误统计，仅用于解释本轮结果；没有用于重新拟合或改选配置。

| Gold → Prediction | 条数 |
|---|---:|
| balance_not_updated_after_bank_transfer → transfer_not_received_by_recipient | 5 |
| fiat_currency_support → exchange_via_app | 4 |
| pending_top_up → top_up_failed | 4 |
| change_pin → get_physical_card | 4 |
| virtual_card_not_working → get_disposable_virtual_card | 4 |
| top_up_failed → top_up_reverted | 4 |
| transfer_not_received_by_recipient → balance_not_updated_after_bank_transfer | 3 |
| top_up_reverted → top_up_failed | 3 |
| unable_to_verify_identity → verify_my_identity | 3 |
| disposable_card_limits → getting_virtual_card | 3 |

## 复现与来源

- [预先固定协议](PROTOCOL_ROUND3.md)、[运行与加载说明](BANKING77_REPRODUCE.md)。
- [指标](results_round3/results.json)、[选择锁](results_round3/selection.json)、[逐样本概率](results_round3/test_predictions.jsonl)、[运行时校验](results_round3/verification.json)。
- Banking77：[官方固定源](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)；Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)。
- Laya：[固定上游](https://github.com/NandhaKishorM/laya/tree/7a77439b0923d6efe00052d50a7f787840bfce98)。候选筛选和mean pooling复用既有上游能力。
