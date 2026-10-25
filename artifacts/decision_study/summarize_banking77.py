"""Render completed measurements; never select or fit an experimental candidate."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).parent / "results_round3"
    result = json.loads((root / "results.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    labels = protocol["labels"]
    selected = result["selected_on_validation_only"]
    rows = [json.loads(s) for s in (root / "test_predictions.jsonl").read_text().splitlines()]
    confusion = Counter()
    for row in rows:
        prediction = int(np.argmax(row["probabilities"][selected]))
        if prediction != row["label"]:
            confusion[(row["label"], prediction)] += 1
    errors = [dict(gold=labels[g], predicted=labels[p], count=n)
              for (g, p), n in confusion.most_common()]
    (root / "error_confusions.json").write_text(json.dumps(errors, indent=2))
    names = {
        "laya_default": "Laya 默认 77 类",
        "laya_wide": "Laya 扩大候选预算",
        "lexical": "TF-IDF + 监督线性头",
        "dense": "冻结 Laya Encoder + 监督线性头",
        "supervised": "监督概率混合",
        "laya_top10": "监督 top-10 → Laya",
        "fusion_0.25": "top-10 + prior，α=0.25",
        "fusion_1": "top-10 + prior，α=1",
    }
    text = [
        "# 第三轮：Banking77 领域适配与候选预算",
        "",
        f"验证集选定 `{selected}`。下表由完整逐样本结果直接生成；未按测试集替换赢家。",
        "本轮固定 77 类 schema，监督方法使用目标领域训练标签，不是零样本公平排名。",
        "选定方案为75%词法分类器概率＋25%冻结Laya Encoder分类头概率；"
        "相对词法基线准确率提升，但NLL、Brier和ECE均退化。原Laya权重未微调，不能称通用模型升级。",
        "",
        "| 方法 | 测试准确率 | NLL ↓ | Brier ↓ | ECE15 ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    order = list(names)
    for name in order:
        m = result["test_metrics"][name]
        mark = " **（验证选定）**" if name == selected else ""
        text.append(f"| {names[name]}{mark} | {m['accuracy']:.2%} | {m['nll']:.4f} | "
                    f"{m['brier_multiclass_sum']:.4f} | {m['ece_15']:.4f} |")
    text += [
        "",
        "## 配对证据",
        "",
        "| 选定配置相对 | 准确率变化 | 95%配对bootstrap区间 | 胜 / 负 |",
        "|---|---:|---:|---:|",
    ]
    for name, key in (("默认 Laya", "selected_vs_default"), ("词法监督基线", "selected_vs_lexical"),
                      ("监督混合", "selected_vs_supervised")):
        p = result[key]
        text.append(f"| {name} | {p['delta']*100:+.2f}pp | "
                    f"[{p['ci95'][0]*100:+.2f}, {p['ci95'][1]*100:+.2f}]pp | "
                    f"{p['wins']} / {p['losses']} |")
    text += [
        "",
        "## 数据与实施",
        "",
        f"- 拟合 {len(protocol['split_ids']['fit'])} 条、验证 {len(protocol['split_ids']['select'])} 条、"
        f"测试 {len(protocol['split_ids']['test'])} 条；原始训练10003、测试3080。",
        "- 规范化全文去重排除训练4条、测试8条，其中测试7条与训练重复、1条为测试内部重复；无标签冲突。",
        "- 测试仅在 selection.json 写入后推理。测试源经过排重，不冒充原版3080条官方完整分数。",
        f"- 验证选择词法 `{selection['lexical']}`、稠密 `{selection['dense']}`，"
        f"混合的稠密权重 {selection['dense_weight']}。",
        f"- top-10测试召回率 {result['diagnostics']['shortlist_recall']:.2%}；漏召回按系统错误处理。",
        "- 原始checkpoint的choice:11+温度0.10058被固定上游运行时钳制到0.5；按运行时实际值计算概率。",
        "- Encoder和typed decision head均未更新；新训练的是固定77类线性分类头。",
        "- 初次内部批推理缺少inference_mode，在零验证预测时中断；"
        "补齐上下文后恢复已保存的分类头，复核验证指标逐项一致。"
        "原代码快照、原协议哈希与恢复说明均保留，候选和数据不变。",
        "- 8种方案的测试指标已从逐样本概率完整复算；"
        "16例公共API回放验证了选定方案、默认/扩大预算及top-10路径。"
        "全部选择一致，最大概率差低于3.3e-6；回放不增加质量样本数。",
        "",
        "## 诊断与限制",
        "",
        "各路径的首例 token 使用情况（精确值来自编码器统计）：",
        "",
        "```json",
        json.dumps(result["diagnostics"]["usage_first_row"], indent=2, ensure_ascii=False),
        "```",
        "",
        "测试决策阶段墙钟时间（秒）：",
        "",
        "```json",
        json.dumps(result["diagnostics"]["decision_timings"], indent=2),
        "```",
        "",
        "上述计时为共享CPU、8线程、batch 8，包含编码、forward及逐条结果落盘，"
        "不包含模型加载；top-10时间还不包含候选分类器开销。不能当成完整API延迟或GPU速度。",
        "冻结表征、分类头拟合和各batch原始耗时另见`*_timing.json`与`*_selection.json`。",
        "",
        "该结果不验证跨语言、新增工具、schema变化、分布漂移或真实业务收益。"
        "上游称Banking77未用于其训练，但本项目没有独立审计全部预训练数据。"
        "精确去重不能排除相近表述相关性；bootstrap只覆盖本次case采样，"
        "不覆盖重新划分或训练随机性。没有Jev API、强全参数微调模型或生成模型同条件对照，不能称SOTA。",
        "",
        "## 剩余错误",
        "",
        "以下是测试集上的描述性错误统计，仅用于解释本轮结果；没有用于重新拟合或改选配置。",
        "",
        "| Gold → Prediction | 条数 |",
        "|---|---:|",
    ]
    for row in errors[:10]:
        text.append(f"| {row['gold']} → {row['predicted']} | {row['count']} |")
    text += [
        "",
        "## 复现与来源",
        "",
        "- [预先固定协议](PROTOCOL_ROUND3.md)、[运行与加载说明](BANKING77_REPRODUCE.md)。",
        "- [指标](results_round3/results.json)、[选择锁](results_round3/selection.json)、"
        "[逐样本概率](results_round3/test_predictions.jsonl)、[运行时校验](results_round3/verification.json)。",
        "- Banking77：[官方固定源](https://github.com/PolyAI-LDN/task-specific-datasets/tree/"
        "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)；"
        "Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)。",
        "- Laya：[固定上游](https://github.com/NandhaKishorM/laya/tree/"
        "7a77439b0923d6efe00052d50a7f787840bfce98)。候选筛选和mean pooling复用既有上游能力。",
        "",
    ]
    (root.parent / "REPORT_BANKING77.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    english = ["Laya default", "Laya wider budget", "TF-IDF + linear",
               "Frozen Laya + linear", "Supervised blend", "Laya top-10",
               "Prior fusion 0.25", "Prior fusion 1.0"]
    accuracy = [result["test_metrics"][k]["accuracy"] * 100 for k in order]
    colors = ["#0f766e" if k == selected else "#94a3b8" for k in order]
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.barh(english, accuracy, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("Accuracy (%) — 3,072 deduplicated held-out cases")
    ax.set_title("Banking77: fixed-schema supervised adaptation")
    for i, value in enumerate(accuracy):
        ax.text(value + .6, i, f"{value:.2f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, -.16, "Teal: selected on validation. Supervised arms use target-domain labels. No SOTA claim.",
            transform=ax.transAxes, fontsize=8)
    fig.savefig(root.parent / "banking77_accuracy.png", dpi=160)
    fig.savefig(root.parent / "banking77_accuracy.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
