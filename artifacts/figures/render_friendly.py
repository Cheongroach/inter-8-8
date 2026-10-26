#!/usr/bin/env python3
"""Render two reading-guide figures from the archived measurements."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FONT = OUT / "fonts" / "NotoSansCJKsc-Regular.otf"
if not FONT.exists():
    raise SystemExit("Install NotoSansCJKsc-Regular.otf in figures/fonts first.")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=str(FONT)).get_name(),
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
})

DECISION = ROOT / "decision_study/results_round3/results.json"
TRAINING = ROOT / "16card-experiments/paper/training-metrics.json"
d = json.loads(DECISION.read_text())
rows = json.loads(TRAINING.read_text())
assert len(rows) == 42
assert d["selected_on_validation_only"] == "supervised"
assert all(x["n"] == 3072 for x in d["test_metrics"].values())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=160)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


fig, (ax, pair) = plt.subplots(
    1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [2.2, 1]}
)
fig.subplots_adjust(left=.23, right=.96, bottom=.22, top=.76, wspace=.55)
keys = ["laya_default", "laya_wide", "lexical", "dense", "supervised", "laya_top10"]
labels = [
    "Laya 默认 77 类", "Laya 扩大候选预算", "词法监督分类器",
    "冻结 Laya Encoder＋分类头", "75% 词法＋25% 稠密（选定）", "监督 top-10＋Laya 纯重排",
]
values = [d["test_metrics"][k]["accuracy"] * 100 for k in keys]
ax.barh(range(6), values, color=["#aab3bf", "#aab3bf", "#2875ad", "#73889d", "#17836b", "#aab3bf"], height=.62)
ax.set_yticks(range(6), labels)
ax.invert_yaxis()
ax.set_xlim(0, 104)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("测试准确率（%，越高越好）")
ax.set_title("A  各方案的实际准确率", loc="left", pad=14)
for i, v in enumerate(values):
    ax.text(v + 1, i, f"{v:.2f}%", va="center", fontsize=10)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#e4e7eb")
b = d["selected_vs_lexical"]
delta, (lo, hi) = b["delta"] * 100, np.array(b["ci95"]) * 100
assert b["wins"] - b["losses"] == 26
assert np.isclose(delta, 26 / 3072 * 100)
pair.axvline(0, color="#73889d", linestyle="--", lw=1)
pair.errorbar(
    [delta], [.5], xerr=[[delta - lo], [hi - delta]],
    fmt="o", color="#17836b", capsize=6, lw=2,
)
pair.set_xlim(-.2, 1.65)
pair.set_ylim(0, 1)
pair.set_yticks([])
pair.set_xticks([0, .5, 1, 1.5])
pair.set_xlabel("准确率差（百分点）")
pair.set_title("B  融合相对词法基线", loc="left", pad=14)
pair.text(delta, .68, f"+{delta:.2f} pp", ha="center", fontsize=15)
pair.text(.05, .10, f"44 胜 / 18 负\n净多答对 26 条\n95% 区间 [{lo:.2f}, {hi:.2f}] pp", fontsize=10)
fig.suptitle("Banking77：监督融合小幅提高准确率", x=.08, ha="left", fontsize=19, y=.97)
fig.text(.08, .855, "拟合 9,229 / 验证 770 / 测试 3,072；原 Encoder 与 typed head 固定", fontsize=11)
fig.text(.08, .06, "A：监督方案使用领域标签，不能作为零样本公平排名。\nB：横线为 2,000 次配对样本 bootstrap 区间；不覆盖训练种子或重新划分的波动。", fontsize=10)
save(fig, "banking77-friendly")

configs = [
    ("native16", "native", 1, "全局 FSDP / RDMA", "#2875ad"),
    ("native16", "socket", 1, "全局 FSDP / Socket", "#8c98a5"),
    ("native-hybrid16", "socket", 1, "机内分片 HSDP / Socket", "#17836b"),
    ("cpu-islands", "cpu-tcp", 16, "每 16 步参数平均 / CPU TCP", "#bb7936"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
fig.subplots_adjust(left=.235, right=.96, bottom=.22, top=.76, wspace=.15)
shown = {}
for ax, model, title in zip(axes, ["q35", "q359", "q38"], ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.8-27B"]):
    vals = []
    for i, (layout, network, interval, label, color) in enumerate(configs):
        records = [r for r in rows if (r["model"], r["layout"], r["network"], r["interval"]) == (model, layout, network, interval)]
        records.sort(key=lambda r: r["repeat"])
        assert len(records) == 2, (model, layout, network)
        times = [r["wall_seconds"] for r in records]
        mean = float(np.mean(times))
        vals.append(mean)
        shown[f"{model}:{layout}:{network}:{interval}"] = times
        ax.barh(i, mean, color=color, height=.62, hatch="//" if interval == 16 else None)
        ax.scatter(times, [i - .1, i + .1], s=23, color="#202831", zorder=4)
        ax.text(mean + 12, i, f"{mean:.2f}", va="center", fontsize=10)
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlim(0, 740)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel("16 个更新耗时（秒）")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4e7eb")
axes[0].set_yticks(range(4), [c[3] for c in configs])
axes[0].invert_yaxis()
fig.suptitle("H20×16：Socket 下，把分片留在机内可减少耗时", x=.04, ha="left", fontsize=19, y=.97)
fig.text(.04, .855, "两台机器 × 每台 8 卡；每配置 2 次短程计时；同一模型各配置数据量一致", fontsize=11)
fig.text(.04, .055, "柱：两次均值；黑点：两次实际观测，不是置信区间。斜纹：参数平均改变训练规则，未验证同质量时间。\n27B 的全局 batch 为 16，另两模型为 32；三个面板共用秒轴，跨模型数值不表示相同训练任务。", fontsize=10)
save(fig, "h20-training-friendly")

metadata = {
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [DECISION, TRAINING]},
    "banking77_accuracy_percent": dict(zip(keys, values)),
    "paired_delta_pp": delta,
    "paired_ci95_pp": [lo, hi],
    "training_two_observations_seconds": shown,
    "mean_vs_raw_observations": "bars are arithmetic means; black points are raw repetitions",
}
(OUT / "figure-data.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
