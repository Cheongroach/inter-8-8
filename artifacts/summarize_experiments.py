"""Create a measured report from exported worker records; retain incomplete labels."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    records = []
    for path in (root / "experiment-results").rglob("rank-0.jsonl"):
        records += [(str(path.relative_to(root / "experiment-results")), json.loads(line))
                    for line in path.read_text().splitlines()]
    lines = [
        "# H20＋910B4 实测记录",
        "",
        "本报告由已导出的 worker JSONL 生成。四个最终任务均完成；本表按rank0记录展示，跨组比较应使用combination-comparison.md中全部八个rank最慢时间。历史失败/缺失项不能当作零耗时。",
        "",
        "模型为 Qwen2.5-7B-Instruct 与 Qwen3.5-4B 文本主干，GSM8K 输入。共同实现采用 Transformers 5.2 SDPA 和 FSDP，全参数 FP32 master/AdamW、BF16 计算。",
        "生成/训练分工使用真实生成回答的 CE 更新，是系统预验，不是 PPO/GRPO 或最终质量验证。",
        "",
        "## 生成",
        "",
        "| 记录 | 每卡 batch | 输出上限 | 组 token/s |",
        "|---|---:|---:|---:|",
    ]
    for path, row in records:
        if row["event"] == "generation_case":
            lines.append(f"| {path.rsplit('/', 1)[0]} | {row['batch_size']} | "
                         f"{row['max_new_tokens']} | {row['tokens_per_second']:.2f} |")
    lines += [
        "",
        "每个配置三次运行，排除第一次；剩余两次样本不同且 prompt 长度不同。未增加专用推理服务、连续批处理或 Qwen3.5 FLA kernel，因此不是硬件最优值。",
        "",
        "## 全参数训练",
        "",
        "| 记录 | case | 有效计时步数 | 监督 token/s | 组墙钟秒 | 跨岛交换秒（rank 0） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    groups = {}
    for path, row in records:
        if row["event"] == "training_step" and row["step"] > 0:
            groups.setdefault((path, row["case"]), []).append(row)
    for (path, case), rows in sorted(groups.items()):
        wall = sum(x.get("end_to_end_seconds", x["seconds"]) for x in rows)
        tokens = sum(x["supervised_tokens"] for x in rows)
        exchange = sum(x["outer_exchange_seconds"] for x in rows)
        lines.append(f"| {path.rsplit('/', 1)[0]} | {case} | {len(rows)} | "
                     f"{tokens / wall:.2f} | {wall:.3f} | {exchange:.3f} |")
    lines += [
        "",
        "profile 的 microbatch 1/2/4 使用不同的全局 batch，不能比较端点 loss 的高低。paired 的同一步固定 16 样本；岛内 token/s 表示该岛监督 token，组合需使用两岛 token 总和与整体时间。",
        "上表排除首个更新便于观察热身后计算；判断整体收益还需完整事件，包括首步、等待、传输及权重发布。Local SGD 的 Adam 状态留在本岛，参数平均不是 DiLoCo。",
        "",
        "## 生成/训练分工",
        "",
        "| 记录 | 执行 | 发布间隔 | 总秒 | 监督 token |",
        "|---|---|---:|---:|---:|",
    ]
    for path, row in records:
        if row["event"] == "pipeline_case":
            lines.append(f"| {path.rsplit('/', 1)[0]} | {row['mode']} | {row['interval']} | "
                         f"{row['wall_seconds']:.3f} | {row['supervised_tokens']:.0f} |")
    lines += ["", "## 已完成的 suite", ""]
    for path, row in records:
        if row["event"] == "suite_complete":
            lines.append(f"- {path.rsplit('/', 1)[0]}")
    lines += [
        "", "## 原始记录与任务", "",
        "- 原始 JSONL 位于本目录 `experiment-results/`。",
        "- 4＋4：`cf7c09e7515c8aaa` / `0cce0a79afb9fb2b`。",
        "- 同构 8 卡：`352c9047eb604b07` / `10dd7ccc41b9c220`。",
        "- 固定模型质量、多个 seed、完整 RL loss、压缩发布和恢复机制尚未验收。",
    ]
    (root / "experiment-results-report.md").write_text("\n".join(lines) + "\n")
    print(root / "experiment-results-report.md")


if __name__ == "__main__":
    main()
