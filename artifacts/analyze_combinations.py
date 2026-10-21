"""Aggregate only completed, paired records; expose short-run limits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiment-results"


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def all_ranks(path):
    return [r for file in path.parent.glob("rank-*.jsonl") for r in rows(file)]


def main():
    training = []
    pipelines = []
    for model in ("q25", "q35"):
        baseline = {}
        for device in ("cuda8", "npu8"):
            records = rows(RESULTS / device / model / "baseline/rank-0.jsonl")
            steps = [r for r in records if r["event"] == "training_step"]
            if len(steps) == 8:
                baseline[device] = sum(r["end_to_end_seconds"] for r in steps)
        for case in ("sync-equal", "sync-h3-n1", "sync-h1-n3", "local-h4", "local-h8"):
            islands = []
            for device in ("cuda", "npu"):
                records = rows(RESULTS / device / model / "paired/rank-0.jsonl")
                steps = [r for r in records if r["event"] == "training_step" and r["case"] == case]
                if len(steps) != 8:
                    break
                islands.append(dict(device=device, seconds=sum(r["end_to_end_seconds"] for r in steps),
                                    tokens=sum(r["supervised_tokens"] for r in steps),
                                    exchange_seconds=sum(r["outer_exchange_seconds"] for r in steps)))
            if len(islands) == 2 and len(baseline) == 2:
                wall = max(r["seconds"] for r in islands)
                training.append(dict(model=model, case=case, measured_step_seconds=wall,
                                     supervised_tokens=sum(r["tokens"] for r in islands),
                                     relative_time=wall / min(baseline.values()), baseline=baseline,
                                     island_metrics=islands))
        layouts = [
            ("H生成/N训练", RESULTS / "cuda" / model, RESULTS / "npu" / model),
            ("N生成/H训练", RESULTS / "npu" / model, RESULTS / "cuda" / model),
        ]
        for label, producer, trainer in layouts:
            producer_device = "cuda" if label.startswith("H") else "npu"
            for prefix in ("pipeline", "pipeline-buckets"):
                for path in (producer / f"{prefix}-{producer_device}").glob("rank-0.jsonl"):
                    p_rows = all_ranks(path)
                    t_rows = all_ranks(trainer / f"{prefix}-{producer_device}/rank-0.jsonl")
                    add_pipeline(pipelines, model, label, prefix, p_rows, t_rows)
        for device, label in (("cuda8", "H20同构"), ("npu8", "910B4同构")):
            for folder in (RESULTS / device / model).glob("pipeline-*"):
                add_pipeline(pipelines, model, label, folder.name,
                             all_ranks(folder / "producer/rank-0.jsonl"), all_ranks(folder / "trainer/rank-0.jsonl"))
    data = dict(training=training, pipelines=pipelines,
                timing_note="Training uses max of each island's sum of measured step times, not a complete case timer. Pipeline includes update, wait and final weight publication, excludes model load.")
    (ROOT / "combination-metrics.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    lines = ["# 混合组合对照", "", "从双方完成的原始事件聚合，空白表示没有完成证据。", "", "## 固定数据的全参数训练", "",
             "每个 case 为 8 个更新、全局每步 16 样本。时间含每步等待、拷贝与交换；取两岛各自累计值的较大者。不是完整 case 墙钟，未计入 case 间加载、评测与日志。", "",
             "| 模型 | 方案 | 累计步秒 | H20×8 秒 | 910B4×8 秒 | 相对更快同构 |", "|---|---|---:|---:|---:|---:|"]
    for r in training:
        lines.append(f"| {r['model']} | {r['case']} | {r['measured_step_seconds']:.2f} | {r['baseline']['cuda8']:.2f} | {r['baseline']['npu8']:.2f} | {r['relative_time']:.2f}× |")
    lines += ["", "Local SGD 保留各岛 AdamW 状态，改变优化轨迹，不能据短程验证 loss 声称同质量收益。", "", "## 真实生成与 CE 更新分工", "",
              "每个 case 为 4 个更新，4 卡生成、4 卡训练，每生成卡 batch 2，输出上限 128。整体时间取双方最慢 rank 的较大者；包含最后一次完整权重发布。", "",
              "| 模型 | 布局 | 发布实现 | 执行 | 间隔 | 秒 | 监督 token | token/s |", "|---|---|---|---|---:|---:|---:|---:|"]
    for r in pipelines:
        lines.append(f"| {r['model']} | {r['layout']} | {r['implementation']} | {r['mode']} | {r['interval']} | {r['wall_seconds']:.2f} | {r['supervised_tokens']:.0f} | {r['tokens_per_second']:.2f} |")
    lines += ["", "frozen/间隔0表示固定教师，只传轨迹，不回传权重。它适用于离线数据生成，训练语义不同于周期刷新策略。", "",
              "每个配置只有一次短程实验；同构采用同样共享 SDPA/FSDP 实现，未用 vLLM/FLA。这里验证系统可运行和开销，不能代表硬件最优、最终模型质量或完整 PPO/GRPO。", "",
              "参数发布包含版本、形状及 SHA256 校验。原始逐参数实现与 CPU 分块发布分别记录，不能跨实现择优拼出异构胜利。", ""]
    (ROOT / "combination-comparison.md").write_text("\n".join(lines))
    print("training cases", len(training), "pipeline cases", len(pipelines))


def add_pipeline(target, model, layout, implementation, producer, trainer):
    keys = {(r["mode"], r["interval"]) for r in producer if r["event"] == "pipeline_case"}
    for mode, interval in sorted(keys):
        own = [r for r in producer if r["event"] == "pipeline_case" and
               (r["mode"], r["interval"]) == (mode, interval)]
        peer = [r for r in trainer if r["event"] == "pipeline_case" and
                (r["mode"], r["interval"]) == (mode, interval)]
        if len(own) != 4 or len(peer) != 4:
            continue
        assert len({r["supervised_tokens"] for r in peer}) == 1
        assert next(r for r in own if r["rank"] == 0)["supervised_tokens"] == peer[0]["supervised_tokens"]
        wall = max(r["wall_seconds"] for r in own + peer)
        tokens = peer[0]["supervised_tokens"]
        target.append(dict(model=model, layout=layout, implementation=implementation,
                           mode=mode, interval=interval, wall_seconds=wall,
                           supervised_tokens=tokens, tokens_per_second=tokens / wall))


if __name__ == "__main__":
    main()
