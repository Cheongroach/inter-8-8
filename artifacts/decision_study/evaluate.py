"""Run a frozen AG News quality study against a pinned upstream Laya checkout.

The encoder stays unchanged. Domain probability recalibration and permutation
averaging are selected on validation data; test labels never select an arm.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from metrics import (
    apply_affine,
    fit_affine,
    fit_temperature,
    paired_accuracy,
    probabilities,
    quality,
)


CRITERIA = {
    "world": "world news and international politics",
    "sports": "sports",
    "business": "business and economy",
    "sci_tech": "science and technology",
}
EXPECTED_REVISION = "7a77439b0923d6efe00052d50a7f787840bfce98"
EXPECTED_TEST_BLOB = "130b40c3329c993468b8ed8a22c331315e478e0a"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_dataset(path, prefix, limit=None):
    result = []
    with path.open(newline="") as stream:
        for index, fields in enumerate(csv.reader(stream)):
            if len(fields) != 3 or fields[0] not in ("1", "2", "3", "4"):
                raise ValueError(f"Malformed AG News row {index}")
            text = " ".join(fields[1:]).strip()
            digest = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
            result.append(dict(id=f"{prefix}:{index}", label=int(fields[0]) - 1,
                               state=text, text_sha256=digest))
            if limit is not None and len(result) == limit:
                break
    return result


def freeze_splits(data_root, fit_count, select_count, test_count):
    test_bytes = (data_root / "test.csv").read_bytes()
    git_blob = hashlib.sha1(
        f"blob {len(test_bytes)}\0".encode() + test_bytes
    ).hexdigest()
    if git_blob != EXPECTED_TEST_BLOB:
        raise ValueError("Test CSV differs from the pinned source Git blob")
    train = read_dataset(data_root / "train.csv", "train")
    test = read_dataset(data_root / "test.csv", "test")
    if len(train) != 20000 or len(test) != 7600:
        raise ValueError("Require the fixed 20,000-row training prefix and complete test CSV")
    seen = set()
    unique_train, unique_test = [], []
    for rows, destination in ((train, unique_train), (test, unique_test)):
        for row in rows:
            if row["text_sha256"] in seen:
                continue
            seen.add(row["text_sha256"])
            destination.append(row)
    key = lambda row: hashlib.sha256(("20261025:" + row["id"]).encode()).hexdigest()
    unique_train.sort(key=key)
    unique_test.sort(key=key)
    return {
        "fit": unique_train[:fit_count],
        "select": unique_train[fit_count:fit_count + select_count],
        "test": unique_test[:test_count],
    }


def infer(agent, rows, output_path, batch_size):
    keys = list(CRITERIA)
    variants = []
    timings = []
    usage = []
    for rotation in range(4):
        order = keys[rotation:] + keys[:rotation]
        questions = {"topic": {"type": "choice",
                              "instructions": "What is the topic of `article`?",
                              "criteria": {key: CRITERIA[key] for key in order}}}
        captured = []
        original = agent._forward

        def capture(batch):
            logits, action = original(batch)
            captured.extend(logits[:, :4].copy().tolist())
            return logits, action

        agent._forward = capture
        started = time.perf_counter()
        try:
            results = agent.predict_batch(
                [{"article": row["state"]} for row in rows], questions,
                batch_size=batch_size, sort_by_length=False,
            )
        finally:
            agent._forward = original
        timings.append(time.perf_counter() - started)
        if len(captured) != len(rows) or len(results) != len(rows):
            raise RuntimeError("Inference lost or duplicated rows")
        logits = np.asarray(captured)[:, [order.index(key) for key in keys]]
        variants.append(logits)
        usage.append([result.get("usage") for result in results])
        print(json.dumps({"event": "rotation_done", "rotation": rotation,
                          "rows": len(rows), "seconds": timings[-1]}), flush=True)
    logits = np.stack(variants, axis=1)
    with output_path.open("w") as stream:
        for index, row in enumerate(rows):
            record = {k: row[k] for k in ("id", "label", "text_sha256", "split")}
            record["logits_by_rotation"] = logits[index].tolist()
            record["usage_by_rotation"] = [values[index] for values in usage]
            stream.write(json.dumps(record) + "\n")
    return logits, timings


def analyze(logits, rows, shipped_temperature):
    masks = {name: np.asarray([row["split"] == name for row in rows])
             for name in ("fit", "select", "test")}
    labels = np.asarray([row["label"] for row in rows])
    candidates = {"shipped": probabilities(logits[:, 0], shipped_temperature)}
    parameters = {"shipped": {"temperature": shipped_temperature}}
    temperature = fit_temperature(logits[masks["fit"], 0], labels[masks["fit"]])
    candidates["temperature"] = probabilities(logits[:, 0], temperature)
    parameters["temperature"] = {"temperature": temperature}
    for penalty in (0.01, 0.1, 1.0):
        weight, bias = fit_affine(logits[masks["fit"], 0], labels[masks["fit"]], penalty)
        name = f"affine_{penalty}"
        candidates[name] = apply_affine(logits[:, 0], weight, bias)
        parameters[name] = {"penalty": penalty, "weight": weight.tolist(), "bias": bias.tolist()}
    for name, temp in (("ensemble_shipped", shipped_temperature), ("ensemble_temperature", temperature)):
        candidates[name] = np.stack(
            [probabilities(logits[:, rotation], temp) for rotation in range(4)]
        ).mean(axis=0)
        parameters[name] = {"temperature": temp, "rotations": 4}
    selection_scores = {
        name: quality(p[masks["select"]], labels[masks["select"]])
        for name, p in candidates.items()
    }
    # Accuracy first, then proper scoring loss, then one-forward preference.
    chosen = min(selection_scores, key=lambda name: (
        -selection_scores[name]["accuracy"], selection_scores[name]["nll"],
        4 if name.startswith("ensemble") else 1, name,
    ))
    scores = {name: quality(p[masks["test"]], labels[masks["test"]])
              for name, p in candidates.items()}
    chosen_map = parameters[chosen]
    flip = np.argmax(logits[:, 0], axis=1) != np.argmax(logits[:, 2], axis=1)
    return {
        "selected_on_validation_only": chosen,
        "selected_parameters": chosen_map,
        "selection_metrics": selection_scores,
        "test_metrics": scores,
        "test_paired_selected_vs_shipped": paired_accuracy(
            candidates["shipped"][masks["test"]], candidates[chosen][masks["test"]],
            labels[masks["test"]],
        ),
        "test_order_flip_rate_rotation_0_vs_2": float(flip[masks["test"]].mean()),
        "selection_rule": "validation accuracy, NLL, fewer forwards, name",
        "test_arm_scope": "all preregistered arms disclosed; test never selects a winner",
        "upstream_training_overlap": "AG News is listed as in-training upstream; not an OOD claim",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=128)
    parser.add_argument("--select-count", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise ValueError("Upstream revision differs from frozen protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "predictions.jsonl").exists():
        raise FileExistsError("Refuse to overwrite recorded test predictions")
    splits = freeze_splits(args.data, args.fit_count, args.select_count, args.test_count)
    rows = [dict(row, split=name) for name, subset in splits.items() for row in subset]
    manifest = {
        "upstream_revision": revision,
        "model_sha256": sha256(args.model / "model.safetensors"),
        "config_sha256": sha256(args.model / "rl_agent_config.json"),
        "data_sha256": {name: sha256(args.data / f"{name}.csv") for name in ("train", "test")},
        "training_source_scope": "first 20,000 rows of the pinned 120,000-row source; exported before inference",
        "verified_test_git_blob": EXPECTED_TEST_BLOB,
        "split_ids": {name: [row["id"] for row in subset] for name, subset in splits.items()},
        "criteria": CRITERIA,
        "seed": 20261025,
        "device": "cpu",
        "batch_size": args.batch_size,
        "threads": args.threads,
        "implementation_sha256": {"evaluate.py": sha256(__file__),
                                  "metrics.py": sha256(Path(__file__).with_name("metrics.py"))},
    }
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(args.upstream))
    import torch
    torch.set_num_threads(args.threads)
    import laya
    started = time.perf_counter()
    agent = laya.load(str(args.model), device="cpu", backend="eager")
    load_seconds = time.perf_counter() - started
    agent.predict_batch([{"article": "A football team won its match."}],
                        {"topic": {"type": "choice", "instructions": "Topic?",
                                   "criteria": CRITERIA}}, batch_size=1)
    logits, timings = infer(agent, rows, args.output / "predictions.jsonl", args.batch_size)
    config = json.loads((args.model / "rl_agent_config.json").read_text())
    temperature = config["temperature_by_options"]["choice:3-5"]
    report = analyze(logits, rows, temperature)
    report.update({"torch": torch.__version__, "laya": laya.__version__,
                   "model_load_seconds": load_seconds, "rotation_seconds": timings,
                   "timing_scope": "CPU eager batched inference; excludes load; no hardware SOTA claim"})
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
