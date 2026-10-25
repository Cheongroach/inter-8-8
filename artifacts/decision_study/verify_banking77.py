"""Replay exported heads/runtime and recompute recorded test metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from banking77 import digest, write_json
from banking77_runtime import BankingRouter
from metrics import quality


def main():
    parser = argparse.ArgumentParser()
    for name in ("artifact", "upstream", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact
    protocol = json.loads((root / "protocol.json").read_text())
    result = json.loads((root / "results.json").read_text())
    if result["selection_sha256"] != digest(root / "selection.json"):
        raise ValueError("Selection lock changed")
    implementation = protocol["implementation_sha256"]
    if (root / "resume.json").exists():
        resume = json.loads((root / "resume.json").read_text())
        if resume["original_protocol_sha256"] != digest(root / "protocol.json"):
            raise ValueError("Initial protocol changed during recovery")
        if digest(root / "banking77_initial.py") != implementation["banking77.py"]:
            raise ValueError("Initial code snapshot changed")
        implementation = resume["current_implementation_sha256"]
    for name, expected in implementation.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    records = [json.loads(s) for s in (root / "test_predictions.jsonl").read_text().splitlines()]
    if [r["id"] for r in records] != protocol["split_ids"]["test"]:
        raise ValueError("Test IDs reordered, lost, or duplicated")
    recomputed = {}
    labels = [r["label"] for r in records]
    for name in result["test_metrics"]:
        score = quality([r["probabilities"][name] for r in records], labels)
        if max(abs(score[k] - result["test_metrics"][name][k]) for k in score) > 1e-10:
            raise ValueError(f"Metric recomputation differs: {name}")
        recomputed[name] = score
    rows = [json.loads(s) for s in (root / "rows.jsonl").read_text().splitlines()]
    # Spread across the entire held-out file; replay never adds quality samples.
    indices = np.linspace(0, len(records) - 1, 16, dtype=int)
    by_id = {r["id"]: r for r in rows}
    replay = [by_id[records[i]["id"]] for i in indices]
    router = BankingRouter(root, args.upstream, args.model)
    selected = router.selected
    replay_results = {}
    for method in dict.fromkeys((selected, "laya_default", "laya_wide", "laya_top10")):
        router.selected = method
        actual = router.predict([r["text"] for r in replay])
        expected = np.array([records[i]["probabilities"][method] for i in indices])
        difference = float(np.max(np.abs(actual - expected)))
        choices_match = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
        if not choices_match or difference > 2e-4:
            raise ValueError(f"Runtime replay differs: {method}, choices={choices_match}, max_abs={difference}")
        replay_results[method] = dict(choices_match=choices_match, max_probability_error=difference)
    verification = dict(
        metric_recomputation=True, checked_methods=list(recomputed),
        test_ids_and_selection_lock_verified=True,
        frozen_implementation_verified=True,
        replay_rows=len(replay), replay_ids=[r["id"] for r in replay],
        selected=selected, runtime_replay=replay_results,
        actual_runtime_temperatures=router.agent.temperature_by_options,
        scope="Artifact/runtime parity only; replay is not additional held-out evidence",
        verifier_sha256=digest(__file__), runtime_sha256=digest(Path(__file__).with_name("banking77_runtime.py")),
    )
    write_json(root / "verification.json", verification)
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
