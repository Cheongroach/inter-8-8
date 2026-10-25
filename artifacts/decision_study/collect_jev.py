"""Collect a version-pinned Jev baseline on the exact frozen Laya cases.

Only public AG News text is transmitted. A missing API credential records an
explicit not-run status; it never produces placeholder model probabilities.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

from evaluate import CRITERIA, read_dataset, sha256
from metrics import quality


MODEL = "jev-1.14.0"


def parse_response(body):
    if not isinstance(body, dict):
        raise ValueError("Jev response must be an object")
    if body.get("model") != MODEL:
        raise ValueError("Unexpected Jev model version")
    answers = body.get("answers", {})
    if not isinstance(answers, dict):
        raise ValueError("Answers must be an object")
    answer = answers.get("topic", {})
    if not isinstance(answer, dict):
        raise ValueError("Topic answer must be an object")
    values = answer.get("probabilities", {})
    if not isinstance(values, dict) or answer.get("type") != "choice" or set(values) != set(CRITERIA):
        raise ValueError("Response does not match the frozen four-class schema")
    p = [values[key] for key in CRITERIA]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in p):
        raise ValueError("Invalid probability value")
    if not math.isclose(sum(p), 1, abs_tol=1e-8):
        raise ValueError("Probability mass does not sum to one")
    choice = answer.get("choice")
    if choice not in values or values[choice] != max(p):
        raise ValueError("Choice and probability argmax disagree")
    return p


def frozen_cases(protocol, data):
    if protocol.get("criteria", CRITERIA) != CRITERIA:
        raise ValueError("Question criteria differ from the frozen protocol")
    source = {}
    for name in ("train", "test"):
        if sha256(data / f"{name}.csv") != protocol["data_sha256"][name]:
            raise ValueError("Dataset content changed")
        source.update({row["id"]: row for row in read_dataset(data / f"{name}.csv", name)})
    ids = [case for values in protocol["split_ids"].values() for case in values]
    if len(set(ids)) != len(ids) or any(case not in source for case in ids):
        raise ValueError("Duplicate or missing case ID")
    return {name: [source[case] for case in values]
            for name, values in protocol["split_ids"].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("TYPESAFE_API_KEY")
    if not token:
        (args.output / "jev-status.json").write_text(json.dumps({
            "model": MODEL, "status": "not_run", "reason": "missing TYPESAFE_API_KEY",
            "api_calls": 0,
        }, indent=2))
        raise SystemExit(2)
    protocol = json.loads(args.protocol.read_text())
    splits = frozen_cases(protocol, args.data)
    target = args.output / "jev-predictions.jsonl"
    if target.exists():
        raise FileExistsError("Refuse to overwrite existing API observations")
    test_p, test_y, timings = [], [], []
    with target.open("w") as stream:
        for name, rows in splits.items():
            for row in rows:
                payload = {"model": MODEL, "state": {"article": row["state"]},
                           "questions": {"topic": {"type": "choice",
                                                  "instructions": "What is the topic of `article`?",
                                                  "criteria": CRITERIA}}}
                request = urllib.request.Request(
                    "https://api.typesafe.ai/v1/systemone",
                    data=json.dumps(payload).encode(), method="POST",
                    headers={"Authorization": f"Bearer {token}",
                             "Content-Type": "application/json"},
                )
                started = time.perf_counter()
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        body = json.load(response)
                except urllib.error.HTTPError as error:
                    raise RuntimeError(f"Jev HTTP status {error.code}; no automatic retry") from None
                seconds = time.perf_counter() - started
                probabilities = parse_response(body)
                record = {key: row[key] for key in ("id", "label", "text_sha256")}
                record.update(split=name, model=body["model"], response=body,
                              seconds=seconds)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                timings.append(seconds)
                if name == "test":
                    test_p.append(probabilities)
                    test_y.append(row["label"])
    (args.output / "jev-results.json").write_text(json.dumps({
        "model": MODEL, "test_metrics": quality(test_p, test_y),
        "api_calls": len(timings), "latency_p50_seconds": float(np.median(timings)),
        "timing_scope": "sequential hosted API, including network; not GPU compute latency",
        "baseline_scope": "canonical candidate order; no permutation probe",
    }, indent=2))


if __name__ == "__main__":
    main()
