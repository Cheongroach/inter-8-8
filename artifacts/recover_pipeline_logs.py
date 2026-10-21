"""Recover only JSON events from archived stdout, without bootstrap content."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text()
    decoder = json.JSONDecoder()
    pending, groups, current = {}, {}, {}
    for match in re.finditer(r'\{"event"\s*:', text):
        try:
            row, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if "role" not in row:
            continue
        key = row["model"], row["role"], row["rank"]
        if row["event"] == "suite_complete":
            groups.setdefault((*key, current.get(key, "parameters")), []).append(row)
            continue
        pending.setdefault(key, []).append(row)
        if row["event"] == "pipeline_case":
            implementation = row.get("publish_format", "parameters")
            current[key] = implementation
            groups.setdefault((*key, implementation), []).extend(pending.pop(key))
    for (model, role, rank, implementation), records in groups.items():
        path = args.output / model / f"pipeline-shared-{implementation}" / role / f"rank-{rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        if rank == 0:
            print(model, role, implementation, len([r for r in records if r["event"] == "pipeline_case"]))


if __name__ == "__main__":
    main()
