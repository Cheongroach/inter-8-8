"""Bounded worker runner; preserve caches on failures for a targeted repair."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--island", required=True)
    args = p.parse_args()
    code = Path("/tmp/laya-stage/code")
    stages = []
    for suite in ("profile", "paired"):
        for model in ("q25", "q35"):
            stages.append((f"{model}-{suite}", [
                "python3", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                str(code / "hardware_experiments.py"),
                "--assets", "/tmp/laya-stage/assets",
                "--output", f"/tmp/laya-stage/results/{model}/{suite}",
                "--device", args.device, "--model", model, "--island", args.island, "--suite", suite,
            ]))
    stages.append(("followup", ["bash", "/tmp/laya-stage/followup.sh"]))
    for name, command in stages:
        if "paired" in name or name == "followup":
            deadline = time.monotonic() + 1500
            needed = Path("/tmp/laya-experiment-peer.json") if "paired" in name else (
                Path("/tmp/laya-stage/followup.sh"))
            while not needed.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Missing stage configuration: {name}")
                time.sleep(2)
        for attempt in range(3):
            print(json.dumps(dict(event="stage_start", name=name, attempt=attempt)), flush=True)
            result = subprocess.run(command, cwd=code)
            if result.returncode == 0:
                break
            print(json.dumps(dict(event="stage_failed_waiting_for_repair", name=name,
                                  code=result.returncode, attempt=attempt)), flush=True)
            resume = Path("/tmp/laya-stage/resume")
            deadline = time.monotonic() + 1500
            while not resume.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"No repair for {name}")
                time.sleep(2)
            resume.unlink()
        else:
            raise RuntimeError(f"Stage exhausted repair attempts: {name}")
    print("LAYA_EXPERIMENTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
