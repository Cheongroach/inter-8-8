"""Run four generation replicas and four FSDP trainers on eight identical cards."""
import argparse
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    args = p.parse_args()
    for model in ("q25", "q35"):
        processes = []
        for role, indices in (("producer", "0,1,2,3"), ("trainer", "4,5,6,7")):
            env = dict(os.environ)
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                         "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR",
                         "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
                env.pop(name, None)
            env["CUDA_VISIBLE_DEVICES" if args.device == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"] = indices
            if args.device == "npu":
                env["HCCL_IF_BASE_PORT"] = "52000" if role == "producer" else "53000"
            output = Path(f"/tmp/laya-stage/results/{model}/pipeline-homogeneous/{role}")
            output.mkdir(parents=True, exist_ok=True)
            command = [
                "python3", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                "/tmp/laya-stage/code/disaggregated_experiment.py",
                "--assets", "/tmp/laya-stage/assets", "--output", str(output),
                "--device", args.device, "--model", model, "--role", role,
                "--host", "::1", "--port", "18461", "--batch", "2", "--steps", "4",
                "--max-new", "128",
            ]
            processes.append(subprocess.Popen(command, env=env))
        statuses = [proc.wait() for proc in processes]
        if any(statuses):
            raise RuntimeError(f"Homogeneous pipeline failed: {statuses}")


if __name__ == "__main__":
    main()
