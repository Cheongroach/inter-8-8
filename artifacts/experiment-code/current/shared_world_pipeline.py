"""One eight-rank root communicator and two four-rank device meshes."""
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                 "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR",
                 "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(name, None)
    for model in ("q25", "q35"):
      for publish_format, cases in (("parameters", "serial:4,overlap:4,frozen:0"),
                                    ("buckets", "serial:4,overlap:4,frozen:0")):
        subprocess.run([
            "python3", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            "/tmp/laya-stage/code/disaggregated_experiment.py", "--shared-world",
            "--assets", "/tmp/laya-stage/assets", "--device", args.device,
            "--model", model, "--role", "producer", "--host", "::1", "--port", "18461",
            "--output", f"/tmp/laya-stage/results/{model}/pipeline-shared-{publish_format}",
            "--publish-format", publish_format, "--cases", cases,
        ], env=env, check=True)


if __name__ == "__main__":
    main()
