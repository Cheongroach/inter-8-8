"""Append bounded publication comparisons to live workers' current shell."""
import argparse
import json
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    sid = {"cuda": "cf7c09e7515c8aaa", "npu": "0cce0a79afb9fb2b"}[args.device]
    hosts = {"cuda": "redacted-ipv6-6a6307a6",
             "npu": "redacted-ipv6-c4beaaa1"}
    ports = {"cuda": 10572, "npu": 9694}
    extra = "\n"
    for model in ("q25", "q35"):
        for producer in ("cuda", "npu"):
            role = "producer" if args.device == producer else "trainer"
            extra += ("python3 -m torch.distributed.run --standalone --nproc_per_node=4 "
                      "disaggregated_experiment.py --assets /tmp/laya-stage/assets "
                      f"--output /tmp/laya-stage/results/{model}/pipeline-buckets-{producer} "
                      f"--device {args.device} --model {model} --role {role} "
                      f"--host {hosts[producer]} --port {ports[producer]} "
                      "--publish-format buckets --cases serial:4,overlap:4,frozen:0\n")
    code = ("from pathlib import Path; p=Path('/tmp/laya-stage/followup.sh'); "
            f"p.open('a').write({extra!r}); print('appended')")
    result = subprocess.run([
        "/home/tiger/.merlin-cli/bin/merlin-cli", "--control-plane", "cn", "job-v2", "runs", "execute-script",
        "--json", json.dumps(dict(job_run_id=sid, exec="python3 -c " + shlex.quote(code),
                                 exec_timeout_seconds=30))], capture_output=True, text=True, timeout=60)
    print(result.stdout, result.stderr)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
