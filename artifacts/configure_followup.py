"""Stage the authorized producer/trainer experiments on the two live workers."""
import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    jobs = {"cuda": "cf7c09e7515c8aaa", "npu": "0cce0a79afb9fb2b"}
    hosts, ports = {}, {}
    for device in jobs:
        pod = json.loads(Path(
            f"/tmp/laya-stage-jobs/{device}-retry-evidence/pods.json").read_text())["data"][0]
        assert pod["status"] == 1 and pod["host"]
        hosts[device] = pod["host"]
        ports[device] = int(pod["port_list"].split(",")[0])
    for device in jobs:
        script = "set -euo pipefail\nexport PYTHONPATH=/tmp/laya-stage/app:${PYTHONPATH:-}\n"
        if device == "npu":
            script += "source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
        script += "cd /tmp/laya-stage/code\n"
        for model in ("q25", "q35"):
            for producer in ("cuda", "npu"):
                role = "producer" if device == producer else "trainer"
                port = ports[producer]
                script += (
                    "python3 -m torch.distributed.run --standalone --nproc_per_node=4 "
                    "disaggregated_experiment.py --assets /tmp/laya-stage/assets "
                    f"--output /tmp/laya-stage/results/{model}/pipeline-{producer} "
                    f"--device {device} --model {model} --role {role} "
                    f"--host {hosts[producer]} --port {port} --steps 4 --batch 2 --max-new 128\n"
                )
        files = {name: (root / name).read_bytes() for name in (
            "hardware_experiments.py", "disaggregated_experiment.py")}
        files["../followup.sh"] = script.encode()
        statements = ["from pathlib import Path", "import base64,hashlib,zlib"]
        peer = json.dumps(dict(host=hosts["cuda"], port=ports["cuda"]))
        statements.extend([
            "p=Path('/tmp/laya-experiment-peer.json')",
            f"p.with_suffix('.partial').write_text({peer!r})",
            "p.with_suffix('.partial').replace(p)",
        ])
        for name, data in files.items():
            blob = base64.b64encode(zlib.compress(data)).decode()
            digest = hashlib.sha256(data).hexdigest()
            path = str(Path("/tmp/laya-stage/code") / name)
            statements.extend([
                f"data=zlib.decompress(base64.b64decode({blob!r}))",
                f"assert hashlib.sha256(data).hexdigest()=={digest!r}",
                f"p=Path({path!r})", "p.with_suffix('.staged').write_bytes(data)",
                "p.with_suffix('.staged').replace(p)", f"print('staged',{name!r},{digest!r})",
            ])
        command = "python3 -c " + shlex.quote("\n".join(statements))
        payload = dict(job_run_id=jobs[device], exec=command, exec_timeout_seconds=30)
        result = subprocess.run([
            "/home/tiger/.merlin-cli/bin/merlin-cli", "--control-plane", "cn",
            "job-v2", "runs", "execute-script", "--json", json.dumps(payload),
        ], capture_output=True, text=True, timeout=60)
        (root / f"{device}-followup-staging.log").write_text(result.stdout + result.stderr)
        print(device, result.returncode, result.stdout[-1200:], result.stderr[-1200:])
        if result.returncode:
            raise RuntimeError(f"Staging failed on {device}; see saved log")


if __name__ == "__main__":
    main()
