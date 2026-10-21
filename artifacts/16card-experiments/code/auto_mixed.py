"""Autonomous two-node homogeneous16 transport and island-policy comparisons."""
import json
import os
import socket
import subprocess
from datetime import timedelta
from pathlib import Path


def main():
    import torch.distributed as dist
    root = Path("/tmp/laya-stage")
    device = os.environ["LAYA_DEVICE"]
    node = 0 if device == "cuda" else 1
    host = os.environ.get("METIS_CUDA_0_HOST") or os.environ["ARNOLD_CUDA_0_HOST"]
    port_value = os.environ.get("METIS_CUDA_0_PORT") or os.environ["ARNOLD_CUDA_0_PORT"]
    ports = [int(x) for x in port_value.split(",")]
    launch_port, control_port = ports[:2]
    outer_port = launch_port
    pipeline_port = launch_port
    store = dist.TCPStore(host, control_port, 2, node == 0,
                         timeout=timedelta(seconds=2400))
    store.set(f"host-{node}", socket.gethostname())
    assert store.get("host-0") != store.get("host-1"), "Need two physical hosts"
    print(json.dumps(dict(event="two_hosts_ready", device=device, node=node,
                          hosts=[store.get(f"host-{i}").decode() for i in range(2)])), flush=True)
    matrix = []
    for model in ("q359", "q38", "q35"):
        for interval in (1, 4, 16):
            matrix.append(dict(model=model, policy="cpu-islands", layout="cpu-islands",
                               network="cpu-tcp", interval=interval))
        for producer in ("cuda", "npu"):
            matrix.append(dict(model=model, policy="pipeline", layout="pipeline-" + producer,
                               network="cpu-tcp", interval=0, producer=producer))
    for stage, item in enumerate(matrix):
        store.set(f"ready-{stage}-{node}", "1")
        store.wait([f"ready-{stage}-{i}" for i in range(2)])
        env = dict(os.environ)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                    "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                    "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "NCCL_IB_DISABLE", "NCCL_NET"):
            env.pop(key, None)
        env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET", GLOO_SOCKET_IFNAME="eth0")
        if item["network"] == "socket":
            env.update(NCCL_IB_DISABLE="1", NCCL_NET="Socket")
        output = root / "results" / f"{stage}-{item['model']}-{item['layout']}-{item['network']}-h{item['interval']}"
        output.mkdir(parents=True, exist_ok=True)
        env["NCCL_DEBUG_FILE"] = str(output / "nccl.%h.%p.log")
        launcher = ["python3", "-m", "torch.distributed.run", "--nproc_per_node=8"]
        config = dict(item, island=node, host=host, port=outer_port,
                      steps=16, repeats=2, batch=1 if item["model"] == "q38" else 2, env={k:env[k] for k in
                          ("NCCL_IB_DISABLE", "NCCL_NET", "GLOO_SOCKET_IFNAME") if k in env})
        config_path = root / f"auto-stage-{stage}.json"
        config_path.write_text(json.dumps(config))
        if item["policy"] == "native16":
            launcher += ["--nnodes=2", f"--node_rank={node}", f"--master_addr={host}", f"--master_port={launch_port}"]
        else:
            launcher += ["--standalone"]
        if item["policy"] == "pipeline":
            producer_host = store.get("host-0" if item["producer"] == "cuda" else "host-1").decode()
            # Use allocated port for the endpoint that listens.
            own_port = int(os.environ["PORT0"])
            store.set(f"pipeline-port-{stage}-{node}", str(own_port))
            producer_node = 0 if item["producer"] == "cuda" else 1
            chosen_port = int(store.get(f"pipeline-port-{stage}-{producer_node}"))
            arguments = [str(root / "code/pipeline16.py"), "--assets", str(root/"assets"),
                "--output", str(output), "--device", device, "--model", item["model"],
                "--role", "producer" if device == item["producer"] else "trainer", "--host", producer_host,
                "--port", str(chosen_port), "--steps", "16", "--batch", "1" if item["model"] == "q38" else "2", "--max-new", "128",
                "--cases", "overlap:4,overlap:16,frozen:0"]
        else:
            arguments = [str(root / "code/train16.py"), "--device", device, "--model", item["model"],
                         "--config", str(config_path), "--output", str(output)]
        with (output / f"node-{node}.log").open("w") as f:
            result = subprocess.run(launcher + arguments, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        for file in output.glob("*.jsonl"):
            for line in file.read_text().splitlines():
                print(line, flush=True)
        for file in output.glob("nccl.*.log"):
            for line in file.read_text().splitlines():
                if any(x in line for x in ("Using network", "NET/Socket", "via NET/", "NET/IB : Using")):
                    print("LAYA_TRANSPORT " + line, flush=True)
        store.set(f"done-{stage}-{node}", str(result))
        store.wait([f"done-{stage}-{i}" for i in range(2)])
        statuses = [int(store.get(f"done-{stage}-{i}")) for i in range(2)]
        record = dict(event="auto_stage_finished", stage=stage, item=item, statuses=statuses, node=node)
        (root / f"auto-done-{stage}.json").write_text(json.dumps(record))
        print(json.dumps(record), flush=True)
        if any(statuses):
            print(json.dumps(dict(event="failed_stage_retained", stage=stage)), flush=True)
    print("LAYA_MIXED16_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
