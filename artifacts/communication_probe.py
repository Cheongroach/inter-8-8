"""Bounded two-worker CUDA/NPU CPU-transport probe; no training or downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import time
from datetime import timedelta
from pathlib import Path


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def receive_exact(conn, size):
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        count = conn.recv_into(view[offset:])
        if count == 0:
            raise ConnectionError(f"EOF at {offset}/{size}")
        offset += count
    return result


def send_tensor(conn, tensor, sequence):
    raw = memoryview(tensor.numpy()).cast("B")
    header = json.dumps({
        "sequence": sequence, "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }).encode()
    conn.sendall(struct.pack("!I", len(header)) + header)
    conn.sendall(raw)


def receive_tensor(conn, sequence, elements, torch):
    header_size = struct.unpack("!I", receive_exact(conn, 4))[0]
    assert header_size < 4096
    header = json.loads(receive_exact(conn, header_size))
    assert header["sequence"] == sequence
    assert header["bytes"] == elements * 4
    raw = receive_exact(conn, header["bytes"])
    assert hashlib.sha256(raw).hexdigest() == header["sha256"]
    return torch.frombuffer(raw, dtype=torch.float32)


def command_output(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=15).decode()[:6000]
    except Exception as error:
        return str(error)


def synchronize(torch, device):
    getattr(torch, device).synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "npu"), required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--peer-file", default="/tmp/laya-peer.json")
    args = parser.parse_args()
    import torch
    if args.device == "npu":
        import torch_npu
        npu_version = torch_npu.__version__
    else:
        npu_version = None
    getattr(torch, args.device).set_device(0)
    device = torch.device(args.device, 0)
    hostname = socket.gethostname()
    probe = torch.ones(8, device=device)
    assert probe.cpu().sum().item() == 8
    env_keys = (
        "METIS_WORKER_0_HOST", "METIS_WORKER_0_PORT", "ARNOLD_ID",
        "ARNOLD_WORKER_NUM", "ARNOLD_WORKER_GPU", "HOSTNAME",
    )
    emit(
        "ready", rank=args.rank, device=args.device, hostname=hostname,
        torch_version=torch.__version__, torch_npu_version=npu_version,
        device_name=getattr(torch, args.device).get_device_name(0),
        gloo_available=torch.distributed.is_gloo_available(),
        environment={k: os.environ.get(k) for k in env_keys},
        network=command_output(["ip", "-j", "address"]),
        hardware=command_output(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name", "--format=csv,noheader"]
            if args.device == "cuda" else ["npu-smi", "info"]
        ),
    )
    peer_path = Path(args.peer_file)
    deadline = time.monotonic() + 1200
    while not peer_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("peer configuration not delivered within 1200s")
        time.sleep(1)
    config = json.loads(peer_path.read_text())
    host, port = config["host"], int(config["port"])
    emit("peer_configured", rank=args.rank, **config)
    if args.rank == 0:
        address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0]
        server = socket.socket(address[0], socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port) if address[0] == socket.AF_INET else ("::", port))
        server.listen(1)
        server.settimeout(180)
        conn, remote = server.accept()
        emit("tcp_connected", rank=0, local=conn.getsockname(), peer=remote)
    else:
        start = time.monotonic()
        while True:
            try:
                conn = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.monotonic() - start > 180:
                    raise
                time.sleep(1)
        emit("tcp_connected", rank=1, local=conn.getsockname(), peer=conn.getpeername())
    conn.settimeout(120)
    # Known exact FP32 values make independent average verification possible.
    total_payload = 0
    for size in (4096, 1 << 20, 16 << 20, 64 << 20, 256 << 20):
        elements = size // 4
        reference = torch.arange(elements, dtype=torch.int64).remainder_(127).float()
        expected = reference + 0.5
        local_device = (reference + args.rank).to(device)
        for repeat in range(3):
            sequence = f"tcp:{size}:{repeat}"
            synchronize(torch, args.device)
            start = time.perf_counter()
            local = local_device.cpu()
            stage_seconds = time.perf_counter() - start
            start = time.perf_counter()
            if args.rank == 1:
                send_tensor(conn, local, sequence)
                result = receive_tensor(conn, sequence, elements, torch)
            else:
                remote_tensor = receive_tensor(conn, sequence, elements, torch)
                assert torch.equal(remote_tensor, reference + 1)
                result = (local + remote_tensor).mul_(0.5)
                send_tensor(conn, result, sequence)
            network_seconds = time.perf_counter() - start
            assert torch.equal(result, expected)
            start = time.perf_counter()
            local_device.copy_(result)
            synchronize(torch, args.device)
            redispatch_seconds = time.perf_counter() - start
            assert torch.equal(local_device.cpu(), expected)
            local_device.copy_((reference + args.rank).to(device))
            total_payload += size * 2
            emit(
                "tcp_average_passed", rank=args.rank, payload_bytes=size,
                repeat=repeat, roundtrip_seconds=network_seconds,
                roundtrip_payload_gbit_s=size * 16 / network_seconds / 1e9,
                device_to_cpu_seconds=stage_seconds,
                cpu_to_device_seconds=redispatch_seconds,
            )
        del reference, expected, local_device, local, result
    # Exchange Gloo rendezvous configuration over the already proven channel.
    # A second allocated port may be supplied; otherwise rank 0 reserves a free port.
    if args.rank == 0:
        gloo_port = config.get("gloo_port")
        if not gloo_port:
            reserve = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
            reserve.bind(("::", 0) if ":" in host else ("", 0))
            gloo_port = reserve.getsockname()[1]
            reserve.close()
        encoded = json.dumps({"host": host, "port": gloo_port}).encode()
        conn.sendall(struct.pack("!I", len(encoded)) + encoded)
        assert receive_exact(conn, 1) == b"Y"
    else:
        gloo_config = json.loads(receive_exact(conn, struct.unpack("!I", receive_exact(conn, 4))[0]))
        gloo_port = gloo_config["port"]
        conn.sendall(b"Y")
    conn.close()
    if args.rank == 0:
        server.close()
    emit("tcp_complete", rank=args.rank, bidirectional_payload_bytes=total_payload)
    import torch.distributed as dist
    url_host = f"[{host}]" if ":" in host else host
    emit("gloo_initializing", rank=args.rank, host=host, port=gloo_port)
    dist.init_process_group(
        backend="gloo", init_method=f"tcp://{url_host}:{gloo_port}",
        rank=args.rank, world_size=2, timeout=timedelta(seconds=180),
    )
    peer_metadata = [None, None]
    dist.all_gather_object(peer_metadata, {"hostname": hostname, "device": args.device})
    assert {item["device"] for item in peer_metadata} == {"cuda", "npu"}
    assert len({item["hostname"] for item in peer_metadata}) == 2
    emit("gloo_connected", rank=args.rank, peers=peer_metadata)
    for size in (4096, 1 << 20, 16 << 20, 64 << 20, 256 << 20):
        elements = size // 4
        reference = torch.arange(elements, dtype=torch.int64).remainder_(127).float()
        expected = reference + 0.5
        local_device = (reference + args.rank).to(device)
        for repeat in range(3):
            local = local_device.cpu()
            dist.barrier()
            start = time.perf_counter()
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            seconds = time.perf_counter() - start
            local.mul_(0.5)
            assert torch.equal(local, expected)
            local_device.copy_(local)
            synchronize(torch, args.device)
            assert torch.equal(local_device.cpu(), expected)
            local_device.copy_((reference + args.rank).to(device))
            emit(
                "gloo_average_passed", rank=args.rank, payload_bytes=size,
                repeat=repeat, all_reduce_seconds=seconds,
                payload_gbit_s=size * 8 / seconds / 1e9,
            )
        del reference, expected, local_device, local
    dist.barrier()
    dist.destroy_process_group()
    emit("probe_passed", rank=args.rank, device=args.device, hostname=hostname)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("probe_failed", error_type=type(error).__name__, error=str(error))
        raise
