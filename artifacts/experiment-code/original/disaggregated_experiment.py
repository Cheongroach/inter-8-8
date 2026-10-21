"""Measure real rollout/update overlap and full BF16 weight publication.

This is a system rehearsal using CE on generated trajectories, not PPO/GRPO
or an experiment establishing RL learning quality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from hardware_experiments import SupervisedLM, local_tensor


def send_json(sock, value):
    data = json.dumps(value).encode()
    sock.sendall(struct.pack("!Q", len(data)) + data)


def receive_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(min(size, 8 << 20))
        if not chunk:
            raise EOFError("Peer closed connection")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def receive_json(sock):
    return json.loads(receive_exact(sock, struct.unpack("!Q", receive_exact(sock, 8))[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "npu"), required=True)
    p.add_argument("--model", choices=("q25", "q35"), required=True)
    p.add_argument("--role", choices=("producer", "trainer"), required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--max-new", type=int, default=128)
    args = p.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    rank = int(os.environ["LOCAL_RANK"])
    accelerator = getattr(torch, args.device)
    accelerator.set_device(rank)
    device = torch.device(args.device, rank)
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    dist.init_process_group("nccl" if args.device == "cuda" else "hccl",
                            timeout=timedelta(seconds=600))
    args.output.mkdir(parents=True, exist_ok=True)
    events = []

    def log(event, **fields):
        value = dict(event=event, rank=rank, role=args.role, device=args.device,
                     model=args.model, timestamp=time.time(), **fields)
        events.append(value)
        print(json.dumps(value), flush=True)
        (args.output / f"rank-{rank}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in events) + "\n")

    def broadcast_json(value):
        data = json.dumps(value).encode() if rank == 0 else b""
        size = torch.tensor(len(data), dtype=torch.int64, device=device)
        dist.broadcast(size, 0)
        value = torch.tensor(list(data), dtype=torch.uint8, device=device) if rank == 0 else (
            torch.empty(size.item(), dtype=torch.uint8, device=device))
        dist.broadcast(value, 0)
        return json.loads(bytes(value.cpu().tolist()))

    sock = None
    if rank == 0:
        if args.role == "producer":
            listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("::", args.port))
            listener.listen(1)
            listener.settimeout(900)
            sock, _ = listener.accept()
            listener.close()
        else:
            deadline = time.monotonic() + 900
            while True:
                try:
                    sock = socket.create_connection((args.host, args.port), timeout=15)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
        sock.settimeout(600)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = args.assets / args.model
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples = [json.loads(x) for x in (args.assets / "data/train.jsonl").read_text().splitlines()]

    def load():
        dtype = torch.bfloat16 if args.role == "producer" else torch.float32
        if args.model == "q35":
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            config = json.loads((path / "config.json").read_text())
            lm, info = Qwen3_5ForCausalLM.from_pretrained(
                path, config=Qwen3_5TextConfig(**config["text_config"]), dtype=dtype,
                attn_implementation="sdpa", key_mapping={r"^model.language_model.": "model."},
                local_files_only=True, output_loading_info=True)
        else:
            lm, info = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, attn_implementation="sdpa",
                local_files_only=True, output_loading_info=True)
        assert not info.get("missing_keys") and not info.get("mismatched_keys"), info
        if args.role == "producer":
            return lm.to(device).eval()
        lm.config.use_cache = False
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = SupervisedLM(lm).to(device)
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.backbone.layers:
            fully_shard(layer, mp_policy=policy)
        fully_shard(model, mp_policy=policy)
        return model.train()

    for mode, interval in (("serial", 1), ("overlap", 1), ("serial", 2),
                           ("overlap", 2), ("serial", 4), ("overlap", 4)):
        model = load()
        optimizer = None if args.role == "producer" else torch.optim.AdamW(
            model.parameters(), lr=1e-5, foreach=False, weight_decay=0.01)
        version = 0
        if rank == 0:
            if args.role == "trainer":
                send_json(sock, dict(ready=True, mode=mode, interval=interval))
                assert receive_json(sock)["start"]
            else:
                assert receive_json(sock) == dict(ready=True, mode=mode, interval=interval)
                send_json(sock, dict(start=True))
        dist.barrier()
        accelerator.synchronize()
        started = time.perf_counter()
        total_tokens = 0

        def publish():
            start = time.perf_counter()
            payload = 0
            if args.role == "trainer":
                for name, param in model.named_parameters():
                    # All trainer ranks participate in the local FSDP all-gather.
                    full = param.detach().full_tensor() if hasattr(param, "full_tensor") else param.detach()
                    if rank == 0:
                        canonical = name.replace("backbone.", "model.", 1) if name.startswith("backbone.") else (
                            name.replace("output.", "lm_head.", 1))
                        cpu = full.to(dtype=torch.bfloat16, device="cpu").contiguous()
                        raw = cpu.view(torch.uint8).numpy().tobytes()
                        send_json(sock, dict(name=canonical, shape=list(cpu.shape), bytes=len(raw),
                                             sha256=hashlib.sha256(raw).hexdigest(), version=version))
                        sock.sendall(raw)
                        payload += len(raw)
                    del full
                if rank == 0:
                    send_json(sock, dict(done=True, version=version))
                    assert receive_json(sock)["loaded"] == version
            else:
                parameters = dict(model.named_parameters())
                while True:
                    header = broadcast_json(receive_json(sock) if rank == 0 else None)
                    if header.get("done"):
                        assert header["version"] == version
                        break
                    assert header["version"] == version
                    param = parameters[header["name"]]
                    assert list(param.shape) == header["shape"]
                    if rank == 0:
                        raw = receive_exact(sock, header["bytes"])
                        assert hashlib.sha256(raw).hexdigest() == header["sha256"]
                        cpu = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(header["shape"])
                        param.data.copy_(cpu)
                        payload += len(raw)
                    dist.broadcast(param.data, 0)
                if rank == 0:
                    send_json(sock, dict(loaded=version))
            accelerator.synchronize()
            log("weight_publish", mode=mode, interval=interval, version=version,
                seconds=time.perf_counter() - start, payload_bytes=payload)

        for step in range(args.steps):
            if args.role == "producer":
                prompts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": examples[step * args.batch * 4 + rank * args.batch + j]["question"]}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                    for j in range(args.batch)]
                tokenizer.padding_side = "left"
                encoded = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                tick = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**encoded, max_new_tokens=args.max_new,
                                            do_sample=False, pad_token_id=tokenizer.pad_token_id)
                accelerator.synchronize()
                generation_seconds = time.perf_counter() - tick
                trajectories = []
                for j in range(args.batch):
                    tokens = output[j, encoded["input_ids"].shape[1]:].cpu().tolist()
                    if tokenizer.eos_token_id in tokens:
                        tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    tokens = [t for t in tokens if t != tokenizer.pad_token_id]
                    assert tokens, "Empty rollout"
                    trajectories.append(dict(prompt=prompts[j][-384:], response=tokens, version=version))
                # HCCL/NCCL-friendly gathering of small metadata via broadcasts.
                rows = []
                for source in range(4):
                    data = json.dumps(trajectories).encode() if rank == source else b""
                    size = torch.tensor(len(data), dtype=torch.int64, device=device)
                    dist.broadcast(size, source)
                    buf = torch.tensor(list(data), dtype=torch.uint8, device=device) if rank == source else (
                        torch.empty(size.item(), dtype=torch.uint8, device=device))
                    dist.broadcast(buf, source)
                    rows.extend(json.loads(bytes(buf.cpu().tolist())))
                if rank == 0:
                    send_json(sock, dict(step=step, trajectories=rows, version=version))
                    if mode == "serial":
                        assert receive_json(sock)["updated"] == step
                log("rollout", mode=mode, interval=interval, step=step, version=version,
                    seconds=time.perf_counter() - tick, generation_seconds=generation_seconds,
                    tokens=sum(len(x["response"]) for x in rows))
            else:
                message = broadcast_json(receive_json(sock) if rank == 0 else None)
                assert message["step"] == step and message["version"] == version
                rows = message["trajectories"]
                own = rows[rank * args.batch:(rank + 1) * args.batch]
                ids = [x["prompt"] + x["response"] for x in own]
                width = max(map(len, ids))
                inputs = torch.full((args.batch, width), tokenizer.pad_token_id, dtype=torch.long, device=device)
                labels = torch.full_like(inputs, -100)
                attention = torch.zeros_like(inputs)
                for j, row in enumerate(own):
                    inputs[j, :len(ids[j])] = torch.tensor(ids[j], device=device)
                    labels[j, len(row["prompt"]):len(ids[j])] = torch.tensor(row["response"], device=device)
                    attention[j, :len(ids[j])] = 1
                count = torch.tensor(labels[:, 1:].ne(-100).sum().item(), device=device, dtype=torch.float32)
                dist.all_reduce(count)
                optimizer.zero_grad(set_to_none=True)
                tick = time.perf_counter()
                loss = model(input_ids=inputs, labels=labels, attention_mask=attention)
                (loss * (4 / count.item())).backward()
                optimizer.step()
                accelerator.synchronize()
                total_tokens += count.item()
                log("update", mode=mode, interval=interval, step=step, behavior_version=version,
                    seconds=time.perf_counter() - tick, tokens=count.item(), loss_sum=loss.item())
                if rank == 0 and mode == "serial":
                    send_json(sock, dict(updated=step))
            if (step + 1) % interval == 0:
                version += 1
                publish()
        if rank == 0:
            if args.role == "trainer":
                send_json(sock, dict(complete=True, total_tokens=total_tokens))
            else:
                result = receive_json(sock)
                assert result["complete"]
                total_tokens = result["total_tokens"]
        dist.barrier()
        accelerator.synchronize()
        log("pipeline_case", mode=mode, interval=interval, steps=args.steps,
            wall_seconds=time.perf_counter() - started, supervised_tokens=total_tokens,
            objective="CE on actual generated responses; no RL quality claim",
            optimizer_state_policy="single trainer; producer BF16 parameters only")
        del model, optimizer
        gc.collect()
        accelerator.empty_cache()
    if sock:
        sock.close()
    log("suite_complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
