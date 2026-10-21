"""Two eight-card islands, native FSDP16 and matched CPU/TCP island policies."""
import argparse
import gc
import hashlib
import json
import os
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from hardware_experiments import SupervisedLM, local_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("/tmp/laya-stage/assets"))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    island = config["island"]
    native = config["policy"] in ("native16", "native-hybrid16", "standalone8")
    if args.device == "npu":
        import torch_npu  # noqa: F401
    accelerator = getattr(torch, args.device)
    accelerator.set_device(local_rank)
    device = torch.device(args.device, local_rank)
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    backend = "nccl" if args.device == "cuda" else "hccl"
    dist.init_process_group(backend, timeout=timedelta(seconds=600))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == (16 if native and config["policy"] != "standalone8" else 8)
    args.output.mkdir(parents=True, exist_ok=True)
    def log(event, **fields):
        row = dict(event=event, timestamp=time.time(), rank=rank, local_rank=local_rank,
                   island=island, device=args.device, model=args.model, config=config,
                   host=socket.gethostname(), **fields)
        with (args.output / f"island-{island}-rank-{rank}.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    log("runtime", torch_version=torch.__version__, backend=backend,
        env={k: os.environ.get(k) for k in (
            "NCCL_IB_DISABLE", "NCCL_NET", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
            "HCCL_SOCKET_IFNAME", "HCCL_INTRA_ROCE_ENABLE")},
        interfaces=[x.name for x in Path("/sys/class/net").iterdir()])
    def network_counters():
        files = list(Path("/sys/class/net").glob("*/statistics/*"))
        files += list(Path("/sys/class/infiniband").glob("*/ports/*/counters/*"))
        values = {}
        for f in files:
            try:
                values[str(f)] = int(f.read_text().strip())
            except (OSError, ValueError):
                pass
        return values
    if local_rank == 0:
        log("network_before", counters=network_counters())
    outer = None
    if not native:
        store = dist.TCPStore(config["host"], config["port"], 16,
                              island == 0 and local_rank == 0,
                              timeout=timedelta(seconds=600))
        outer = dist.ProcessGroupGloo(dist.PrefixStore(f"shard-{local_rank}", store),
                                      island, 2, timedelta(seconds=600))

    def scalar(value):
        t = torch.tensor([value], dtype=torch.float64)
        outer.allreduce([t]).wait()
        return t.item()

    def exchange(model, gradients, average=False):
        started = time.perf_counter()
        values = [(name, local_tensor(param.grad if gradients else param.data))
                  for name, param in model.named_parameters()]
        assert all(v is not None for _, v in values)
        digest = hashlib.sha256(json.dumps(
            [(n, list(v.shape), str(v.dtype)) for n, v in values]).encode()).digest()
        manifest = torch.tensor(list(digest), dtype=torch.int32)
        outputs = [[manifest.clone(), manifest.clone()]]
        outer.allgather(outputs, [manifest]).wait()
        assert torch.equal(*outputs[0])
        total = 0
        cpu_seconds = network_seconds = copy_back_seconds = 0.0
        bucket, elements = [], 0
        def flush():
            nonlocal total, bucket, elements, cpu_seconds, network_seconds, copy_back_seconds
            if not bucket:
                return
            tick = time.perf_counter()
            flat = torch.cat([v.detach().float().cpu().reshape(-1) for v in bucket])
            cpu_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            outer.allreduce([flat]).wait()
            network_seconds += time.perf_counter() - tick
            if average:
                flat.mul_(0.5)
            tick = time.perf_counter()
            offset = 0
            for v in bucket:
                length = v.numel()
                v.copy_(flat[offset:offset + length].reshape(v.shape))
                offset += length
            accelerator.synchronize()
            copy_back_seconds += time.perf_counter() - tick
            total += flat.numel() * 4
            bucket, elements = [], 0
        with torch.no_grad():
            for _, v in values:
                bucket.append(v)
                elements += v.numel()
                if elements >= 16 << 20:
                    flush()
            flush()
        return dict(seconds=time.perf_counter() - started, bytes=total,
                    cpu_seconds=cpu_seconds, network_seconds=network_seconds,
                    copy_back_seconds=copy_back_seconds)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = args.assets / args.model
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data = [json.loads(x) for x in (args.assets / "data/train.jsonl").read_text().splitlines()]
    interval = config.get("interval", 1)
    for repeat in range(config.get("repeats", 2)):
        if args.model in ("q35", "q359", "q38"):
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            c = json.loads((path / "config.json").read_text())
            lm, info = Qwen3_5ForCausalLM.from_pretrained(
                path, config=Qwen3_5TextConfig(**c["text_config"]), dtype=torch.bfloat16,
                key_mapping={r"^model.language_model.": "model."},
                attn_implementation="sdpa", local_files_only=True, output_loading_info=True)
        else:
            lm, info = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
                output_loading_info=True)
        assert not info.get("missing_keys") and not info.get("mismatched_keys"), info
        lm.config.use_cache = False
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = SupervisedLM(lm).train()
        del lm
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mesh = None
        if config.get("layout", config["policy"]) == "native-hybrid16":
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh(args.device, (2, 8), mesh_dim_names=("replicate", "shard"))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.backbone.layers:
            layer.float()
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        model.backbone.embed_tokens.float()
        model.output.float()
        model.backbone.norm.float()
        fully_shard(model, mesh=mesh, mp_policy=policy)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, foreach=False)
        # Match clocks at the case boundary without forcing every local step to synchronize.
        dist.barrier()
        if outer is not None:
            scalar(1)
        accelerator.synchronize()
        case_started = time.perf_counter()
        supervised = 0
        for step in range(config.get("steps", 16)):
            tick = time.perf_counter()
            batch_size = config.get("batch", 2)
            global_index = rank if native else island * 8 + rank
            indices = [step * (world if native else 16) * batch_size + global_index * batch_size + j
                       for j in range(batch_size)]
            prepared = []
            for index in indices:
                row = data[index]
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                full_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]},
                     {"role": "assistant", "content": row["answer"]}], tokenize=False,
                    enable_thinking=False)
                assert full_text.startswith(prompt_text), "Chat text prefix differs"
                # Tokenize supervision separately to avoid a BPE merge across
                # the generation prompt/answer boundary; matches rollout inputs.
                prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
                response = tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False)[:256]
                prompt = prompt[-(512 - len(response)):]
                prepared.append((prompt + response, [-100] * len(prompt) + response))
            width = max(len(x) for x, _ in prepared)
            inputs = torch.full((batch_size, width), tokenizer.pad_token_id, device=device, dtype=torch.long)
            labels = torch.full_like(inputs, -100)
            attention = torch.zeros_like(inputs)
            for j, (tokens, y) in enumerate(prepared):
                inputs[j, :len(tokens)] = torch.tensor(tokens, device=device)
                labels[j, :len(y)] = torch.tensor(y, device=device)
                attention[j, :len(tokens)] = 1
            count = torch.tensor(labels[:, 1:].ne(-100).sum().item(), device=device, dtype=torch.float32)
            dist.all_reduce(count)
            denominator = count.item()
            if outer is not None and interval == 1:
                denominator = scalar(denominator)
            opt.zero_grad(set_to_none=True)
            compute_tick = time.perf_counter()
            loss = model(inputs, attention, labels)
            (loss * (world / denominator)).backward()
            accelerator.synchronize()
            compute = time.perf_counter() - compute_tick
            transferred = {}
            if outer is not None and interval == 1:
                transferred = exchange(model, gradients=True)
            opt.step()
            if outer is not None and interval > 1 and (step + 1) % interval == 0:
                transferred = exchange(model, gradients=False, average=True)
            accelerator.synchronize()
            supervised += count.item()
            log("step", repeat=repeat, step=step, seconds=time.perf_counter() - tick,
                compute_seconds=compute, exchange=transferred, supervised_tokens=count.item(),
                loss_sum=float(loss.detach().item()), finite_loss=bool(torch.isfinite(loss.detach()).item()))
        dist.barrier()
        # End barrier included in true case timing, all ranks finish the same workload.
        if outer is not None:
            scalar(1)
        accelerator.synchronize()
        case_wall = time.perf_counter() - case_started
        log("case", repeat=repeat, wall_seconds=case_wall,
            supervised_tokens=supervised, steps=config.get("steps", 16),
            parameter_bytes=sum(p.numel() for p in model.parameters()) * 4,
            optimizer_policy="AdamW local states; parameter averaging" if interval > 1 else "global token normalized AdamW",
            peak_memory_bytes=accelerator.max_memory_allocated())
        del model, opt, loss, inputs, labels, attention
        gc.collect()
        accelerator.empty_cache()
    if local_rank == 0:
        log("network_after", counters=network_counters())
    log("complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
