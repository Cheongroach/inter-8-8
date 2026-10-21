"""Real-model generation, full-parameter training and paired low-frequency tests."""

from __future__ import annotations

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
from torch import nn


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


class SupervisedLM(nn.Module):
    """Compute assistant CE only at supervised positions, retaining full backbone."""

    def __init__(self, lm):
        super().__init__()
        self.backbone = lm.model
        self.output = lm.lm_head

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state[:, :-1]
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        logits = self.output(hidden[valid])
        return torch.nn.functional.cross_entropy(logits.float(), targets[valid], reduction="sum")


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


class PairedShards:
    """Pair corresponding local FSDP shards; never reduce unrelated shard ranges."""

    def __init__(self, config, local_rank, island):
        host = config["host"]
        port = int(config["port"])
        store = dist.TCPStore(
            host, port, 8, island == 0 and local_rank == 0,
            timeout=timedelta(seconds=900), wait_for_workers=True,
        )
        prefix = dist.PrefixStore(f"laya-shard-{local_rank}", store)
        self.pg = dist.ProcessGroupGloo(prefix, island, 2, timedelta(seconds=300))

    def sum_scalar(self, value):
        value = torch.tensor([value], dtype=torch.float64)
        self.pg.allreduce([value]).wait()
        return value.item()

    def exchange(self, model, gradients=False, average=False):
        start = time.perf_counter()
        values = []
        for name, parameter in model.named_parameters():
            value = parameter.grad if gradients else parameter.data
            if value is None:
                raise RuntimeError(f"Missing synchronized gradient: {name}")
            values.append((name, local_tensor(value)))
        # Manifest must agree before sharing a flattened buffer.
        manifest = hashlib.sha256(json.dumps([
            (name, list(value.shape), str(value.dtype)) for name, value in values
        ]).encode()).digest()
        manifest_tensor = torch.tensor(list(manifest), dtype=torch.int32)
        peer = torch.empty_like(manifest_tensor)
        outputs = [[manifest_tensor.clone(), peer]]
        self.pg.allgather(outputs, [manifest_tensor]).wait()
        assert torch.equal(outputs[0][0], outputs[0][1]), "Shard manifests differ"
        transferred = 0
        # Buckets limit host copies and avoid full model materialization.
        bucket = []
        elements = 0

        def flush():
            nonlocal bucket, elements, transferred
            if not bucket:
                return
            flat = torch.cat([value.detach().float().cpu().reshape(-1) for value in bucket])
            self.pg.allreduce([flat]).wait()
            if average:
                flat.mul_(0.5)
            offset = 0
            for value in bucket:
                length = value.numel()
                value.copy_(flat[offset:offset + length].reshape(value.shape))
                offset += length
            transferred += flat.numel() * 4
            bucket, elements = [], 0

        with torch.no_grad():
            for _, value in values:
                bucket.append(value)
                elements += value.numel()
                if elements >= 8 << 20:
                    flush()
            flush()
        return time.perf_counter() - start, transferred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "npu"), required=True)
    p.add_argument("--model", choices=("q25", "q35"), required=True)
    p.add_argument("--island", type=int, required=True)
    p.add_argument("--suite", choices=("profile", "paired", "baseline"), default="profile")
    p.add_argument("--peer-file", type=Path, default=Path("/tmp/laya-experiment-peer.json"))
    args = p.parse_args()
    if args.device == "npu":
        import torch_npu  # noqa: F401
    rank = int(os.environ["RANK"])
    accelerator = getattr(torch, args.device)
    accelerator.set_device(rank)
    device = torch.device(args.device, rank)
    torch.set_num_threads(2)
    torch.manual_seed(20261019)
    dist.init_process_group(
        "nccl" if args.device == "cuda" else "hccl",
        timeout=timedelta(seconds=600),
    )
    world = dist.get_world_size()
    args.output.mkdir(parents=True, exist_ok=True)
    events = []

    def record(name, **fields):
        entry = {"event": name, "rank": rank, "device": args.device, "model": args.model, **fields}
        events.append(entry)
        print(json.dumps(entry), flush=True)
        (args.output / f"rank-{rank}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in events) + "\n"
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = args.assets / args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples = [json.loads(line) for line in (args.assets / "data/train.jsonl").read_text().splitlines()]
    validation = [json.loads(line) for line in (args.assets / "data/test.jsonl").read_text().splitlines()]

    def load_model(training=False):
        dtype = torch.float32 if training else torch.bfloat16
        if args.model == "q35":
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            config = json.loads((model_path / "config.json").read_text())
            lm, loading = Qwen3_5ForCausalLM.from_pretrained(
                model_path, config=Qwen3_5TextConfig(**config["text_config"]),
                dtype=dtype, attn_implementation="sdpa", local_files_only=True,
                key_mapping={r"^model.language_model.": "model."},
                output_loading_info=True,
            )
        else:
            lm, loading = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
                output_loading_info=True,
            )
        assert not loading.get("missing_keys"), loading.get("missing_keys")
        assert not loading.get("mismatched_keys"), loading.get("mismatched_keys")
        record("model_loaded", training=training,
               unexpected_keys=sorted(loading.get("unexpected_keys", []))[:8])
        return lm

    def chat(row, answer=False):
        messages = [{"role": "user", "content": row["question"]}]
        if answer:
            messages.append({"role": "assistant", "content": row["answer"]})
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=not answer,
            enable_thinking=False, return_dict=False,
        )

    def batches(rows, indices, length):
        items = []
        for index in indices:
            row = rows[index % len(rows)]
            prompt = chat(row)
            full = chat(row, answer=True)
            assert full[:len(prompt)] == prompt, "Chat template not prefix-stable"
            # Keep response supervision by explicitly truncating the prompt.
            response = full[len(prompt):]
            response = response[:min(256, length // 2)]
            prompt = prompt[-max(1, length - len(response)):]
            ids = prompt + response
            labels = [-100] * len(prompt) + response
            items.append((ids, labels))
        padded = max(len(x) for x, _ in items)
        inputs = torch.full((len(items), padded), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        attention = torch.zeros_like(inputs)
        for i, (ids, y) in enumerate(items):
            inputs[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(y)] = torch.tensor(y)
            attention[i, :len(ids)] = 1
        return {k: v.to(device) for k, v in {
            "input_ids": inputs, "labels": labels, "attention_mask": attention,
        }.items()}

    def cleanup():
        gc.collect()
        accelerator.empty_cache()

    if args.suite in ("profile", "baseline"):
        # Independent generation replicas use all four cards, with actual EOS.
        lm = load_model().to(device).eval()
        tokenizer.padding_side = "left"
        for batch_size in (1, 4, 8):
            for max_new in (64, 128):
                durations, tokens = [], []
                for repeat in range(3):
                    prompts = [chat(examples[rank * 32 + repeat * batch_size + j])
                               for j in range(batch_size)]
                    encoded = tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    dist.barrier()
                    accelerator.synchronize()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        output = lm.generate(
                            **encoded, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    accelerator.synchronize()
                    elapsed = time.perf_counter() - start
                    generated = output[:, encoded["input_ids"].shape[1]:]
                    count = generated.ne(tokenizer.pad_token_id).sum().item()
                    durations.append(elapsed)
                    tokens.append(count)
                    record("generation_step", batch_size=batch_size, max_new_tokens=max_new,
                           repeat=repeat, seconds=elapsed, generated_tokens=count,
                           preview=tokenizer.decode(generated[0], skip_special_tokens=True)[:250])
                total = torch.tensor([sum(tokens[1:]), sum(durations[1:])], device=device)
                dist.all_reduce(total[:1], op=dist.ReduceOp.SUM)
                dist.all_reduce(total[1:], op=dist.ReduceOp.MAX)
                record("generation_case", batch_size=batch_size, max_new_tokens=max_new,
                       tokens_per_second=(total[0] / total[1]).item(),
                       wall_seconds=total[1].item(), warmup_excluded=True)
        del lm, output
        tokenizer.padding_side = "right"
        cleanup()

    paired = None
    if args.suite == "paired":
        assert world == 4, "Paired experiment requires matching four-rank islands"
        config = json.loads(args.peer_file.read_text())
        paired = PairedShards(config, rank, args.island)
    cases = (
        [("baseline-b2", 2, 1, 8)] if args.suite == "baseline" else
        [("profile-b1", 1, 1, 4), ("profile-b2", 2, 1, 4), ("profile-b4", 4, 1, 4)]
        if paired is None else
        [("sync-equal", 2, 1, 8), ("sync-h3-n1", 3 if args.island == 0 else 1, 1, 8),
         ("sync-h1-n3", 1 if args.island == 0 else 3, 1, 8),
         ("local-h4", 2, 4, 8), ("local-h8", 2, 8, 8)]
    )
    for case, batch_size, interval, steps in cases:
        cleanup()
        lm = load_model(training=True)
        total_parameters = sum(x.numel() for x in lm.parameters())
        lm.config.use_cache = False
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = SupervisedLM(lm).to(device)
        del lm
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in model.backbone.layers:
            fully_shard(layer, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mp_policy=policy, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False, weight_decay=0.01)
        record("training_ready", case=case, parameters=total_parameters,
               trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
               local_batch=batch_size, sync_interval=interval,
               parameter_dtype="fp32 master; bf16 FSDP compute",
               optimizer="AdamW; state dtype logged after first update",
               host=socket.gethostname(), torch_version=torch.__version__)
        model.train()
        for step in range(steps):
            accelerator.synchronize()
            end_to_end_start = time.perf_counter()
            if paired is not None:
                h_batch = 3 if case == "sync-h3-n1" else 1 if case == "sync-h1-n3" else 2
                offset = 4 * h_batch if args.island == 1 else 0
                indices = [step * 16 + offset + rank * batch_size + j for j in range(batch_size)]
            else:
                indices = [step * world * batch_size + rank * batch_size + j for j in range(batch_size)]
            batch = batches(examples, indices, 512)
            valid = batch["labels"][:, 1:].ne(-100).sum().item()
            count = torch.tensor(valid, device=device, dtype=torch.float32)
            dist.all_reduce(count)
            denominator = count.item()
            if paired is not None and interval == 1:
                denominator = paired.sum_scalar(denominator)
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            accelerator.synchronize()
            start = time.perf_counter()
            loss = model(**batch)
            (loss * (world / denominator)).backward()
            accelerator.synchronize()
            compute_seconds = time.perf_counter() - start
            transfer_seconds, payload = 0, 0
            if paired is not None and interval == 1:
                transfer_seconds, payload = paired.exchange(model, gradients=True)
            optimizer.step()
            if paired is not None and interval > 1 and (step + 1) % interval == 0:
                transfer_seconds, payload = paired.exchange(model, average=True)
            accelerator.synchronize()
            seconds = time.perf_counter() - start
            end_to_end_seconds = time.perf_counter() - end_to_end_start
            metrics = torch.tensor([loss.detach().item(), valid, seconds, end_to_end_seconds], device=device)
            dist.all_reduce(metrics[:2])
            dist.all_reduce(metrics[2:], op=dist.ReduceOp.MAX)
            record("training_step", case=case, step=step, local_batch=batch_size,
                   loss=(metrics[0] / metrics[1]).item(), supervised_tokens=metrics[1].item(),
                   seconds=metrics[2].item(), compute_seconds=compute_seconds,
                   end_to_end_seconds=metrics[3].item(),
                   outer_exchange_seconds=transfer_seconds, local_payload_bytes=payload,
                   peak_memory_bytes=accelerator.max_memory_allocated(),
                   optimizer_state_dtypes=sorted({
                       str(v.dtype) for state in optimizer.state.values()
                       for v in state.values() if isinstance(v, torch.Tensor)
                   }) if step == 0 else None)
        model.eval()
        eval_count = 8 // world
        batch = batches(validation, [rank * eval_count + j for j in range(eval_count)], 512)
        with torch.no_grad():
            value = model(**batch)
        metrics = torch.tensor([value.item(), batch["labels"][:, 1:].ne(-100).sum().item()], device=device)
        dist.all_reduce(metrics)
        record("eval_after", case=case, loss=(metrics[0] / metrics[1]).item(),
               supervised_tokens=metrics[1].item())
        del model, optimizer, batch, loss
        cleanup()
    dist.barrier()
    record("suite_complete", suite=args.suite)
    dist.destroy_process_group()
    if args.suite == "baseline" and args.model == "q35" and rank == 0:
        import subprocess
        followup = Path("/tmp/laya-stage/code/homogeneous_pipeline.py")
        if followup.exists():
            subprocess.run(["python3", str(followup), "--device", args.device], check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("suite_failed", error_type=type(error).__name__, error=str(error))
        raise
