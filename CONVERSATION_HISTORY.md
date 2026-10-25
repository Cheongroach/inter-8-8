# Project conversation and decision history

This file records the user-visible requests and the resulting engineering
decisions. It is a curated project history, not a dump of system prompts,
credentials, hidden tool output, or unrelated conversations.

## Request timeline

1. "应该H20和910B一起申请看看"
   - Started with a real cross-device communication probe rather than assuming
     that CUDA and HCCL collectives interoperate.
2. "这个有什么用呢？单纯把两张卡拼在一起。有没有H20+910B4，各8张。但是比任何一个16张训练效果更好？调研下相关文献，然后决定下。如果ok我们就做。"
   - Set the strong acceptance criterion: mixed 8+8 must beat both homogeneous
     16-card systems at comparable work and quality.
3. "目标可以调整，不必纠结于8+8。根据直觉，合乎道理。可以试一下4+4。模型可以选择两种。一种是对Hopper/910B时代有优化的，一种是最新的Qwen3.5/3.6/3.8等模型。先做实验，看看是否有收益。"
   - Added a smaller 4+4 preflight and newer Qwen architecture coverage.
4. "可以试多几种组合。比如生成/训练分工；不同batch size混合；其他低频互联的模式下，强行将两种卡绑在一起算，是否值当等。"
   - Expanded the matrix to global synchronization, Local SGD, unequal batch
     mixes, disaggregated generation/training, weight publication intervals,
     bucketed publication, and frozen-teacher operation.
5. "做一下8+8 vs同构16卡，关闭机间RDMA等情况，是否有一战之力？"
   - Added real two-node 16-card baselines and an explicit RDMA-versus-Socket
     transport comparison.
6. "go on"
   - Continued execution and result recovery.
7. "可以先排队16卡。"
   - Authorized queueing two-worker, eight-device-per-worker jobs.
8. "现在咋样了"
   - Reported that the first homogeneous H20 arm had run while other arms were
     still waiting for resources.
9. "Qwen2.5太旧了。使用新架构也对比下。比如Qwen3.5/3.6/3.8一类的。"
   - Replaced the primary matrix with Qwen3.5-9B and Qwen3.8-27B, retaining
     Qwen3.5-4B as a smaller validation point.
10. "启动下"
    - Submitted the new-model matrix.
11. "可以给一个完整的评测吧，写成一个类似小论文的结果。"
    - Produced the paper-style report, raw event dataset, aggregate CSV/JSON,
      transport excerpts, and plots.
12. "将历史对话、工程文件、代码、数据和结论都放到：https://github.com/Cheongroach/inter-8-8"
    - Created this public export with infrastructure identifiers removed.

## Engineering progression

### Communication feasibility

The first probe verified application-level TCP and CPU Gloo exchange between an
H20 worker and a 910B4 worker. Standard NCCL and HCCL do not provide a mixed
device collective, so the mixed path exchanges corresponding FSDP shards
through host tensors.

### 4+4 preflight

The preflight covered:

- full-parameter CE updates for Qwen2.5-7B and Qwen3.5-4B text backbones;
- per-update gradient exchange and parameter averaging every 4/8 updates;
- unequal H20/NPU microbatch allocations;
- both directions of generation/training role assignment;
- serialized and overlapped execution;
- per-parameter and bucketed weight publication;
- frozen-teacher operation.

It established feasibility but did not show a stable mixed-hardware speed
advantage over the faster homogeneous eight-card baseline.

### Two-node H20 x16 matrix

The completed new-model matrix contains 42 training runs:

- three models;
- global FSDP with RDMA and Socket;
- 2 x 8 HSDP with RDMA and Socket;
- CPU/TCP island exchange at intervals 1, 4, and 16;
- two repetitions per configuration.

It also contains nine generation/training pipeline cases across publication
intervals 4, 16, and frozen-teacher operation.

### Current conclusion

Low-bandwidth operation benefits strongly from topology-aware sharding and less
frequent cross-node synchronization. This is not evidence that mixing H20 and
910B4 is intrinsically beneficial. A valid heterogeneous win still requires
the mixed arm to beat tuned homogeneous H20 and Ascend baselines at a common
quality threshold.



## 2026-10-25 follow-up

The user cancelled the two prolonged-queue sixteen-card arms and requested
continued Jev/Laya iteration if SOTA was not established. Their stopped state
was confirmed. Two separate AG News calibration rounds were completed with
real Laya weights; the negative first round is retained. A fresh 512-case
holdout supports the second round's NLL improvement, with unchanged accuracy
and slightly worse Brier. Jev remains unmeasured. No SOTA claim is made.

The user requested continued improvement and resumed after an interruption.
A third round moved to supervised Banking77 routing rather than more news
calibration. The official source was pinned and deduplicated before fitting;
the test set remained separate until validation selection was written.
Lexical, frozen-encoder, wider candidate-budget, shortlist, and prior-fusion
controls were completed.

The selected 75% lexical / 25% frozen-Laya probability blend reached 91.67%
accuracy on 3,072 held-out examples, improving on the 90.82% lexical baseline
by 0.85 points (paired 95% interval [0.36, 1.33]). Its calibration metrics
worsened versus that baseline. Direct Laya reranking regressed despite
99.71% top-10 candidate recall. The original Laya encoder and typed head
were not fine-tuned; the new linear heads and runtime loader are included.

An initial internal forward call failed before validation predictions because
it lacked the inference context supplied by the public API. The corrected
runner restored saved heads, reproduced their validation metrics, and kept the
original protocol/code snapshot. All eight test metrics were recomputed from
case probabilities; 16-case public API replay matched every choice, with
maximum probability discrepancy below 3.3e-6. These replay cases do not add
independent quality evidence. Jev and SOTA comparisons remain unresolved.
