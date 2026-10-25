# H20 + Ascend 910B4 Heterogeneous Training Experiments

This repository contains the code, measurements, reports, and decision history
for experiments asking whether a fixed pool of NVIDIA H20 and Ascend 910B4
accelerators can outperform either homogeneous alternative.

The primary report is:

- [Cross-hardware low-frequency synchronization study](artifacts/16card-experiments/paper/report.md)

## Latest update: 2026-10-25

Both missing sixteen-card experiment arms are now stopped and will not be
restarted. The original three-arm training comparison remains incomplete.

A separate [Jev/Laya decision quality study](artifacts/decision_study/REPORT.md)
now contains two real-weight CPU rounds, per-case logits, frozen protocols, and
a loadable AG News calibration map. The first calibration round regressed.
The second round preserves 93.75% accuracy on a fresh 512-case custom holdout,
reduces NLL from 0.1803 to 0.1746 and ECE from 0.0334 to 0.0235, while Brier
slightly worsens. This is task-specific probability calibration, not SOTA.
Jev has no measured result because the API credential is unavailable.

## Main result

The completed H20 x16 experiments show that transport and topology dominate the
short-run system result:

- Qwen3.8-27B global FSDP took 70.56 seconds per 16 updates with RDMA and
  660.09 seconds with forced Socket transport.
- A 2 x 8 HSDP layout reduced the Socket result to 216.60 seconds while
  preserving synchronous optimizer semantics.
- Parameter averaging every 16 updates reduced it to 83.30 seconds, but this is
  Local SGD with different optimization semantics and no time-to-quality proof.
- Earlier 4+4 experiments did not establish a stable mixed-hardware advantage
  over the faster homogeneous baseline.

The new-model Ascend x16 and H20 x8 + Ascend x8 arms had not produced results at
the reporting cutoff. The repository therefore does **not** claim that the
three-arm comparison is complete or that heterogeneous hardware is faster at
equal quality.

## Repository map

- `artifacts/16card-experiments/paper/`: paper-style report, 42 training runs,
  nine pipeline cases, figures, and transport evidence.
- `artifacts/16card-experiments/code/`: two-node FSDP/HSDP, CPU-island, and
  generation/training pipeline implementations.
- `artifacts/experiment-results/`: rank-level 4+4 and homogeneous 8-card data.
- `artifacts/communication-results/`: CUDA/Ascend TCP and Gloo communication
  feasibility evidence.
- `artifacts/experiment-code/`: original and revised experiment snapshots.
- `artifacts/*.md`: research, experiment design, comparisons, and conclusions.
- `CONVERSATION_HISTORY.md`: user-visible project decision timeline.
- `PUBLICATION_NOTES.md`: public-export scope and sanitization rules.

## Reproducibility boundaries

The original training experiments are infrastructure measurements, not a model-quality paper. The separate decision study states its own held-out calibration scope. The completed
training timing cases use two repetitions and the generation/training pipeline
uses one observation per case. Local SGD was not evaluated to a fixed quality
threshold. The Qwen3.8 measurements use the shared Transformers-compatible
Qwen3.5 implementation and still require formal output/gradient parity against
the official reference before being used for a quality claim.

Model weights are not committed. Immutable revisions and SHA256 manifests are
under `artifacts/experiment-assets-manifest.json` and
`artifacts/16card-experiments/new-model-manifest.json`.

