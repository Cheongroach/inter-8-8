# Public export notes

## Included

- Experiment source code and archived revisions.
- Rank-level JSONL measurements.
- Aggregate CSV/JSON metrics.
- Communication feasibility records.
- Model revision and SHA256 manifests.
- Research, plans, comparison reports, conclusions, and the final paper-style
  report.
- A curated history of user-visible requests and engineering decisions.
- Banking77's CC BY 4.0 source files and attribution, domain classifier
  parameters, frozen-protocol predictions, and runtime parity records.

## Sanitized

The following identifiers were deterministically replaced without changing
performance measurements:

- physical host names;
- internal IPv4/IPv6 and MAC addresses;
- GPU UUIDs;
- scheduler trace, request, trial, pod, container, and run-pair identifiers;
- links to private infrastructure or internal documents.

Public model repository revisions and content SHA256 values were retained.
Archived code-hash manifests describe the pre-sanitization experiment snapshot.
`PUBLIC_SHA256SUMS` records the exact files in this public export.

## Excluded

The export intentionally excludes Python bytecode/cache files, transient
staging logs, raw scheduler submissions/status snapshots, authentication data,
system prompts, hidden tool output, and bootstrap logs that may contain
environment details. These are operational artifacts rather than experiment
inputs or measurements.

The sanitization utility is available at `tools/sanitize_public.py`.
