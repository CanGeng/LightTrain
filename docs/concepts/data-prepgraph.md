# Data & PrepGraph

> [中文版](data-prepgraph.zh-CN.md) · [Docs index](../README.md)

## The simple data module

For most runs, `data: { name: simple, ... }` wraps a dataset + tokenizer +
collator + sampler into a DataLoader:

```yaml
data:
  name: simple
  dataset:   { name: line_file_text, path: corpus.txt, max_len: 256 }
  tokenizer: { name: byte }
  collator:  { name: causal_lm, max_len: 256 }
  sampler:   { name: shuffle, seed: ${seed} }
  batch_size: 4
  num_workers: 0
```

Built-ins (see [reference/registry.md](../reference/registry.md) for the full list):

- **datasets**: `line_file_text`, `preference_jsonl`, `artifact_joined`
- **collators**: `causal_lm`, `preference`, `multimodal`
- **samplers**: `shuffle`, `sequential`, `length_grouped`, `curriculum`,
  `stateful_resumable`
- **tokenizers**: `byte` (vocab 260)

## PrepGraph — content-addressed data prep

PrepGraph is a DAG of preparation nodes. Each node's fingerprint is
`sha256(canonical_config + code_version + schema_version + sorted upstream_fps)`;
results land atomically under `runs/<exp>/prep/<kind>/<name>/<fp>/` with
`MANIFEST_COMPLETE.json` written last. A node whose inputs and config are
unchanged is reused from cache; change anything upstream and only the affected
subtree recomputes.

`lighttrain train` auto-runs PrepGraph when `prep_graph:` is set, so explicit
`prep` is rarely needed.

Node kinds: `load`, `tokenize`, `chunk`, `pack`, `mix`, `join`, `index`,
`validate`, `materialize`.

### Packing strategies (`pack` node)

`strategy:` selects how documents fill the context window, each emitting
`truncation_rate` / `token_utilization` metrics:

- `concat_chunk` (default) — padding-free baseline
- `next_fit` — greedy pad-flush
- `best_fit` — best-fit-decreasing bin packing (opt-in, fewer truncations)

### Commands

```bash
lighttrain prep        -c cfg.yaml [--dry-run] [--workers N] [--pool thread|process]
lighttrain prep-graph  -c cfg.yaml [--out g.dot]   # render the DAG
lighttrain prep-status -c cfg.yaml                 # cache status per node
lighttrain prep-clean  -c cfg.yaml [--orphans]
lighttrain inspect-data -c cfg.yaml --n 4 --decoded
```

## Resume & data position

Checkpoints save sampler state and the consumed-batch count, so resume seeks the
sampler step-exactly mid-epoch (independent of DataLoader prefetch depth). See
[Diagnostics](../operations/diagnostics.md) and `resume-verify` in [CLI](../guide/cli.md).

## See also

- [Configuration](../guide/configuration.md) — the `data:` schema
- [Extending](../extending/extending.md) — write a custom dataset / collator / prep node
- [reference/registry.md](../reference/registry.md) — all data components
