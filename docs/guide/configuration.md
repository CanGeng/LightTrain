# Configuration

> [中文版](configuration.zh-CN.md) · [Docs index](../README.md)

A recipe is OmegaConf YAML validated by a Pydantic v2 schema (`RootConfig`).
Loading order: **YAML read → `defaults:` merge → CLI overrides → `${}`
interpolation → Pydantic validation**.

## Required sections

Missing any of these errors out immediately:

| Section | Error |
| ------- | ----- |
| `model:` | `recipe is missing 'model:' section` |
| `data:` | `recipe is missing 'data:' section` |
| `optim:` | `recipe is missing 'optim:' section` |

## Root fields

| Field | Default | Notes |
| ----- | ------- | ----- |
| `mode` | `lab` | `lab` auto-attaches diagnostics; `prod` is lean |
| `seed` | `42` | torch / numpy / python |
| `exp` | `default` | names the run dir |
| `run_root` | `runs` | output root |
| `run_dir` | none | override the full output path |
| `user_modules` | `[]` | extra Python files to import (run `@register` on your components) |
| `models:` / `optimizers:` | none | the named model/optimizer set — see [Training § multi-model](../concepts/training.md#multi-model) |

## Specifying the model (`model_profiles`)

Declare one or more complete model configs and pick one by name:

```yaml
model: base                 # selector (CLI: model=big)
model_profiles:
  base: { name: tiny_lm, d_model: 256, n_layers: 4, n_heads: 8 }
  big:  { name: tiny_lm, d_model: 512, n_layers: 8, n_heads: 8 }
```

```bash
lighttrain train -c r.yaml model=big                      # switch profile
lighttrain train -c r.yaml model_profiles.base.d_model=384  # tweak a field
```

Profile selection is orthogonal to the model **set** (`models:`); a `models:`
entry's `spec` can be `{ profile: <name> }`. (Bare `model:` dict blocks were
removed in v0.1.8 — always declare profiles; run
`lighttrain migrate config <recipe> --to-profiles` to convert an old recipe.)

## Component reference syntax

Either a registry short name (**A**) or a direct import (**B**):

```yaml
optim: { name: adamw, lr: 3e-4 }                 # A — registry
optim: { _target_: torch.optim.AdamW, lr: 3e-4 } # B — import
```

## `defaults:` composition & interpolation

```yaml
defaults:
  - base/model_tiny        # path relative to this file, no .yaml suffix
  - ../shared/adamw_optim
trainer: { max_steps: 3000 }   # overrides values from the merged files
```

```yaml
seed: 1337
scheduler: { total_steps: ${trainer.max_steps} }   # same-config reference
data: { dataset: { seed: ${seed} } }
optim: { lr: ${oc.env:LR,3e-4} }                    # env var w/ default
```

## Default fall-backs (provide only model+data+optim)

| Component | Auto fallback |
| --------- | ------------- |
| `loss` | `cross_entropy` (shift + CE) |
| `trainer` | `pretrain`, max_steps=1000, ckpt every 500 |
| `engine` | `standard`, bf16 |
| `data.name` | `simple` |
| `tokenizer` | `byte` (vocab 260) |
| `collator` | `causal_lm` (right-pad) |
| `scheduler` | none (constant lr) |
| `callbacks` / `logger` | `[]` (lab mode still attaches diagnostics) |
| device | CUDA if available else CPU |

In `lab` mode these diagnostics callbacks attach automatically: `invariants`,
`frozen_step` (every 1000 steps), `file_signals`, plus a callback-isolation sink.

## Field tables (most-used)

`trainer:` — `name` (`pretrain`/`preference`/`reward_model`/`ppo`/`grpo` + your
own), `max_steps`, `val_every`, `ckpt_every`, `log_every`, `grad_clip`,
`accumulate`. RL adds `rollout_backend`/`temperature`/`top_p`/`do_sample`/
`buffer_max_size`; `trainer:` forwards any extra kwarg (filtered by the trainer
signature).

`engine:` — `name`, `mixed_precision` (`no`/`fp16`/`bf16`), `update_rule.name`
(`standard`/`sam`/`mezo`/`rl`).

`data:` — `name`, `batch_size`, `num_workers`, `pin_memory`, `drop_last`, plus
`dataset` / `tokenizer` / `collator` / `sampler` sub-blocks.

For the exhaustive field & protocol tables see
[Registry & protocols](../reference/registry.md).

## See also

- [Architecture](../concepts/architecture.md) — how these sections become a run (init order)
- [Training paradigms](../concepts/training.md) — recipe shapes per paradigm
- [Distributed](../operations/distributed.md) — the `parallel:` block
