# lighttrain

PyTorch language model training framework.

Created by Claude Code.

Still in the testing phase.
---

## Overview

lighttrain is a highly customizable training framework designed for research workflows: pretraining small language models, supervised fine-tuning, preference learning, reinforcement learning, and distillation. The core — Registry, Config, Engine, UpdateRule, Trainer, EventBus, PrepGraph — is small enough to read end-to-end, while research-grade extensions (PEFT, vLLM, alternative architectures, sweep tooling) are available as opt-in [frontier plugins](plugins/).

Design goals:
**registry-first**, **failure-first**, **plugin-clean**, **single-GPU honest**, **lab-friendly**, and **audit-ready**.

---

## Installation

```bash
git clone <this-repo> lighttrain && cd lighttrain
pip install -e .
```

Optional extras:

```bash
pip install -e ".[peft]"          # LoRA / IA³ / AdaLoRA adapters
pip install -e ".[peft,quant]"    # + bitsandbytes 4-bit (Linux + CUDA only)
```

---

## Quickstart

```bash
lighttrain --version
lighttrain dry-run  -c recipes/pretrain_causal.yaml   # resolve config and print; no training
lighttrain train    -c recipes/pretrain_causal.yaml   # run training

# Data preparation
lighttrain prep       -c recipes/sft_chat.yaml --dry-run
lighttrain prep       -c recipes/sft_chat.yaml
lighttrain inspect-data -c recipes/sft_chat.yaml --n 4 --decoded
lighttrain train      -c recipes/sft_chat.yaml        # auto-invokes prep, then trains

# Teacher → student distillation
lighttrain produce-artifact -c recipes/produce_teacher.yaml
lighttrain train            -c recipes/student_kd.yaml
```

`lighttrain train` automatically invokes `PrepRunner.run()` when `cfg.prep_graph` is set, so explicit `prep` calls are rarely necessary.

---

## Architecture

The framework defines five clean seams; everything else remains straightforward.

1. **Registry** ([lighttrain/registry/_core.py](lighttrain/registry/_core.py)) —
   Short name to class resolution over a pre-declared category set:
   `model`, `loss`, `optimizer`, `dataset`, `data_module`, `tokenizer`,
   `collator`, `sampler`, `callback`, `logger`, `trainer`, `engine`,
   `update_rule`, `processor`, `prep_node`, and others.

2. **Config** ([lighttrain/config/](lighttrain/config/)) —
   OmegaConf loading (`defaults:` composition, `${var}` interpolation, CLI
   overrides) combined with Pydantic v2 schema (`RootConfig` with optional
   `prep_graph:` block).

3. **Engine and UpdateRule** ([lighttrain/engine/](lighttrain/engine/),
   [lighttrain/update_rules/](lighttrain/update_rules/)) —
   The engine owns the accelerator and delegates per-step mathematics
   (forward / backward / clip / step / scheduler) to a swappable `UpdateRule`.

4. **EventBus** ([lighttrain/callbacks/base.py](lighttrain/callbacks/base.py)) —
   39 lifecycle events dispatched via `getattr`; per-callback exceptions
   isolated; results aggregate to a `Signal`
   (`STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`).

5. **PrepGraph** ([lighttrain/prepgraph/](lighttrain/prepgraph/)) —
   Content-addressed DAG of data preparation nodes. Each node's fingerprint
   is `sha256(canonical_config + code_version + schema_version + sorted upstream_fps)`;
   results land atomically under `<store_root>/<kind>/<name>/<fp>/` with
   `MANIFEST_COMPLETE.json` written last.

---

## Built-in Components

### Models and Tokenizers

| Category | Registered names |
| -------- | ---------------- |
| Models | `tiny_lm` (~3M–30M-param pre-norm GPT), `hf_causal` (HuggingFace `AutoModelForCausalLM`), `tiny_rwkv`, `tiny_mamba`, `tiny_unet`, `jepa` |
| PEFT adapters | `lora`, `ia3`, `adalora` |
| Tokenizers | `byte` (vocab 260; PAD / BOS / EOS / UNK + 256 raw bytes) |

### Data Pipeline

| Category | Registered names |
| -------- | ---------------- |
| Datasets | `line_file_text`, `preference_jsonl`, `artifact_joined` |
| Collators | `causal_lm`, `preference`, `multimodal` |
| Samplers | `shuffle`, `sequential`, `length_grouped`, `curriculum`, `stateful_resumable` |
| Data modules | `simple`, `prep_graph` |
| PrepGraph nodes | `load`, `tokenize`, `chunk`, `pack`, `mix`, `validate`, `materialize`, `join`, `index` |
| Processors | `chat_template`, `hf_text`, `simple_image`, `hf_image`, `mel_spectrogram`, `hf_audio`, `frame_folder`, `decord_video` |

### Optimization

| Category | Registered names |
| -------- | ---------------- |
| Optimizers | `adamw`, `lion` (regex-based param groups, first-match-wins) |
| Schedulers | `constant`, `linear`, `warmup_cosine`, `wsd` |
| Update rules | `standard`, `mezo`, `sam`, `forward_forward`, `pcn`, `dfa` |

### Losses

| Category | Registered names |
| -------- | ---------------- |
| Standard | `cross_entropy`, `mlm`, `z_loss`, `composite` |
| Distillation | `kl_topk`, `hidden_mse`, `hidden_cosine`, `attention_transfer` |
| Preference | `bradley_terry`, `dpo`, `ipo`, `simpo`, `orpo`, `kto` |
| Reinforcement learning | `ppo_surrogate`, `grpo_loss` |
| Auxiliary | `info_nce`, `moe_balance` |

### Trainers

| Registered name | Use case |
| --------------- | -------- |
| `pretrain` | Causal language model pretraining and SFT |
| `dpo` / `ipo` / `simpo` / `orpo` / `kto` | Offline preference learning |
| `reward_model` | Reward model training |
| `ppo` | Online PPO with rollout buffer and GAE |
| `grpo` | Group Relative Policy Optimization |

### Callbacks and Loggers

| Category | Registered names |
| -------- | ---------------- |
| Callbacks | `ema`, `best_ckpt`, `throughput`, `early_stop`, `nan_skip`, `invariants`, `nan_hunter`, `frozen_step`, `loss_attribution`, `dead_neuron`, `grad_flow`, `sample_preview`, `dynamic_artifact`, `lineage_recorder`, `file_signals` |
| Loggers | `console`, `jsonl`, `tensorboard` |

---

## Run Directory

`make_run_dir(...)` produces a self-contained capsule under
`runs/<exp>/<ts>-<slug>-<short_hash>/`:

```
runs/<exp>/<ts>-<slug>-<short_hash>/
├── config.snapshot.yaml      # exact YAML as supplied
├── config.resolved.yaml      # post-merge / post-overrides / post-interpolation
├── env.json                  # Python / PyTorch / CUDA / git SHA / hostname / argv
├── logs/
│   ├── metrics.jsonl
│   └── events.out.tfevents.*
└── checkpoints/
    ├── step_500/{model.safetensors, optimizer.pt, scheduler.pt, rng.pt, manifest.json}
    ├── last.json
    └── best.json
```

PrepGraph cache lands separately under `runs/<exp>/prep/<kind>/<name>/<fp>/`.
A directory missing `manifest.json` (checkpoint) or `MANIFEST_COMPLETE.json`
(prep node) is treated as incomplete and skipped on the next run.

---

## CLI Reference

| Command | Description |
| ------- | ----------- |
| `lighttrain --version` / `--help` | Version and help |
| `lighttrain init <path>` | Generate a minimal recipe skeleton |
| `lighttrain dry-run -c <cfg>` | Resolve config and print; no training |
| `lighttrain train -c <cfg>` | Full training loop; auto-runs PrepGraph if `cfg.prep_graph` is set |
| `lighttrain overfit -c <cfg> --n N` | Overfit on N batches |
| `lighttrain resume --run <dir>` | Functional resume from a run directory |
| `lighttrain prep -c <cfg>` | Run data preparation only |
| `lighttrain prep-graph -c <cfg> --out g.dot` | Render the PrepGraph as a Graphviz dot file |
| `lighttrain prep-clean -c <cfg>` | Remove cached prep artefacts |
| `lighttrain prep-status -c <cfg>` | Show cache status for each node |
| `lighttrain inspect-data -c <cfg>` | Decoded batch preview, length histogram, label-mask coverage |
| `lighttrain produce-artifact -c <cfg>` | Run an `ArtifactProducer` from the recipe's `artifacts:` block |
| `lighttrain lineage tag / untag / pin / invalidate / gc / prune-orphans / graph` | SQLite lineage operations |
| `lighttrain migrate config / artifact-header / checkpoint` | Schema migrations with `.pre-migration-bak` backup |
| `lighttrain doctor --run <dir>` | Inspect checkpoints, lineage, frozen steps, NaN repros, crash bundles |
| `lighttrain freeze-step --run <dir> --step N` | Capture a single-step replay bundle |
| `lighttrain replay-step <bundle.zip>` | Replay a frozen step bundle |
| `lighttrain replay --run <dir>` | Replay the latest crash bundle or frozen step |
| `lighttrain profile -c <cfg> --steps N` | `torch.profiler` chrome trace |
| `lighttrain estimate -c <cfg>` | Trainable parameters, memory bound, tokens/s estimate |
| `lighttrain eval -c <cfg>` | Evaluate perplexity and EvalSuite metrics |
| `lighttrain regression-gate -c <cfg> --metric <name> --threshold <f>` | CI gate; exits 1 on failure |
| `lighttrain sweep -c <cfg> -s <sweep.yaml>` | Hyperparameter sweep (grid / random / Optuna) |
| `lighttrain compare <run_a> <run_b> …` | Config diff and metric comparison |
| `lighttrain fork --from <ckpt> -c <cfg>` | Branch from a checkpoint with lineage |
| `lighttrain convert-checkpoint --input <ckpt> --output <out> --to <fmt>` | Convert between `.pt`, `.safetensors`, and HuggingFace formats |
| `lighttrain export --config <cfg> --out <dir> --to <fmt>` | Export model weights; `gguf` requires llama.cpp on PATH |

---

## Experiment Lifecycle

### Hyperparameter Sweeps

```bash
lighttrain sweep -c base.yaml -s sweep.yaml --strategy grid    # Cartesian product
lighttrain sweep -c base.yaml -s sweep.yaml --strategy random  # random sampling
lighttrain sweep -c base.yaml -s sweep.yaml --strategy optuna  # Optuna TPE (frontier plugin)
```

Sweep configuration (`sweep.yaml`):

```yaml
name: my_sweep
metric: loss
direction: minimize
n_trials: 12
params:
  optim.lr: [1e-4, 3e-4, 1e-3]
  optim.weight_decay: {low: 0.0, high: 0.1}
stop:
  type: median
  grace: 3
```

### Comparing Runs

```bash
lighttrain compare runs/exp_a/ runs/exp_b/ [--png out.png]
```

Produces a config diff (changed fields only), a metric table (last value per
key), and fork ancestry when available.

### Checkpoint Branching

```bash
lighttrain fork --from runs/gen1/.../checkpoints/step_50 --config recipes/finetune.yaml
```

Copies or symlinks the checkpoint into a new run directory, writes
`fork_meta.json`, and registers a `fork_of` edge in the parent's lineage store.

---

## Failure Diagnostics

The failure-first subsystem ensures a crashed run answers "what failed?" and
"what do I do next?" before manual inspection.

- **Invariants** — configurable checks (`loss_finite`, `grad_norm_bounded`,
  `lr_nonneg`, `label_mask_nonzero`, `param_count_stable`, `dtype_stable`,
  `batch_nonempty`) with `abort` / `skip` / `warn` actions.
- **NaN hunter** — module forward hooks pinpoint the origin of NaN values and
  write a self-contained `repro.py`.
- **Frozen step bundles** — a single-file ZIP snapshot of model, optimizer, and
  batch state, replayable with `lighttrain replay-step`.
- **Loss attribution** — per-sample, per-token, and per-module loss breakdown.
- **OOM report** — structured report with a suggested degradation patch.
- **Realtime control** — polling of `<run_dir>/control/` for in-flight
  interventions (`lr.json`, `stop`, `eval_now`, `inject.py`).
- **`lighttrain doctor`** — aggregated diagnostics index over an entire run.

---

## Distributed Training

The `parallel:` block scales lighttrain from a single GPU to DDP, FSDP, DeepSpeed ZeRO, Tensor Parallelism, Pipeline Parallelism, Sequence Parallelism, and Expert Parallelism — no changes to model or trainer code required.

**Supported paradigms**

| Paradigm | `grad_sync.name` / strategy | Notes |
|---|---|---|
| DDP | `ddp` | All-reduce gradients; full model on every rank |
| FSDP | `fsdp` | Shards params + grads + optimizer state |
| DeepSpeed ZeRO | `deepspeed` | ZeRO-1/2/3 via DS engine |
| Tensor Parallelism | `model_parallel_strategy: tensor_parallel` | ColWise/RowWise Linear surgery |
| Pipeline Parallelism | `pipeline:` sub-block | 1F1B / GPipe schedules |
| Sequence Parallelism | `model_parallel_strategy: sequence_parallel` | Pairs with TP |
| Expert Parallelism | `ep:` degree | Sub-groups of the DP dimension |

**Launch examples**

```bash
# Single-node DDP (4 GPUs)
torchrun --nproc_per_node=4 -m lighttrain.cli train -c plugins/distributed/recipes/ddp.yaml

# TP=2 + DDP=4 (8 GPUs)
torchrun --nproc_per_node=8 -m lighttrain.cli train -c plugins/distributed/recipes/tp_ddp.yaml

# gloo + CPU — multi-process communication test (no GPU required)
torchrun --nproc_per_node=4 -m lighttrain.cli train -c plugins/distributed/recipes/nano_model.yaml
```

See [`frontier_plugins/distributed/recipes/`](plugins/distributed/recipes/) for full YAML examples (DDP, FSDP, ZeRO-2, TP+DDP, 3D parallel, gloo CPU test).
See [`docs/user_guide.md`](docs/user_guide.md) for the complete `parallel:` field reference.

---

## PEFT and Memory Efficiency

```yaml
# LoRA
model:
  name: lora
  base:
    name: hf_causal
    pretrained: meta-llama/Llama-3.2-1B
  r: 8
  target_modules: [q_proj, v_proj]

# AdaLoRA (importance-based rank pruning)
model:
  name: adalora
  base:
    name: hf_causal
    pretrained: gpt2
  r: 12
  target_r: 8
  update_interval: 200
  total_step: 2000

# QLoRA (Linux + CUDA; requires pip install -e ".[quant]")
model:
  name: qlora
  base:
    name: hf_causal
    pretrained: meta-llama/Llama-3.2-1B
  load_in_4bit: true
```

**LayerOffload** (large models on consumer hardware):

```yaml
engine:
  name: layer_offload
  resident_layers: 4
  prefetch: true
optimizer:
  name: cpu_offload
  base:
    name: adamw
```

---

## Preference Learning and Reinforcement Learning

### Offline Preference Training

```bash
lighttrain produce-artifact -c recipes/produce_teacher.yaml   # generate reference log-probabilities
lighttrain train            -c recipes/dpo_offline.yaml        # DPO fine-tuning
```

```yaml
trainer:
  name: dpo          # or: ipo | simpo | orpo | kto | reward_model
  beta: 0.1
  ref_namespace: ref
```

### Online RL (PPO / GRPO)

```yaml
trainer:
  name: ppo
  rollout_steps: 32
  ppo_epochs: 4
  clip_eps: 0.2
  grad_clip: 1.0    # optional; default 1.0

trainer:
  name: grpo
  group_size: 4
  clip_eps: 0.2
  grad_clip: 1.0    # optional; default 1.0
```

### EvalSuite

```python
from lighttrain.eval import Evaluator, RegressionGate, GenerationEvalTask, VerifierJudge

evaluator = Evaluator(
    [GenerationEvalTask(judge=VerifierJudge(verify_pattern=r"\d+"), ...)],
    eval_every_n_steps=200,
)
gate = RegressionGate(metric_name="mean_score", threshold=0.3, op=">")
```

```bash
lighttrain eval             -c <recipe> [--checkpoint <path>] [--json out.json]
lighttrain regression-gate  -c <recipe> --metric mean_score --threshold 0.3
```

`regression-gate` exits with code 1 on failure, making it suitable for CI pipelines.

---

## Alternative Architectures

Stateful (RWKV, Mamba) and non-Transformer objectives ship as frontier plugins:

```bash
lighttrain train -c recipes/pretrain_rwkv.yaml     # RWKV stateful pretraining
lighttrain train -c recipes/diffusion_eps.yaml     # diffusion eps-prediction
lighttrain train -c recipes/jepa.yaml              # JEPA masked-patch prediction
lighttrain train -c recipes/pcn_demo.yaml          # Predictive Coding Networks
lighttrain train -c recipes/ff_demo.yaml           # Forward-Forward
lighttrain train -c recipes/mezo_sft.yaml          # MeZO zero-order SFT
```

Alternative update rules:

```yaml
update_rule:
  name: mezo      # or: sam | forward_forward | pcn | dfa | standard | rl
  eps: 1e-3
```

> **`rl` update rule**: Used internally by GRPO/PPO/Preference trainers. It skips
> the standard `model(**batch)` forward (the trainer handles that) and runs only
> the backward/clip/optimizer/callbacks sequence. Supports the same three dispatch
> paths as `standard` (grad_sync, accelerator, bare).

---

## HuggingFace Integration

The `hf_causal` adapter reads `HF_TOKEN`, `HF_ENDPOINT`, and `HF_HUB_ENDPOINT`
from the environment. The CLI auto-loads a project-root `.env` file so
credentials stay out of shell profiles and version control.

```bash
cp .env.example .env   # then edit .env
```

```env
HF_TOKEN=hf_xxx
HF_ENDPOINT=https://hf-mirror.com
```

```yaml
model:
  name: hf_causal
  pretrained: meta-llama/Llama-3.2-1B
  dtype: bfloat16
```

---

## Extending

Any class that satisfies a Protocol in
[lighttrain/protocols.py](lighttrain/protocols.py) can be registered and used
without modifying core code.

### Custom optimizer

```python
from lighttrain import register

@register("optimizer", "my_adamw")
class MyAdamW:
    def __init__(self, lr=1e-3, **kw): ...
    def build(self, model): ...
    def step(self): ...
    def zero_grad(self, set_to_none=True): ...
```

### Custom callback

```python
from lighttrain import register, Signal

@register("callback", "my_oom_guard")
class MyOOMGuard:
    def on_loss_computed(self, *, loss, **_):
        if not loss.isfinite():
            return Signal.SKIP_STEP
```

Only implement the lifecycle hooks needed — `getattr` dispatch handles the
rest. The full list of 39 events is in `CALLBACK_EVENTS` in
[lighttrain/protocols.py](lighttrain/protocols.py).

### Custom PrepGraph node

```python
from lighttrain.prepgraph.node import PrepNode, NodeResult, RunContext
from lighttrain.registry import register

@register("prep_node", "my_kind")
class MyNode(PrepNode):
    kind = "my_kind"
    schema_kind = "rows"

    def run(self, ctx: RunContext) -> NodeResult:
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=...,
            store=...,
            extras={"row_count": ...},
        )
```

The `config` must produce the same fingerprint across processes; avoid
reading mutable global state (e.g. `time.time()`) in `__init__`.

---

## Documentation

| Document | Contents |
| -------- | -------- |
| [docs/user_guide.md](docs/user_guide.md) | Complete CLI reference, YAML schema, and internal initialization order |
| [docs/registry_and_protocols.md](docs/registry_and_protocols.md) | All 39 registered categories, protocol signatures, and built-in entries |

---

## License

MIT.
