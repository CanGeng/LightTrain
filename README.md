# lighttrain

> English · [中文](README.zh-CN.md)

A PyTorch language-model training framework for fast research iteration:
pretraining, SFT, preference learning, online RL, and distillation. The core —
Registry, Config, Engine, UpdateRule, Trainer, EventBus, PrepGraph — is small
enough to read end-to-end; research-grade extras (PEFT, alternative
architectures, sweeps, distributed) are opt-in.

Design goals: **registry-first**, **failure-first**, **plugin-clean**,
**lab-friendly**, **audit-ready**.

> Status: testing phase. Distributed is **data-parallel only** (DDP / FSDP /
> DeepSpeed ZeRO); DDP, FSDP, and DeepSpeed ZeRO-2 are validated on a real
> single-node multi-GPU box (NCCL), but **not** on multi-node GPU clusters — use
> at your own risk for production. (Tensor / pipeline / expert / sequence
> parallelism were removed.) The test suite is ~82K lines /
> 4,400+ tests with adversarial regression tests verified by mutation testing.

## Install

```bash
git clone <this-repo> lighttrain && cd lighttrain
pip install -e .
pip install -e ".[peft]"          # optional: LoRA / IA³ / AdaLoRA
pip install -e ".[peft,quant]"    # optional: + bitsandbytes 4-bit (Linux+CUDA)
```

## quickstart

```bash
lighttrain init my_project        # scaffold a commented, runnable recipe
cd my_project
lighttrain dry-run -c cfg.yaml    # resolve & print the config (no training)
lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50-step smoke run
```

The generated `cfg.yaml` runs once you add a `corpus.txt` (one example per line)
and is heavily commented as a living tutorial — uncomment the optional blocks
(`models:`, `parallel:`, `prep_graph:`, PEFT…) to grow it. → [Getting started](docs/guide/getting-started.md)

## Architecture

1. **Registry** — short-name → class resolution over a fixed category set.
2. **Config** — OmegaConf + Pydantic v2; the model is a config group
   (`model_profiles:` + `model: <name>`).
3. **Engine + UpdateRule** — the engine owns the accelerator and delegates the
   per-step math (forward/backward/clip/step) to a swappable `UpdateRule`, so you
   can change the training math without touching the loop. The flat `Trainer`
   composes public primitives (`run_train_loop`, `apply_update`,
   `forward_with_activations`).
4. **EventBus** — 46 lifecycle events; isolated per-callback exceptions; results
   aggregate to a `Signal` (`STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`).
5. **PrepGraph** — content-addressed DAG of data-prep nodes; cached by a
   fingerprint over config + code + schema + upstream.

→ [Architecture](docs/concepts/architecture.md)

## Example: train, then branch & resume

```bash
lighttrain train -c examples/references/recipes/pretrain_causal.yaml
lighttrain fork  --from runs/<...>/checkpoints/step_500 -c examples/references/recipes/finetune.yaml
lighttrain resume --run runs/<...>
```

## Example: a new paradigm in one recipe

The new algorithm is usually the `loss:`, not a new trainer. Online RL:

```yaml
trainer: { name: ppo, rollout_steps: 32, rollout_backend: hf_generate }
loss:    { name: ppo_surrogate, clip_eps: 0.2 }
judge:   { name: verifier, verify_pattern: "\\d+" }   # → reward_fn
```

Multi-model (a frozen teacher + a trainable student) is a named model set; a
custom trainer reads `self.models["teacher"]`. A runnable end-to-end template:
[examples/online_distill.py](examples/online_distill.py)
(`lighttrain train -c examples/references/recipes/online_distill_demo.yaml`).
→ [Training paradigms](docs/concepts/training.md)

## What you get

A self-contained run capsule under `runs/<exp>/<ts>-<slug>-<hash>/`: config
snapshot + resolved config, `env.json`, `logs/metrics.jsonl`, and `checkpoints/`
(`manifest.json` written last = the completeness marker). → [Getting
started](docs/guide/getting-started.md#what-you-get)

## Built-in components (at a glance)

| Kind | Names |
| ---- | ----- |
| Models | `tiny_lm`, `hf_causal`, `tiny_rwkv`, `tiny_mamba`, `jepa`, + PEFT `lora`/`ia3`/`adalora` |
| Trainers | `pretrain`, `preference`, `reward_model`, `ppo`, `grpo` |
| Losses | `cross_entropy`, `dpo`/`ipo`/`simpo`/`orpo`/`kto`, `ppo_surrogate`, `grpo`, `kl_topk`, … |
| Optimizers | `adamw`, `lion` · Schedulers `constant`/`linear`/`warmup_cosine`/`wsd` |
| Data | datasets, collators, samplers, byte tokenizer, PrepGraph nodes |
| Diagnostics | invariants, nan_hunter, frozen_step, loss_attribution, `doctor` |

All of these are concrete `@register` implementations and ship in
`lighttrain.builtin_plugins` (core keeps only protocols + framework); they're
resolved by short name regardless of where the code lives. Full tables:
[Registry & protocols](docs/reference/registry.md).

## Documentation

Everything lives under [`docs/`](docs/README.md) (English + 中文, split by topic):

- [Getting started](docs/guide/getting-started.md) · [CLI](docs/guide/cli.md) ·
  [Configuration](docs/guide/configuration.md)
- [Architecture](docs/concepts/architecture.md) · [Training](docs/concepts/training.md) ·
  [Data & PrepGraph](docs/concepts/data-prepgraph.md)
- [Distributed](docs/operations/distributed.md) · [Diagnostics](docs/operations/diagnostics.md)
- [Alternative architectures](docs/extending/architectures.md) ·
  [Extending](docs/extending/extending.md) · [Recipes](docs/extending/recipes.md) ·
  [Troubleshooting](docs/extending/troubleshooting.md)
- Reference: [Registry & protocols](docs/reference/registry.md)

## License

MIT. Built with the assistance of Claude Code; architecture, test design, and
quality gates are human-directed.
