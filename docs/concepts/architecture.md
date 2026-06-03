# Architecture

> [中文版](architecture.zh-CN.md) · [Docs index](../README.md)

lighttrain defines five clean seams; everything else stays straightforward.

## The five seams

1. **Registry** ([lighttrain/registry/_core.py](../../lighttrain/registry/_core.py)) —
   short-name → class resolution over a pre-declared category set (`model`,
   `loss`, `optimizer`, `dataset`, `trainer`, `engine`, `update_rule`, `judge`,
   `rl_backend`, `prep_node`, …). See [Extending](../extending/extending.md).

2. **Config** ([lighttrain/config/](../../lighttrain/config)) — OmegaConf loading
   + Pydantic v2 schema. The model is chosen by a config group
   (`model_profiles:` + `model: <name>`). See [Configuration](../guide/configuration.md).

3. **Engine + UpdateRule** ([engine/](../../lighttrain/engine),
   [update_rules/](../../lighttrain/update_rules)) — the engine owns the
   accelerator (mixed precision, device) and delegates per-step math
   (forward / backward / clip / step / scheduler) to a swappable `UpdateRule`,
   so research code can change the training math without touching the engine.

   **Trainer primitives** ([trainers/](../../lighttrain/trainers)) — the flat
   `Trainer` has a concrete `fit()` built from public, re-entrant primitives:
   `run_train_loop`, `apply_update`, `forward_with_activations`. The 90% case is
   pure YAML; a new paradigm overrides `produce_batch` / `forward_loss` (and
   optionally `before_step`) or writes a short registered `fit()`. See
   [Training](training.md) and [Extending](../extending/extending.md).

4. **EventBus** ([callbacks/base.py](../../lighttrain/callbacks/base.py)) — 46
   lifecycle events dispatched via `getattr`; per-callback exceptions isolated;
   results aggregate to a `Signal` (`STOP_TRAINING > RETRY_STEP > SKIP_STEP >
   CONTINUE`). See [Diagnostics](../operations/diagnostics.md).

5. **PrepGraph** ([prepgraph/](../../lighttrain/prepgraph)) — content-addressed
   DAG of data-prep nodes; fingerprint =
   `sha256(config + code_version + schema_version + sorted upstream_fps)`;
   results land atomically with `MANIFEST_COMPLETE.json` written last. See
   [Data & PrepGraph](data-prepgraph.md).

## Initialization order (`train`)

`setup_run_from_config` then `trainer.fit()`:

1. **Config load** — YAML → defaults → overrides → interpolation → Pydantic.
2. **Prepare** — import `user_modules` (decorators register), `seed_everything`,
   create run dir, write `config.snapshot.yaml` + `env.json`.
3. **Components (strict order)**
   - **A — topology**: `parallel_ctx` (single-GPU fallback or process group +
     DeviceMesh); `device` derived from it.
   - **B — model + TP surgery** (must precede FSDP/DDP wrapping — sharding
     needs the post-surgery shapes; SP/EP would slot in here once wired —
     today they are rejected with a `ConfigError`).
   - **C — pipeline split** (when `pp > 1`).
   - **D — grad-sync wrap** (`noop`/`ddp`/`fsdp`/`deepspeed`; FSDP builds the
     optimizer *after* wrapping via an `optimizer_factory`).
   - **common**: data_module → scheduler → loss → callbacks → logger → ckpt.
4. **Engine assembly** — update_rule + accelerator + loss into `StandardEngine`.
5. **Trainer assembly** — the runtime always passes `model` / `models` /
   `optimizers` to the trainer; lab-mode diagnostics callbacks auto-attach.
6. **Training loop** — `trainer.fit()` runs the epoch/signal/log/ckpt loop.

Key invariants: topology before everything; TP/SP/EP before FSDP/DDP; FSDP
optimizer built post-wrap; rank-0 gates checkpoint and logging IO (run-metadata
init and PrepGraph are not yet fully rank-0-coordinated); `loss_fn` is a separate
component (swappable via `loss:`); `manifest.json` written last.

## See also

- [Extending](../extending/extending.md) — implement against these seams
- [Training paradigms](training.md) — the Engine/Trainer seam in practice
- [Distributed](../operations/distributed.md) — seam A/B/C/D in detail
