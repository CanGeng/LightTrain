# Diagnostics (failure-first)

> [中文版](diagnostics.zh-CN.md) · [Docs index](../README.md)

The failure-first subsystem ensures a crashed run answers "what failed?" and
"what next?" before manual inspection. Most of it auto-attaches in `lab` mode.

## Building blocks

- **Invariants** — per-step checks with `abort` / `skip` / `warn` actions:
  `loss_finite`, `grad_norm_bounded`, `lr_nonneg`, `label_mask_nonzero`,
  `param_count_stable`, `dtype_stable`, `batch_nonempty`. Configure via
  `invariants:`.
- **NaN hunter** — module forward hooks pinpoint the origin of a NaN and write a
  self-contained `repro.py`.
- **Frozen step bundles** — single-file ZIP snapshot of model + optimizer +
  batch, replayable with `replay-step`. Auto-captured every 1000 steps in lab.
- **Loss attribution** — per-sample, per-token, per-module loss breakdown.
- **OOM report** — structured report with a suggested degradation patch.
- **Realtime control** — poll `<run_dir>/control/` for in-flight interventions
  (`lr.json`, `stop`, `eval_now`, `inject.py`) via the `file_signals` callback.
- **Callback isolation** — per-callback exceptions are caught and written to
  `diagnostics/callback_failures.jsonl` so one bad callback can't kill the run.

## Commands

```bash
lighttrain doctor      --run runs/exp/<...>          # aggregated diagnostics index
lighttrain replay      --run runs/exp/<...>          # replay the latest crash / frozen step
lighttrain freeze-step --run runs/exp/<...> --step N # capture a single-step bundle
lighttrain replay-step bundle.zip                    # replay a frozen step
```

## EventBus signals

Callbacks can return a `Signal` to steer the loop; results aggregate by
precedence: `STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`. Example:

```python
from lighttrain import register, Signal

@register("callback", "my_oom_guard")
class MyOOMGuard:
    def on_loss_computed(self, *, loss, **_):
        if not loss.isfinite():
            return Signal.SKIP_STEP
```

## See also

- [Architecture § EventBus](../concepts/architecture.md) — the 39 lifecycle events
- [Extending](../extending/extending.md) — write custom callbacks / invariants
- [CLI](../guide/cli.md) — `doctor` / `replay` / `freeze-step`
