# Troubleshooting

> [中文版](troubleshooting.zh-CN.md) · [Docs index](../README.md)

## Common errors

| Symptom | Likely cause / fix |
| ------- | ------------------ |
| `recipe is missing 'model:'/'data:'/'optim:' section` | Add the required section — see [Configuration](../guide/configuration.md). |
| `AttributeError` at the first `ckpt_every` (custom optimizer) | Your wrapper lacks `state_dict` / `load_state_dict`; subclass `_BaseWrapper` — see [Extending](extending.md). |
| Custom trainer never sees the teacher / second model | `__init__` must declare `models=` / `optimizers=` to receive the set — see [Training § multi-model](../concepts/training.md#multi-model). |
| `UserWarning: dropped recipe key …` on a model | A stray cross-architecture key; expected with explicit signatures — see [Alternative architectures](architectures.md). |
| Mid-epoch resume fails `resume-verify` | Use fp32 + single worker for bit-exact checks; otherwise pass `--tol`. |
| Checkpoint dir ignored on resume | It lacks `manifest.json` (written last) → treated as incomplete. |
| Loss is NaN | Check `nan_hunter` output + `repro.py` under `diagnostics/`; see [Diagnostics](../operations/diagnostics.md). |

## Known third-party limitations

These live in upstream packages, not in lighttrain (encountered while
reproducing Mamba-3):

- **`state-spaces/mamba`** — `mixer_seq_simple.create_block` only whitelists
  `Mamba1`/`Mamba2`. The module-level adapter pattern
  ([Alternative architectures](architectures.md)) sidesteps it: instantiate
  `Mamba3` yourself and never hit `create_block`.
- **`state-spaces/mamba`** — Mamba-2's fast path (`use_mem_eff_path=True`)
  needs the `causal_conv1d` CUDA extension. Set `use_mem_eff_path=False` for the
  pure-Triton fallback (works out of the box, ~3× slower).
- **`tilelang==0.1.8` + `apache-tvm-ffi==0.1.11`** — crashes in
  `NestedLoopChecker` during MIMO chunk-bwd lowering. For GatedDeltaNet set
  `FLA_TILELANG=0` (Triton backend). Mamba-3 MIMO has no fallback in that env —
  use the SISO variant.

## Eager-import failures

If a research package fails to import because its `__init__` drags in heavy
siblings (a CUDA C-extension, an optional pure-Python dep), pre-seed a stub
module before importing — see the eager-import tip in
[Alternative architectures](architectures.md).

## See also

- [Diagnostics](../operations/diagnostics.md) — failure-first tooling
- [Extending](extending.md) — registration contracts
