# Alternative architectures

> [中文版](architectures.zh-CN.md) · [Docs index](../README.md)

Stateful (RWKV, Mamba) and non-Transformer objectives ship as plugins, selected
like any other model/objective:

```bash
lighttrain train -c examples/references/recipes/pretrain_rwkv.yaml   # RWKV stateful pretraining
lighttrain train -c examples/references/recipes/diffusion_eps.yaml   # diffusion eps-prediction
lighttrain train -c examples/references/recipes/jepa.yaml            # JEPA masked-patch prediction
lighttrain train -c examples/references/recipes/pcn_demo.yaml        # Predictive Coding Networks
lighttrain train -c examples/references/recipes/ff_demo.yaml         # Forward-Forward
lighttrain train -c examples/references/recipes/mezo_sft.yaml        # MeZO zero-order SFT
```

Built-in objectives carry a `loss_family`: `next_token`, `masked_denoising`,
`diffusion`, `flow_matching`, `jepa`. Alternative update rules:
`mezo`, `sam`, `forward_forward`, `pcn`, `dfa` (set `engine.update_rule.name` —
a top-level `update_rule:` is rejected with a clear error).

## Writing a custom trainer (objective-seam contract)

A trainer declares how it relates to the canonical `objective` seam via three
class attributes (the runtime reads and enforces them after construction):

```python
class MyTrainer(Trainer):
    consumes_objective = True          # uses objective.__call__ as the loss (ctx.loss_fn)
    consumes_objective_prepare = True  # runs objective.prepare_batch before the forward
    requires_objective = False         # True ⇒ recipe MUST name a loss/objective (no default)
    def default_objective(self): ...   # used when consuming + recipe omits loss/objective
```

- **Inline-algorithm** trainers (compute their own loss, e.g. reward-model
  Bradley-Terry, online-distill REINFORCE) **must** set
  `consumes_objective = False`; the runtime then errors if a recipe hands them a
  `loss:`/`objective:`.
- A trainer that uses the objective as a loss but brings its own batches (RL /
  preference) sets `consumes_objective_prepare = False`; the runtime then rejects
  a *real* `objective:` (with a non-trivial `prepare_batch`) — a plain `loss:` is
  always fine.

## Writing a custom engine

The engine receives `loss_fn` at construction, but it may be `None`: the default
objective is resolved *after* the trainer is built and the runtime then back-fills
`engine.loss_fn = trainer.objective`. So an engine `__init__` **must tolerate
`loss_fn=None`** (read `ctx.loss_fn` at step time, as `StandardEngine` does).

## Writing model adapters (two rules)

When wrapping a third-party architecture (SSMs, FLA, …), both rules matter.

**Rule 1 — import the lowest-level module, not a high-level factory.**

```python
from mamba_ssm.modules.mamba3 import Mamba3   # good — survives upstream refactors
# not: from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
#      MambaLMHeadModel(..., ssm_layer="Mamba3")   # may carry hidden whitelists
```

High-level factories often carry internal whitelists, version assumptions, or
implicit C-extension deps. Module-level assembly keeps you in control.

**Rule 2 — declare an explicit signature on the registered class** (no
`def __init__(self, **kwargs)`):

```python
@register("model", "mamba2_lm")
class Mamba2LM(_MambaLMAdapter):
    def __init__(self, *, d_model: int, n_layer: int, vocab_size: int,
                 d_state: int = 128) -> None:        # explicit, named
        super().__init__(layer="Mamba2", d_model=d_model, n_layer=n_layer, ...)
```

The resolver (`_filter_kwargs`) drops unknown recipe kwargs **by the registered
class's signature**. A `**kwargs` class semantically claims "I want every key",
so the filter becomes a no-op and a stray cross-architecture key leaks through.
With an explicit signature, stray keys are dropped with a `UserWarning`.

**Escape hatch** — if you genuinely need `**kwargs` for inner forwarding, set
`__lighttrain_filtered_kwargs__ = True` on the class; the resolver then filters
against your explicit params and still blocks recipe-side leaks.

**Eager-import tip** — many research repos drag heavy siblings into their
top-level `__init__`. Pre-seed a stub module before importing, from your
`user_modules` file:

```python
import sys, types
sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
import mamba_ssm   # safe now
# or skip a package __init__ by importing the submodule directly after stubbing the parent
```

See [Troubleshooting](troubleshooting.md) for the concrete mamba/tilelang cases.

## See also

- [Extending](extending.md) — the full registration contract
- [Troubleshooting](troubleshooting.md) — known third-party limitations
- [reference/registry.md](../reference/registry.md) — model / objective entries
