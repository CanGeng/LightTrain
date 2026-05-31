# Alternative architectures

> [中文版](architectures.zh-CN.md) · [Docs index](../README.md)

Stateful (RWKV, Mamba) and non-Transformer objectives ship as plugins, selected
like any other model/objective:

```bash
lighttrain train -c recipes/pretrain_rwkv.yaml   # RWKV stateful pretraining
lighttrain train -c recipes/diffusion_eps.yaml   # diffusion eps-prediction
lighttrain train -c recipes/jepa.yaml            # JEPA masked-patch prediction
lighttrain train -c recipes/pcn_demo.yaml        # Predictive Coding Networks
lighttrain train -c recipes/ff_demo.yaml         # Forward-Forward
lighttrain train -c recipes/mezo_sft.yaml        # MeZO zero-order SFT
```

Built-in objectives carry a `loss_family`: `next_token`, `masked_denoising`
(`mlm`), `diffusion`, `flow_matching`, `jepa`. Alternative update rules:
`mezo`, `sam`, `forward_forward`, `pcn`, `dfa` (set `update_rule.name`).

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
