# Extending lighttrain

> [中文版](extending.zh-CN.md) · [Docs index](../README.md)

Any class satisfying a Protocol in
[lighttrain/protocols.py](../../lighttrain/protocols.py) can be registered and used
without editing core code. Point a recipe's `user_modules:` at your file so the
`@register` decorators run at startup.

```yaml
user_modules:
  - my_components.py
```

The full category list, protocol signatures, and built-in entries live in
[reference/registry.md](../reference/registry.md). Quick patterns below.

## Custom loss

```python
from lighttrain import register

@register("loss", "my_loss")
class MyLoss:
    def __call__(self, model_output, batch, ctx) -> dict:
        logits = model_output.outputs["logits"]
        ...
        return {"loss": loss}   # must contain a scalar "loss" tensor
```

## Custom optimizer

The wrapper must supply `build`, `step`, `zero_grad`, **and `state_dict` /
`load_state_dict`** (the checkpoint manager calls the last two on the *wrapper*).
Subclass `_BaseWrapper` to get all four plus a default `optim_state_bytes` for
free; you only write `build()`:

```python
import torch
from lighttrain import register
from lighttrain.optim.wrappers import _BaseWrapper, _split_param_groups

@register("optimizer", "my_adamw")
class MyAdamW(_BaseWrapper):
    def build(self, model):
        self._check_unbuilt()
        self.optimizer = torch.optim.AdamW(_split_param_groups(model, self.param_groups, self._kwargs))
        self._built = True
        return self.optimizer
```

> Serializing custom optimizer state? `torch`'s `state_dict()` aliases the live
> `optimizer.state`, so rewriting it in place corrupts the running optimizer. Use
> `self._safe_state_dict(convert)` (copies first). Override `optim_state_bytes`
> so `lighttrain estimate` sees a non-Adam footprint.

## Custom callback

```python
from lighttrain import register, Signal

@register("callback", "my_cb")
class MyCB:
    def on_loss_computed(self, *, loss, **_):
        if not loss.isfinite():
            return Signal.SKIP_STEP
```

Only implement the hooks you need — `getattr` dispatch handles the rest (39
events in `CALLBACK_EVENTS`).

## Custom trainer (new paradigm)

Override the two seams on the flat `Trainer` — `produce_batch` (what a batch is)
and `forward_loss` (forward + loss) — or write a short registered `fit()` calling
the public primitives (`run_train_loop`, `apply_update`,
`forward_with_activations`). Multi-model paradigms read `self.models["..."]` /
`self.optimizers["..."]` — **declare `models=` / `optimizers=` on `__init__`** to
receive the set (see [Training § multi-model](../concepts/training.md#multi-model)). Full
worked example: [examples/online_distill.py](../../examples/online_distill.py).

## Custom PrepGraph node

```python
from lighttrain.prepgraph.node import PrepNode, NodeResult, RunContext
from lighttrain.registry import register

@register("prep_node", "my_kind")
class MyNode(PrepNode):
    kind = "my_kind"
    schema_kind = "rows"
    def run(self, ctx: RunContext) -> NodeResult:
        return NodeResult(fingerprint="", schema_kind=self.schema_kind, rows=..., store=..., extras={"row_count": ...})
```

The `config` must produce the same fingerprint across processes — avoid mutable
global state (e.g. `time.time()`) in `__init__`.

## See also

- [reference/registry.md](../reference/registry.md) — all categories & protocols
- [Architecture](../concepts/architecture.md) — the seams you're plugging into
- [Alternative architectures](architectures.md) — model-adapter rules
