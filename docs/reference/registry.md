# Registry & protocols reference

> [中文完整版 / full Chinese table](registry.zh-CN.md) · [Docs index](../README.md)

This is the English quick reference. For the exhaustive built-in-entry tables and
per-category minimal examples, see the [full Chinese table](registry.zh-CN.md).

## Registration

```python
from lighttrain.registry import (
    register, get, list_entries, categories, register_category, contains, unregister,
)

@register("optimizer", "my_optim")        # decorator (most common)
class MyOptim: ...

@register("loss", "ce")                    # alias: stack decorators
@register("loss", "cross_entropy")
class CrossEntropyLoss: ...

@register("model", "hf_causal", force=True)  # override a built-in
class Patched: ...

register("model", "my_model", MyModelClass)  # function form
register_category("my_plugin_category")      # add a category beyond the built-in set
```

Source: [lighttrain/registry/_core.py](../../lighttrain/registry/_core.py).

## Categories (35 `KNOWN_CATEGORIES`)

| Group | Categories (YAML mount) |
| ----- | ----------------------- |
| Core | `model` (`model:`), `loss` (`loss:`), `optimizer` (`optimizer:`), `scheduler` (`scheduler:`), `dataset` (`data.dataset:`), `processor` (`data.processor:`), `collator` (`data.collator:`), `sampler` (`data.sampler:`) |
| Orchestration | `trainer`, `engine`, `update_rule`, `callback` (list), `metric`, `logger` (list), `objective`, `architecture` |
| Frontier | `generation_strategy`, `judge`, `environment`, `retriever`, `chunker`, `probe` |
| Artifact & data | `artifact_producer`, `artifact_store`, `prep_node` (list), `data_module` (`data:`), `tokenizer` |
| Failure-first & RL | `invariant` (list), `rl_backend`, `value_head`, `reward_adapter` |
| Distributed | `grad_sync_strategy`, `model_parallel_strategy`, `pipeline_schedule` |
| Sweep | `sweep_backend` (`sweep --strategy optuna`) |

## Shared dataclasses (`lighttrain/protocols.py`)

```python
@dataclass
class ModelOutput:
    outputs: dict[str, torch.Tensor]      # logits / eps / recon …
    loss: torch.Tensor | None
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    extras: dict[str, torch.Tensor]
    state: Any | None                     # stateful arch (RWKV/Mamba)

@dataclass
class LossContext:
    step: int; epoch: int
    metrics: dict[str, float]
    loss_family: str | None
    extras: dict[str, Any]

@dataclass
class StepOutput:
    loss: Any | None
    metrics: dict[str, Any]               # includes a "loss" key
    logs: dict[str, Any]
    extras: dict[str, Any]
```

## Key protocol signatures

| Category | Required interface |
| -------- | ------------------ |
| `model` | `forward(**batch) -> ModelOutput`; for RL also `generate(input_ids, **kw) -> Tensor` |
| `loss` | `__call__(model_output, batch, ctx) -> dict` (must contain `"loss"`) |
| `optimizer` | `build(model)`, `step`, `zero_grad`, `state_dict`, `load_state_dict` (+ optional `optim_state_bytes`); subclass `_BaseWrapper` for all but `build` |
| `scheduler` | `step_per_batch: bool`, `step`, `state_dict`, `load_state_dict`; subclass `_SchedulerBase`, implement `_factor(step)` |
| `dataset` | duck-typed `__len__`, `__getitem__(idx)` |
| `collator` | `__call__(samples) -> dict[str, Tensor]` |
| `sampler` | `__iter__`, `__len__`, `state_dict`, `load_state_dict` |
| `tokenizer` | `encode(text) -> list[int]`, `decode(ids) -> str` |
| `data_module` | `train_loader`, `val_loader`, `predict_loader`, `state_dict`, `load_state_dict` |
| `trainer` | flat `Trainer`; override seams `produce_batch` / `forward_loss` / `before_step`, or a registered `fit()` |
| `engine` | `step(batch, ctx) -> dict` |
| `update_rule` | `setup(model, sample)`, `step(model, batch, ctx) -> dict` (with `"loss"`), `state_dict`, `load_state_dict` |
| `callback` | any of 46 `CALLBACK_EVENTS` hooks; may return a `Signal` |
| `logger` | `log_scalars`, `log_histograms`, `log_text`, `log_artifact`, `flush` |
| `judge` | `score(items, ctx=None) -> list` (declares `reward_kind`) |
| `rl_backend` | `generate(model, input_ids, **kw) -> Tensor` |
| `value_head` | `linear` — per-token (PPO critic) or last-token (RM head) |
| `reward_adapter` | wraps a judge into `reward_fn(prompt_ids, response_ids) -> list[float]` |
| `prep_node` | subclass `PrepNode`; `kind`, `schema_kind`, `run(ctx) -> NodeResult` |
| `invariant` | callable `(*, loss, batch, metrics, model, step, **kw) -> bool` |

## Built-in entries (condensed)

- **models**: `tiny_lm`, `hf_causal`, `lora`, `ia3`, `adalora`, and plugin `jepa`, `qlora`, `tiny_rwkv`, `tiny_mamba`, `tiny_unet`
- **objectives** (all plugin; the Protocol `ObjectiveProfile` is core): `next_token`, `masked_denoising`, `diffusion`, `flow_matching`, `jepa`
  - `objective:` is the **internal canonical training seam** (owns `prepare_batch` + loss). When present it replaces `loss:`; a plain `loss:` is wrapped into a `LossOnlyObjective` (identity `prepare_batch`) so there is one seam. `loss:` and top-level `objective:` are mutually exclusive (an objective may carry a *nested* `loss:`/`aux_losses:`). The default objective belongs to the **trainer** (`Trainer.default_objective()`), not the runtime.
- **architectures** (profile factories, by `trainer.arch_profile`): `transformer` (core), `rwkv` (plugin) — resolve a string to an `ArchitectureProfile` (block/embedding/head seams + stateful reset).
- **losses**: `cross_entropy`/`ce`, `mlm`, `z_loss`, `composite`, `dpo`, `bradley_terry`/`bt`, `ipo`, `simpo`, `orpo`, `kto`, `ppo_surrogate`, `grpo`, `info_nce`, `moe_balance`, `kl_topk`, `hidden_mse`, `hidden_cosine`, `attention_transfer`
- **optimizers**: `adamw`, `lion`, and plugin `cpu_offload` · **schedulers**: `constant`, `linear`, `warmup_cosine`, `wsd`
- **update_rules**: `standard`, `sam`, `mezo`, `rl` (internal), and plugin `forward_forward`, `pcn`, `dfa`
- **trainers**: `pretrain`, `preference`, `reward_model`, `ppo`, `grpo`
- **datasets**: `line_file_text`, `preference_jsonl`, `artifact_joined` · **collators**: `causal_lm`, `preference`, `multimodal` · **samplers**: `shuffle`, `sequential`, `length_grouped`, `curriculum`, `stateful_resumable` · **tokenizers**: `byte`
- **prep_node**: `load`, `tokenize`, `chunk`, `pack`, `mix`, `join`, `index`, `validate`, `materialize`
- **callbacks**: `ema`, `best_ckpt`, `throughput`, `early_stop`, `nan_skip`, `invariants`, `nan_hunter`, `frozen_step`, `loss_attribution`, `dead_neuron`, `grad_flow`, `sample_preview`, `dynamic_artifact`, `lineage_recorder`, `file_signals`
- **loggers**: `console`, `jsonl`, `tensorboard`/`tb` · **judges**: plugin `verifier`, `pairwise_llm` · **rl_backend**: `hf_generate`, and plugin `vllm`
- **grad_sync**: `noop`, `ddp`, `fsdp`, `deepspeed` · **model_parallel**: `tensor_parallel`, `tp_aware`, `sequence_parallel`*, `expert_parallel`* · **pipeline**: `1f1b`, `gpipe`, `interleaved_1f1b`
  - *`sequence_parallel` / `expert_parallel` are registered but **not yet wired into the train runtime** (the selector only picks `tensor_parallel`; EP is a skeleton). See the v0.2.3 changelog "Known issues".*
- **sweep_backend**: plugin `optuna` (requires `pip install -e '.[sweep]'`)

## Exceptions

`RegistryError` (base) · `RegistryConflictError` (re-register without `force`) ·
`UnknownCategoryError` (category not declared) · `NotRegisteredError` (name not found).

For full per-category examples and the complete built-in tables, see the
[full Chinese reference](registry.zh-CN.md).
