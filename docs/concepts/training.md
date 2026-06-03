# Training paradigms

> [中文版](training.zh-CN.md) · [Docs index](../README.md)

One flat `Trainer` plus seam overrides covers pretraining, SFT, preference,
online RL, distillation, and multi-model setups. The algorithm is usually the
`loss:` seam, not a separate trainer.

## Pretraining & SFT

`trainer: pretrain` is the bare flat `Trainer` — pure YAML, no subclass. SFT is
the same trainer over chat/instruction data (often via PrepGraph). See
[recipes](../extending/recipes.md): `pretrain_causal`, `sft_chat`, `sft_chat_hf`.

## Offline preference (DPO / IPO / SimPO / ORPO / KTO)

One `preference` trainer; the algorithm is the `loss:` seam:

```yaml
trainer: { name: preference, ref_namespace: ref }
loss:    { name: dpo, beta: 0.1 }      # or ipo | simpo | orpo | kto
```

Typical flow: `produce-artifact` to cache reference log-probs, then `train`.
`reward_model` is a separate trainer (Bradley-Terry) with a pluggable
`value_head:`. Recipes: `dpo_offline`, `produce_teacher`.

## Online RL (PPO / GRPO)

The RL loss is the `loss:` seam; the rollout backend resolves from the
`rl_backend` registry with full sampling knobs:

```yaml
trainer:
  name: ppo
  rollout_steps: 32
  ppo_epochs: 4
  rollout_backend: hf_generate
  temperature: 1.0
  top_p: 1.0
loss: { name: ppo_surrogate, clip_eps: 0.2 }
```

```yaml
trainer: { name: grpo, group_size: 4, buffer_max_size: 4096 }
loss:    { name: grpo, clip_eps: 0.2, beta_kl: 0.0 }
```

The reward comes from a `judge:` (e.g. `verifier`) wrapped into a `reward_fn`
through a `reward_adapter` (pointwise by default). Recipes: `ppo_online`, `grpo`.

## Distillation

Teacher → student. Produce teacher artifacts then train the student with a
distillation `loss:` (`kl_topk`, `hidden_mse`, `hidden_cosine`,
`attention_transfer`). Recipes: `produce_teacher`, `student_kd`.

## Multi-model

A run can declare a **named set of models**. `trainable: false` entries are
frozen, on-device, optionally checkpoint-loaded auxiliaries (a distillation
teacher, a frozen reference); each `trainable: true` entry gets its own
optimizer (GAN / actor-critic). A lone `model:` / `optim:` is sugar for a
one-entry set.

```yaml
models:
  student: { spec: { profile: small }, trainable: true, optimizer: opt_main }
  teacher: { spec: { name: hf_causal, pretrained: gpt2 }, trainable: false,
             checkpoint: path/to/teacher.safetensors }
optimizers:
  opt_main: { name: adamw, lr: 1.0e-3 }
```

A custom trainer reaches them as `self.models["teacher"]` /
`self.optimizers["student"]`.

> **Mechanism that bites:** the runtime always passes `models=` / `optimizers=`
> to the trainer, but the resolver keeps them only if the trainer's `__init__`
> **declares those parameters**. So a custom multi-model trainer must declare
> `models=` / `optimizers=` to receive the set — the built-in `ppo`/`grpo` don't,
> which is why they can't host a second model.

Runnable end-to-end example: a student rolling out on-policy against a frozen
teacher — [examples/online_distill.py](../../examples/online_distill.py)
(`lighttrain train -c recipes/online_distill_demo.yaml`).

## PEFT & memory efficiency

Each adapter is a model profile (`lora` / `ia3` / `adalora` / `qlora`); `LayerOffload`
engine + `cpu_offload` optimizer fit large models on consumer GPUs. Recipes:
`qlora`, `offload_fullparam`, `mezo_sft`.

## See also

- [Configuration](../guide/configuration.md) — recipe field reference
- [Extending](../extending/extending.md) — write a custom trainer / loss
- [Recipe index](../extending/recipes.md) — runnable starting points
