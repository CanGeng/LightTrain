# Recipe index

> [中文版](recipes.zh-CN.md) · [Docs index](../README.md)

The fastest way to start: copy a bundled recipe from
[`recipes/`](../../recipes/) and edit it. Run any of them with
`lighttrain train -c recipes/<name>.yaml`.

## Pretraining & SFT

| Recipe | Demonstrates |
| ------ | ------------ |
| `pretrain_causal` | Causal-LM pretraining (tiny_lm + byte tokenizer) — the canonical starting point |
| `pretrain_rwkv` | RWKV stateful pretraining |
| `sft_chat` | SFT over chat data via PrepGraph |
| `sft_chat_hf` | SFT using a HuggingFace model/tokenizer |
| `vlm_sft` | Vision-language SFT (multimodal collator) |

## Preference & RL

| Recipe | Demonstrates |
| ------ | ------------ |
| `dpo_offline` | Offline DPO (the `loss:` seam under `preference`) |
| `ppo_online` | Online PPO with rollout + GAE + verifier judge |
| `grpo` | Group Relative Policy Optimization |
| `produce_teacher` | Produce reference/teacher artifacts |

## Distillation

| Recipe | Demonstrates |
| ------ | ------------ |
| `student_kd` | Teacher → student knowledge distillation |
| `online_distill_demo` | Two-model online distillation (student rolls out vs a frozen teacher) — the multi-model seam, see [examples/online_distill.py](../../examples/online_distill.py) |

## Alternative objectives & update rules

| Recipe | Demonstrates |
| ------ | ------------ |
| `diffusion_eps` | Diffusion eps-prediction objective |
| `jepa` | JEPA masked-patch prediction |
| `pcn_demo` | Predictive Coding Networks |
| `ff_demo` | Forward-Forward |
| `mezo_sft` | MeZO zero-order SFT (memory-efficient) |

## Memory efficiency

| Recipe | Demonstrates |
| ------ | ------------ |
| `qlora` | QLoRA 4-bit fine-tuning (Linux + CUDA) |
| `offload_fullparam` | LayerOffload + CPU-offload optimizer |

## Experiment lifecycle

| Recipe | Demonstrates |
| ------ | ------------ |
| `fork_resume` | Fork from a checkpoint + resume |
| `sweep_lr` | Learning-rate sweep |
| `sweep_demo` | General sweep config |
| `sweep_r15` | Sweep with early-stopping rules |

## See also

- [Getting started](../guide/getting-started.md) — run your first recipe
- [Training paradigms](../concepts/training.md) — what each recipe shape means
- [Configuration](../guide/configuration.md) — edit recipe fields confidently
