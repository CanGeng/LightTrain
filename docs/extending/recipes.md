# Recipe index

> [中文版](recipes.zh-CN.md) · [Docs index](../README.md)

The fastest way to start: copy a bundled recipe from
[`recipes/`](../../recipes/) and edit it. Most run with
`lighttrain train -c recipes/<name>.yaml`; the distributed overlays (below) are
the exception — they layer onto a full recipe and need a multi-process launcher.

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

## Distributed

These demonstrate parallel topologies. **`ddp` / `fsdp` / `zero2` / `tp_ddp` /
`3d_parallel` are overlays** — they carry only the `parallel:` / `engine:` /
`trainer:` topology (no `model:` / `data:`), so they must be layered onto a
complete recipe (e.g. `pretrain_causal`) and launched with a multi-process
launcher (`torchrun` / `deepspeed`), not a single-process `lighttrain train`.

| Recipe | Demonstrates |
| ------ | ------------ |
| `ddp` | Single-node 4-GPU DDP data parallelism (overlay) |
| `fsdp` | FSDP full sharding (overlay) |
| `zero2` | ZeRO-2 optimizer sharding (overlay) |
| `tp_ddp` | Tensor parallel × DDP (overlay) |
| `3d_parallel` | TP × PP × DP 3-D parallelism (overlay) |
| `nano_model` | gloo + CPU multi-process smoke test (complete recipe; `torchrun --nproc_per_node 4`) |

## See also

- [Getting started](../guide/getting-started.md) — run your first recipe
- [Training paradigms](../concepts/training.md) — what each recipe shape means
- [Configuration](../guide/configuration.md) — edit recipe fields confidently
