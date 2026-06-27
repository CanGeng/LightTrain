# Examples

Runnable, end-to-end entry points for lighttrain.

| Directory | What it holds |
|-----------|---------------|
| [`references/`](references) | One minimal script + recipe per built-in capability — the canonical "how do I use feature X" reference set (pretraining, SFT, DPO/PPO/GRPO, distillation, QLoRA, offload, RWKV/diffusion/JEPA, sweeps, fork-resume). The recipes (YAML) live under [`references/recipes/`](references/recipes). |

Faithful local ports of well-known training repos (e.g. nanoGPT, MiniMind) land
at the top level of this directory, one self-contained folder each.

Run any recipe via the CLI:

```bash
lighttrain train -c examples/references/recipes/pretrain_causal.yaml
```
