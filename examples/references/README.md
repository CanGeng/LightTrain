# Reference examples

One minimal Python script per recipe demonstrating the programmatic API. The
matching recipes (YAML configs) live in [`recipes/`](recipes); each script can
also be run directly, but the CLI is the recommended entry point for day-to-day
use:

```bash
lighttrain train -c examples/references/recipes/pretrain_causal.yaml
```

| Script | Recipe | What it shows |
|--------|--------|---------------|
| [pretrain_causal.py](pretrain_causal.py) | `pretrain_causal.yaml` | Basic causal LM pretraining |
| [sft_chat.py](sft_chat.py) | `sft_chat.yaml` | Supervised fine-tuning |
| [dpo_offline.py](dpo_offline.py) | `dpo_offline.yaml` | Offline DPO preference training |
| [ppo_online.py](ppo_online.py) | `ppo_online.yaml` | Online PPO RL training |
| [grpo.py](grpo.py) | `grpo.yaml` | GRPO group-relative optimization |
| [student_kd.py](student_kd.py) | `student_kd.yaml` | Knowledge distillation |
| [produce_teacher.py](produce_teacher.py) | `produce_teacher.yaml` | Teacher logit artifact production |
| [qlora.py](qlora.py) | `qlora.yaml` | QLoRA 4-bit fine-tuning |
| [offload_fullparam.py](offload_fullparam.py) | `offload_fullparam.yaml` | LayerOffload full-parameter training |
| [pretrain_rwkv.py](pretrain_rwkv.py) | `pretrain_rwkv.yaml` | Stateful RWKV pretraining |
| [diffusion_eps.py](diffusion_eps.py) | `diffusion_eps.yaml` | Diffusion ε-prediction |
| [jepa.py](jepa.py) | `jepa.yaml` | JEPA joint-embedding predictive arch |
| [pcn_demo.py](pcn_demo.py) | `pcn_demo.yaml` | Predictive Coding Network |
| [ff_demo.py](ff_demo.py) | `ff_demo.yaml` | Forward-Forward algorithm |
| [mezo_sft.py](mezo_sft.py) | `mezo_sft.yaml` | MeZO memory-efficient zeroth-order |
| [sweep_demo.py](sweep_demo.py) | `sweep_demo.yaml` + `sweep_r15.yaml` | Hyperparameter sweep |
| [fork_resume.py](fork_resume.py) | `fork_resume.yaml` | Fork + resume with lineage |
