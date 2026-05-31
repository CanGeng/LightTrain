# 配方索引

> [English](recipes.md) · [文档索引](../README.md)

最快的起步方式：从 [`recipes/`](../../recipes/) 拷一个内置 recipe 改。任意一个都用
`lighttrain train -c recipes/<名字>.yaml` 运行。

## 预训练与 SFT

| Recipe | 演示 |
| ------ | ---- |
| `pretrain_causal` | Causal-LM 预训练（tiny_lm + byte 分词器）—— 标准起点 |
| `pretrain_rwkv` | RWKV 有状态预训练 |
| `sft_chat` | 经 PrepGraph 的 chat 数据 SFT |
| `sft_chat_hf` | 用 HuggingFace 模型/分词器的 SFT |
| `vlm_sft` | 视觉-语言 SFT（多模态 collator） |

## 偏好与 RL

| Recipe | 演示 |
| ------ | ---- |
| `dpo_offline` | 离线 DPO（`preference` 下的 `loss:` 缝） |
| `ppo_online` | 在线 PPO，rollout + GAE + verifier judge |
| `grpo` | Group Relative Policy Optimization |
| `produce_teacher` | 产出参考/teacher artifact |

## 蒸馏

| Recipe | 演示 |
| ------ | ---- |
| `student_kd` | teacher → student 知识蒸馏 |
| `online_distill_demo` | 双模型在线蒸馏（student 对抗冻结 teacher rollout）—— 多模型缝，见 [examples/online_distill.py](../../examples/online_distill.py) |

## 其他目标与 update rule

| Recipe | 演示 |
| ------ | ---- |
| `diffusion_eps` | diffusion eps 预测目标 |
| `jepa` | JEPA 掩码 patch 预测 |
| `pcn_demo` | 预测编码网络 |
| `ff_demo` | Forward-Forward |
| `mezo_sft` | MeZO 零阶 SFT（省显存） |

## 显存效率

| Recipe | 演示 |
| ------ | ---- |
| `qlora` | QLoRA 4-bit 微调（Linux + CUDA） |
| `offload_fullparam` | LayerOffload + CPU offload 优化器 |

## 实验生命周期

| Recipe | 演示 |
| ------ | ---- |
| `fork_resume` | 从 checkpoint fork + resume |
| `sweep_lr` | 学习率扫描 |
| `sweep_demo` | 通用扫描配置 |
| `sweep_r15` | 带早停规则的扫描 |

## 相关

- [快速开始](../guide/getting-started.zh-CN.md) —— 跑你的第一个 recipe
- [训练范式](../concepts/training.zh-CN.md) —— 各 recipe 形态的含义
- [配置](../guide/configuration.zh-CN.md) —— 放心地改 recipe 字段
