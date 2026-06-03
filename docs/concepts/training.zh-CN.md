# 训练范式

> [English](training.md) · [文档索引](../README.md)

一个扁平 `Trainer` + 缝重写，覆盖预训练、SFT、偏好、在线 RL、蒸馏、多模型。
算法通常是 `loss:` 缝，而非单独的 trainer。

## 预训练与 SFT

`trainer: pretrain` 就是裸的扁平 `Trainer`——纯 YAML，无子类。SFT 是同一 trainer
跑 chat/指令数据（常经 PrepGraph）。见 [配方](../extending/recipes.zh-CN.md)：`pretrain_causal`、
`sft_chat`、`sft_chat_hf`。

## 离线偏好（DPO / IPO / SimPO / ORPO / KTO）

一个 `preference` trainer；算法是 `loss:` 缝：

```yaml
trainer: { name: preference, ref_namespace: ref }
loss:    { name: dpo, beta: 0.1 }      # 或 ipo | simpo | orpo | kto
```

典型流程：`produce-artifact` 缓存参考 log-prob，再 `train`。`reward_model` 是单独的
trainer（Bradley-Terry），带可插拔 `value_head:`。配方：`dpo_offline`、`produce_teacher`。

## 在线 RL（PPO / GRPO）

RL loss 是 `loss:` 缝；rollout 后端从 `rl_backend` 注册表解析，带完整采样 knob：

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

reward 来自 `judge:`（如 `verifier`），经 `reward_adapter`（默认 pointwise）包成
`reward_fn`。配方：`ppo_online`、`grpo`。

## 蒸馏

teacher → student。先产出 teacher artifact，再用蒸馏 `loss:`（`kl_topk`、
`hidden_mse`、`hidden_cosine`、`attention_transfer`）训 student。配方：
`produce_teacher`、`student_kd`。

## 多模型

一次 run 可声明**命名模型集**。`trainable: false` 的条目是冻结、在设备上、可选
加载 checkpoint 的辅助模型（蒸馏 teacher、冻结 reference）；每个 `trainable: true`
条目有自己的优化器（GAN / actor-critic）。单 `model:` / `optim:` 是单条目集的糖。

```yaml
models:
  student: { spec: { profile: small }, trainable: true, optimizer: opt_main }
  teacher: { spec: { name: hf_causal, pretrained: gpt2 }, trainable: false,
             checkpoint: path/to/teacher.safetensors }
optimizers:
  opt_main: { name: adamw, lr: 1.0e-3 }
```

自定义 trainer 经 `self.models["teacher"]` / `self.optimizers["student"]` 取用。

> **容易踩的机制：** runtime 总是把 `models=` / `optimizers=` 传给 trainer，但
> resolver 只在 trainer 的 `__init__` **声明了这些参数**时才保留它们。所以自定义
> 多模型 trainer 必须声明 `models=` / `optimizers=` 才能收到这套——内置 `ppo`/`grpo`
> 没声明，这正是它们装不下第二个模型的原因。

可跑的端到端示例：student 在线 rollout 对抗冻结 teacher——
[examples/online_distill.py](../../examples/online_distill.py)
（`lighttrain train -c recipes/online_distill_demo.yaml`）。

## PEFT 与显存效率

每个适配器是一个 model profile（`lora` / `ia3` / `adalora` / `qlora`）；`LayerOffload`
engine + `cpu_offload` 优化器让大模型跑在消费级 GPU 上。配方：`qlora`、
`offload_fullparam`、`mezo_sft`。

## 相关

- [配置](../guide/configuration.zh-CN.md) —— recipe 字段参考
- [扩展](../extending/extending.zh-CN.md) —— 写自定义 trainer / loss
- [配方索引](../extending/recipes.zh-CN.md) —— 可跑起点
