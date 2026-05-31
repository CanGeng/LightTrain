# 配置

> [English](configuration.md) · [文档索引](../README.md)

recipe 是 OmegaConf YAML，经 Pydantic v2 schema（`RootConfig`）校验。
加载顺序：**读 YAML → 合并 `defaults:` → 注入 CLI override → `${}` 插值 → Pydantic 校验**。

## 必填节点

缺任一立即报错：

| 节点 | 报错 |
| ---- | ---- |
| `model:` | `recipe is missing 'model:' section` |
| `data:` | `recipe is missing 'data:' section` |
| `optim:` | `recipe is missing 'optim:' section` |

## 根字段

| 字段 | 默认 | 说明 |
| ---- | ---- | ---- |
| `mode` | `lab` | `lab` 自动挂诊断；`prod` 精简 |
| `seed` | `42` | torch / numpy / python |
| `exp` | `default` | 决定 run 目录命名 |
| `run_root` | `runs` | 输出根目录 |
| `run_dir` | 无 | 直接指定完整输出路径 |
| `user_modules` | `[]` | 额外导入的 Python 文件（让你的组件执行 `@register`） |
| `models:` / `optimizers:` | 无 | 命名模型/优化器集——见 [训练 § 多模型](../concepts/training.zh-CN.md#多模型) |

## 指定模型（`model_profiles`）

声明一个或多个完整模型配置，按名字选中：

```yaml
model: base                 # 选择器（CLI：model=big）
model_profiles:
  base: { name: tiny_lm, d_model: 256, n_layers: 4, n_heads: 8 }
  big:  { name: tiny_lm, d_model: 512, n_layers: 8, n_heads: 8 }
```

```bash
lighttrain train -c r.yaml model=big                      # 切 profile
lighttrain train -c r.yaml model_profiles.base.d_model=384  # 改字段
```

profile 选择与模型**集**（`models:`）正交；`models:` 条目的 `spec` 可写
`{ profile: <名字> }`。（裸 `model:` 字典块已在 v0.1.8 移除——务必声明 profile；
旧 recipe 用 `lighttrain migrate config <recipe> --to-profiles` 转换。）

## 组件引用语法

注册表短名（**A**）或直接 import（**B**）：

```yaml
optim: { name: adamw, lr: 3e-4 }                 # A —— 注册表
optim: { _target_: torch.optim.AdamW, lr: 3e-4 } # B —— import
```

## `defaults:` 组合与插值

```yaml
defaults:
  - base/model_tiny        # 相对本文件的路径，不带 .yaml
  - ../shared/adamw_optim
trainer: { max_steps: 3000 }   # 覆盖被合并文件里的值
```

```yaml
seed: 1337
scheduler: { total_steps: ${trainer.max_steps} }   # 同配置引用
data: { dataset: { seed: ${seed} } }
optim: { lr: ${oc.env:LR,3e-4} }                    # 环境变量带默认
```

## 默认 fallback（只给 model+data+optim）

| 组件 | 自动 fallback |
| ---- | ------------- |
| `loss` | `cross_entropy`（shift + CE） |
| `trainer` | `pretrain`，max_steps=1000，每 500 步 ckpt |
| `engine` | `standard`，bf16 |
| `data.name` | `simple` |
| `tokenizer` | `byte`（vocab 260） |
| `collator` | `causal_lm`（右填充） |
| `scheduler` | 无（恒定 lr） |
| `callbacks` / `logger` | `[]`（lab 模式仍挂诊断） |
| device | 有 CUDA 用 GPU，否则 CPU |

`lab` 模式自动挂载的诊断 callback：`invariants`、`frozen_step`（每 1000 步）、
`file_signals`，外加一个 callback 隔离 sink。

## 字段表（最常用）

`trainer:` —— `name`（`pretrain`/`preference`/`reward_model`/`ppo`/`grpo` + 自定义）、
`max_steps`、`val_every`、`ckpt_every`、`log_every`、`grad_clip`、`accumulate`。
RL 额外有 `rollout_backend`/`temperature`/`top_p`/`do_sample`/`buffer_max_size`；
`trainer:` 会转发任意额外 kwarg（按 trainer 签名过滤）。

`engine:` —— `name`、`mixed_precision`（`no`/`fp16`/`bf16`）、`update_rule.name`
（`standard`/`sam`/`mezo`/`rl`）。

`data:` —— `name`、`batch_size`、`num_workers`、`pin_memory`、`drop_last`，及
`dataset` / `tokenizer` / `collator` / `sampler` 子块。

完整字段与协议表见 [注册表与协议](../reference/registry.zh-CN.md)。

## 相关

- [架构](../concepts/architecture.zh-CN.md) —— 这些节点如何变成一次 run（初始化顺序）
- [训练范式](../concepts/training.zh-CN.md) —— 各范式的 recipe 形态
- [分布式](../operations/distributed.zh-CN.md) —— `parallel:` 块
