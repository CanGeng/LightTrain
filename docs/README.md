# lighttrain documentation

> English index below · [中文版见底部](#中文文档索引)

The docs are split by topic so you can jump straight to what you need. Each page
has an English version (`<topic>.md`) and a Chinese version (`<topic>.zh-CN.md`).

## Start here

| Topic | What's inside |
| ----- | ------------- |
| [Getting started](guide/getting-started.md) | Install, the I/O contract (what to prepare / what you get), first training run |
| [CLI reference](guide/cli.md) | Every `lighttrain` command, flags, override syntax |
| [Configuration](guide/configuration.md) | YAML schema, `model_profiles`, `defaults:` composition, interpolation, override precedence |

## Core concepts

| Topic | What's inside |
| ----- | ------------- |
| [Architecture](concepts/architecture.md) | The five seams (Registry / Config / Engine+UpdateRule / EventBus / PrepGraph) and the init order |
| [Training paradigms](concepts/training.md) | Pretraining, SFT, preference (DPO/IPO/…), online RL (PPO/GRPO), distillation, multi-model |
| [Data & PrepGraph](concepts/data-prepgraph.md) | Datasets, collators, samplers, the content-addressed prep DAG |

## Scaling & operations

| Topic | What's inside |
| ----- | ------------- |
| [Distributed training](operations/distributed.md) | The `parallel:` block: DDP / FSDP / ZeRO / TP / PP (SP / EP registered but not yet wired) |
| [Diagnostics](operations/diagnostics.md) | Failure-first: invariants, NaN hunter, frozen steps, crash bundles, `doctor` |

## Extending

| Topic | What's inside |
| ----- | ------------- |
| [Alternative architectures](extending/architectures.md) | RWKV / Mamba / JEPA / diffusion + the model-adapter rules |
| [Extending lighttrain](extending/extending.md) | Register your own model / loss / optimizer / callback / trainer / prep node |
| [Recipe index](extending/recipes.md) | One line per bundled recipe — the fastest way to find a starting point |
| [Troubleshooting](extending/troubleshooting.md) | Common errors and known third-party limitations |

## Reference

| Topic | What's inside |
| ----- | ------------- |
| [Registry & protocols](reference/registry.md) | All registered categories, protocol signatures, built-in entries (lookup table) |
| [Changelog](changelog/) | Per-version notes |

---

## 中文文档索引

文档按主题拆分，便于快速定位。每篇都有英文版（`<topic>.md`）与中文版（`<topic>.zh-CN.md`）。

### 从这里开始

| 主题 | 内容 |
| ---- | ---- |
| [快速开始](guide/getting-started.zh-CN.md) | 安装、I/O 约定（要准备什么 / 产出什么）、第一次训练 |
| [命令行参考](guide/cli.zh-CN.md) | 每个 `lighttrain` 命令、参数、override 语法 |
| [配置](guide/configuration.zh-CN.md) | YAML schema、`model_profiles`、`defaults:` 组合、插值、override 优先级 |

### 核心概念

| 主题 | 内容 |
| ---- | ---- |
| [架构](concepts/architecture.zh-CN.md) | 五个缝（Registry / Config / Engine+UpdateRule / EventBus / PrepGraph）与初始化顺序 |
| [训练范式](concepts/training.zh-CN.md) | 预训练、SFT、偏好（DPO/IPO/…）、在线 RL（PPO/GRPO）、蒸馏、多模型 |
| [数据与 PrepGraph](concepts/data-prepgraph.zh-CN.md) | 数据集、collator、sampler、内容寻址的预处理 DAG |

### 扩展与运维

| 主题 | 内容 |
| ---- | ---- |
| [分布式训练](operations/distributed.zh-CN.md) | `parallel:` 块：DDP / FSDP / ZeRO / TP / PP（SP / EP 已注册但尚未接入） |
| [诊断](operations/diagnostics.zh-CN.md) | failure-first：invariant、NaN 溯源、冻结步、崩溃现场、`doctor` |

### 扩展开发

| 主题 | 内容 |
| ---- | ---- |
| [其他架构](extending/architectures.zh-CN.md) | RWKV / Mamba / JEPA / diffusion + 模型适配规则 |
| [扩展 lighttrain](extending/extending.zh-CN.md) | 注册自定义 model / loss / optimizer / callback / trainer / prep 节点 |
| [配方索引](extending/recipes.zh-CN.md) | 每个内置 recipe 一行说明——最快找到起点 |
| [常见问题](extending/troubleshooting.zh-CN.md) | 常见报错与第三方已知限制 |

### 参考

| 主题 | 内容 |
| ---- | ---- |
| [注册表与协议](reference/registry.zh-CN.md) | 所有注册类别、协议签名、内置项（查阅表） |
| [更新日志](changelog/) | 各版本说明 |
