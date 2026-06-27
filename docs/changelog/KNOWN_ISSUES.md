# Known Issues — 未解决问题集中账本

> 本文件是 lighttrain **唯一的开放问题账本**。所有尚未解决的已知问题集中登记于此。
>
> **工作流**：一个问题被解决后 → 完整写进对应版本的 changelog（`v0/vX.Y/vX.Y.Z.md`），此处的条目改为**一句话 tombstone**，指明移入了哪个 changelog。永久放弃 / 核定为非缺陷的条目同样保留一句话说明，避免日后重复发掘。
>
> ID 沿用历次 changelog 审计编号（A=RL 正确性、B=类型债、C=持久化、D=可观测、E=结构、F=文档）。

## 开放（Open）

### E3 — 无内置 `hf_auto` tokenizer；HF 分词器需由用户在 `user_modules` 中自注册

**发现版本**：v0.5.4（nanoGPT/MiniMind 移植审计）

lighttrain 内置 tokenizer 仅有 `byte`；任何需要 `AutoTokenizer.from_pretrained` 的 HF 模型（包括 MiniMind 自定义分词器、`hf_causal` 用户侧自定义词表等）必须在 `user_modules` 里手工注册一个 `@register("tokenizer", "...")` 封装。这是相当常见的需求，应内置一个 `hf_auto` tokenizer（接受 `path:` 参数）。

**当前绕过**：`examples/MiniMind/model/model_adapter.py` 中注册了 `hf_auto` tokenizer，可复用于其他 HF 模型场景。

## 已解决 / 已勾销（Resolved / Dismissed）

### B2 — `check_untyped_defs` 未启用 / tests/ 余量未清
✅ 已解决 → v0.5.1（生产）+ v0.5.2（tests/）（见 [v0.5.2](v0/v0.5/v0.5.2.md)）：v0.5.1 启用 `check_untyped_defs=true` 覆盖 `lighttrain/`;v0.5.2 注解优先清空 `tests/` 的 358 个未注解 body 错并删除 `tests.*` opt-out,check_untyped_defs 现覆盖整个仓库。

### B1 — mypy `ignore_errors` 隔离区未清空
✅ 已解决 → v0.5.1（见 [v0.5.1](v0/v0.5/v0.5.1.md)）：建 CPU-torch parity venv 复现 CI 视角后，torch-stub 批 + 中难批一次真修到零，删除整个 `ignore_errors` 隔离块（12 → 0，env-invariant 双绿）。

### A1 — PPO 未接入 reference-KL
✅ 已解决 → v0.5.0（见 [v0.5.0](v0/v0.5/v0.5.0.md)）：PPOSurrogateLoss 加 `beta_kl` + k3 KL 项，PPOTrainer 仅 beta_kl>0 时建 ref 并注入 per-token `log_probs_ref`。

### A2 — GRPO/PPO `lora_base_as_ref=True` + KL 未接线
✅ 已解决 → v0.5.0（见 [v0.5.0](v0/v0.5/v0.5.0.md)）：ReferencePolicy 新增 `_lora_base_log_probs_per_token`，去掉拒绝守卫，trainer 注入时传 `live_model`。

### C1 — Checkpoint 同 step 覆写非 crash-atomic
✅ 已解决 → v0.5.0（见 [v0.5.0](v0/v0.5/v0.5.0.md)）：save() 改为 staging 目录 + 原子 swap，崩溃绝不毁掉上一份已提交 checkpoint。

### D1 — hot-loop 日志可能刷屏
✅ 已解决 → v0.5.0（见 [v0.5.0](v0/v0.5/v0.5.0.md)）：新增 `lighttrain/utils/log.py::warn_once`，套到 standard/file_signals/lineage_recorder 的逐 step/逐 metric 站点。

### F1 — test_sam.py module docstring 陈旧
✅ 已解决 → v0.5.0（见 [v0.5.0](v0/v0.5/v0.5.0.md)）：改为如实描述 SAM 自 v0.1.6 起 honor SKIP_STEP。

### A3 — GRPO rollout 每 ppo_epoch 重算 → 非缺陷
经核：buffer 已正确「每 outer step 仅 rollout 一次、内层 ppo_epochs 复用同 buffer」（标准 on-policy 模式）。**非缺陷，不予修复。**

### E1 — core+plugin 模型统一迁到 models/architectures → 永久放弃
破坏性核心 import/recipe 路径大重构、零功能收益。**用户决策：永久放弃，不再追踪。**

### E2 — eval/metrics 插件侧空脚手架 → 非缺陷
核心侧 [eval/metrics](../../lighttrain/eval/metrics/__init__.py) 函数齐全；插件侧 [builtin_plugins/eval/metrics](../../lighttrain/builtin_plugins/eval/metrics/) 是有意预留的 `@register("metric")` 落点（category 已注册）。**设计如此，非缺陷。**
