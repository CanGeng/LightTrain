# Known Issues — 未解决问题集中账本

> 本文件是 lighttrain **唯一的开放问题账本**。所有尚未解决的已知问题集中登记于此。
>
> **工作流**：一个问题被解决后 → 完整写进对应版本的 changelog（`v0/vX.Y/vX.Y.Z.md`），此处的条目改为**一句话 tombstone**，指明移入了哪个 changelog。永久放弃 / 核定为非缺陷的条目同样保留一句话说明，避免日后重复发掘。
>
> ID 沿用历次 changelog 审计编号（A=RL 正确性、B=类型债、C=持久化、D=可观测、E=结构、F=文档）。

## 开放（Open）

### A1 — PPO 未接入 reference-KL（`_ref_policy` 冻结后从未消费）
[ppo.py:195](../../lighttrain/builtin_plugins/trainers/ppo.py#L195) 在 `fit()` 里 `freeze_as_ref()` 建了参考策略，但全文件无人读它；[PPOSurrogateLoss](../../lighttrain/builtin_plugins/losses/rl.py#L57) 无 `beta_kl` 参数、loss 里无 KL 项（仅算 approx_kl 供监控）。→ PPO 实际不施加 ref-KL，冻结的 ref 是死重。GRPO 侧已有完整 k3 KL 模板可照搬。
**状态：开放**（登记于 v0.3.1「新登记（转后续）」）

### A2 — GRPO `lora_base_as_ref=True` + KL 未接线（fail-loud 占位）
[grpo.py:188](../../lighttrain/builtin_plugins/trainers/grpo.py#L188) 与 [ref_policy.py:81-87](../../lighttrain/builtin_plugins/rl/ref_policy.py#L81) 对 `per_token=True` + `lora_base_as_ref=True` 直接 `raise`。adapter toggling 已实现（[`_lora_base_log_probs`](../../lighttrain/builtin_plugins/rl/ref_policy.py#L123) 的 disable/enable + try/finally），但只支持序列级，缺 per-token 分支，故 LoRA-base 作 ref 的 KL 路径不可用。
**状态：开放**（登记于 v0.3.1「新登记（转后续）」）

### B1 — mypy `ignore_errors` 隔离区未清空
[pyproject.toml](../../pyproject.toml) 的 `[[tool.mypy.overrides]] ignore_errors=true` 仍隔离一批携带历史类型债的模块，目标 ratchet 到空。其中 architectures（diffusion_unet / jepa / mamba / rwkv）与 distributed（ddp / fsdp / zero）为 **torch-stub 相关**：本机 nightly-torch 与 CI 的 CPU-torch 类型推导分叉，**本机绿 ≠ CI 绿**，不可凭本机删 ignore（见 v0.3.1 告诫）；`_preference_base` / `grpo` / `ppo` / `joined_dataset` / `producer` 为中等难度逻辑/协议类型债。
**状态：开放**（torch-stub 批与中难批留 CI-verified 后续 PR）

### B2 — `check_untyped_defs` 未启用
`[tool.mypy]` 未开 `check_untyped_defs`；开启后约 +89 错（tests/ 未注解函数体为主），另有 ~108 个 `annotation-unchecked` note。作为未来 types ratchet，成本约 5-7 个分批 PR。
**状态：开放**（显式 defer）

### C1 — Checkpoint 同 step 覆写非 crash-atomic
[manager.py save()](../../lighttrain/engine/checkpoint/manager.py#L102) 原地覆写固定文件名（`step_<n>/` 下各文件），同 step 覆写中途崩溃会留混合文件集、毁掉上一份本来完好的 checkpoint。
**状态：开放**（v0.2.6 决策点）

### D1 — hot-loop 日志可能刷屏
少数 warning 位于逐 step / 逐 layer / 逐 metric 循环内（[standard.py](../../lighttrain/builtin_plugins/engine/update_rules/standard.py) RETRY_STEP、[file_signals.py](../../lighttrain/builtin_plugins/callbacks/realtime_control/file_signals.py) 轮询、[lineage_recorder.py](../../lighttrain/builtin_plugins/callbacks/builtins/lineage_recorder.py) metric 转换），可复用 [`_warn_once`](../../lighttrain/builtin_plugins/callbacks/builtins/frozen_step.py#L42) 模式去重。
**状态：开放**（v0.3.2 登记）

### F1 — test_sam.py module docstring 陈旧
[test_sam.py:15](../../tests/engine/update_rules/test_sam.py#L15) 仍写「SAM does NOT honor SKIP_STEP (pinned via xfail)」，但 [sam.py](../../lighttrain/builtin_plugins/engine/update_rules/sam.py) 早已 honor（检测 SKIP_STEP→清零梯度→早返回）、对应测试 `test_sam_honors_skip_step_signal_from_on_loss_computed` 通过、全 suite 无 active xfail。
**状态：开放**（顺手清）

## 已勾销 / 非缺陷（Dismissed）

### A3 — GRPO rollout 每 ppo_epoch 重算 → 非缺陷
经核：buffer 已正确「每 outer step 仅 rollout 一次、内层 ppo_epochs 复用同 buffer」（标准 on-policy 模式）。**非缺陷，不予修复。**

### E1 — core+plugin 模型统一迁到 models/architectures → 永久放弃
破坏性核心 import/recipe 路径大重构、零功能收益。**用户决策：永久放弃，不再追踪。**

### E2 — eval/metrics 插件侧空脚手架 → 非缺陷
核心侧 [eval/metrics](../../lighttrain/eval/metrics/__init__.py) 函数齐全；插件侧 [builtin_plugins/eval/metrics](../../lighttrain/builtin_plugins/eval/metrics/) 是有意预留的 `@register("metric")` 落点（category 已注册）。**设计如此，非缺陷。**
