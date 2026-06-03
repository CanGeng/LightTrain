# 诊断（failure-first）

> [English](diagnostics.md) · [文档索引](../README.md)

failure-first 子系统让崩溃的 run 在人工介入前就回答「什么失败了？」「下一步怎么办？」。
在 `lab` 模式下，runtime 只自动挂载这几项：`invariants`、`frozen_step`、`file_signals`、
`CallbackIsolationSink`。其余诊断（`nan_hunter`、`loss_attribution`、`dead_neuron` 等）只是
注册到注册表，需在 `callbacks:` 中显式列出才会启用。

## 构件

- **Invariant** —— 每步检查，动作可选 `abort` / `skip` / `warn`：
  `loss_finite`、`grad_norm_bounded`、`lr_nonneg`、`label_mask_nonzero`、
  `param_count_stable`、`dtype_stable`、`batch_nonempty`。经 `invariants:` 配置。
- **NaN hunter** —— 模块前向 hook 定位 NaN 源头，写出自包含的 `repro.py`。
- **冻结步包** —— model + optimizer + batch 的单文件 ZIP 快照，可用 `replay-step`
  重放。lab 模式每 1000 步自动捕获。
- **Loss attribution** —— 逐样本、逐 token、逐模块的 loss 归因。
- **OOM 报告** —— 结构化报告，附建议的降级补丁。
- **实时控制** —— 经 `file_signals` callback 轮询 `<run_dir>/control/` 的在途干预
  （`lr.json`、`stop`、`eval_now`、`inject.py`）。
- **Callback 隔离** —— 非关键 callback 的异常被捕获并写入
  `diagnostics/callback_failures.jsonl`；连续失败多次后该 callback 会被 quarantine 跳过，
  坏 callback 不会拖垮整个 run。关键 callback（实例上 `critical = True`，或在 EventBus
  默认 critical 名单内）的异常会重新抛出并终止训练。

## 命令

```bash
lighttrain doctor      --run runs/exp/<...>          # 聚合诊断索引
lighttrain replay      --run runs/exp/<...>          # 重放最近的崩溃 / 冻结步
lighttrain freeze-step --run runs/exp/<...> --step N # 捕获单步包
lighttrain replay-step bundle.zip                    # 重放冻结步
```

## EventBus 信号

callback 可返回 `Signal` 来引导循环；结果按优先级聚合：
`STOP_TRAINING > RETRY_STEP > SKIP_STEP > CONTINUE`。示例：

```python
from lighttrain import register, Signal

@register("callback", "my_oom_guard")
class MyOOMGuard:
    def on_loss_computed(self, *, loss, **_):
        if not loss.isfinite():
            return Signal.SKIP_STEP
```

## 相关

- [架构 § EventBus](../concepts/architecture.zh-CN.md) —— 46 个生命周期事件
- [扩展](../extending/extending.zh-CN.md) —— 写自定义 callback / invariant
- [CLI](../guide/cli.zh-CN.md) —— `doctor` / `replay` / `freeze-step`
