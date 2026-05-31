# 扩展 lighttrain

> [English](extending.md) · [文档索引](../README.md)

任何满足 [lighttrain/protocols.py](../../lighttrain/protocols.py) 中某 Protocol 的类
都能注册并使用，无需改核心。把 recipe 的 `user_modules:` 指向你的文件，启动时即运行
`@register` 装饰器。

```yaml
user_modules:
  - my_components.py
```

完整类别清单、协议签名与内置项见 [reference/registry.zh-CN.md](../reference/registry.zh-CN.md)。
下面是速查模式。

## 自定义 loss

```python
from lighttrain import register

@register("loss", "my_loss")
class MyLoss:
    def __call__(self, model_output, batch, ctx) -> dict:
        logits = model_output.outputs["logits"]
        ...
        return {"loss": loss}   # 必须含标量 "loss" 张量
```

## 自定义优化器

wrapper 必须提供 `build`、`step`、`zero_grad`，**以及 `state_dict` /
`load_state_dict`**（checkpoint 管理器在 *wrapper* 上调后两者）。继承 `_BaseWrapper`
可免费获得这四个 + 默认 `optim_state_bytes`，只需写 `build()`：

```python
import torch
from lighttrain import register
from lighttrain.optim.wrappers import _BaseWrapper, _split_param_groups

@register("optimizer", "my_adamw")
class MyAdamW(_BaseWrapper):
    def build(self, model):
        self._check_unbuilt()
        self.optimizer = torch.optim.AdamW(_split_param_groups(model, self.param_groups, self._kwargs))
        self._built = True
        return self.optimizer
```

> 要序列化自定义优化器状态？`torch` 的 `state_dict()` 与活动 `optimizer.state` 同引用，
> 原地改写会污染正在运行的优化器。用 `self._safe_state_dict(convert)`（先 copy）。覆写
> `optim_state_bytes`，让 `lighttrain estimate` 看见非 Adam 的状态占用。

## 自定义 callback

```python
from lighttrain import register, Signal

@register("callback", "my_cb")
class MyCB:
    def on_loss_computed(self, *, loss, **_):
        if not loss.isfinite():
            return Signal.SKIP_STEP
```

只实现需要的 hook —— `getattr` 分发其余（`CALLBACK_EVENTS` 共 39 个事件）。

## 自定义 trainer（新范式）

重写扁平 `Trainer` 的两个缝 —— `produce_batch`（batch 是什么）与 `forward_loss`
（前向 + loss）—— 或写一个调公共原语（`run_train_loop`、`apply_update`、
`forward_with_activations`）的短 `fit()`。多模型范式经 `self.models["..."]` /
`self.optimizers["..."]` 取用 —— **在 `__init__` 上声明 `models=` / `optimizers=`**
才能收到这套（见 [训练 § 多模型](../concepts/training.zh-CN.md#多模型)）。完整范例：
[examples/online_distill.py](../../examples/online_distill.py)。

## 自定义 PrepGraph 节点

```python
from lighttrain.prepgraph.node import PrepNode, NodeResult, RunContext
from lighttrain.registry import register

@register("prep_node", "my_kind")
class MyNode(PrepNode):
    kind = "my_kind"
    schema_kind = "rows"
    def run(self, ctx: RunContext) -> NodeResult:
        return NodeResult(fingerprint="", schema_kind=self.schema_kind, rows=..., store=..., extras={"row_count": ...})
```

`config` 必须跨进程产生相同 fingerprint —— `__init__` 里避免可变全局状态（如
`time.time()`）。

## 相关

- [reference/registry.zh-CN.md](../reference/registry.zh-CN.md) —— 所有类别与协议
- [架构](../concepts/architecture.zh-CN.md) —— 你要接入的那些缝
- [其他架构](architectures.zh-CN.md) —— 模型适配规则
