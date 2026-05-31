# 其他架构

> [English](architectures.md) · [文档索引](../README.md)

有状态（RWKV、Mamba）与非 Transformer 目标以插件形式提供，像普通 model/objective
一样选用：

```bash
lighttrain train -c recipes/pretrain_rwkv.yaml   # RWKV 有状态预训练
lighttrain train -c recipes/diffusion_eps.yaml   # diffusion eps 预测
lighttrain train -c recipes/jepa.yaml            # JEPA 掩码 patch 预测
lighttrain train -c recipes/pcn_demo.yaml        # 预测编码网络
lighttrain train -c recipes/ff_demo.yaml         # Forward-Forward
lighttrain train -c recipes/mezo_sft.yaml        # MeZO 零阶 SFT
```

内置 objective 带 `loss_family`：`next_token`、`masked_denoising`（`mlm`）、
`diffusion`、`flow_matching`、`jepa`。其他 update rule：`mezo`、`sam`、
`forward_forward`、`pcn`、`dfa`（设 `update_rule.name`）。

## 写模型适配器（两条规则）

包装第三方架构（SSM、FLA 等）时，两条都重要。

**规则 1 —— import 最底层模块，而非高层工厂。**

```python
from mamba_ssm.modules.mamba3 import Mamba3   # 好 —— 能挺过上游重构
# 别用：from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
#       MambaLMHeadModel(..., ssm_layer="Mamba3")   # 可能带隐藏白名单
```

高层工厂常带内部白名单、版本假设或隐式 C 扩展依赖。模块级组装让你掌控全局。

**规则 2 —— 在注册类上声明显式签名**（不要 `def __init__(self, **kwargs)`）：

```python
@register("model", "mamba2_lm")
class Mamba2LM(_MambaLMAdapter):
    def __init__(self, *, d_model: int, n_layer: int, vocab_size: int,
                 d_state: int = 128) -> None:        # 显式、具名
        super().__init__(layer="Mamba2", d_model=d_model, n_layer=n_layer, ...)
```

resolver（`_filter_kwargs`）**按注册类签名**丢弃未知 recipe kwarg。`**kwargs` 类语义上
声称「我要所有键」，于是过滤变成 no-op，跨架构的杂键会泄漏进来。有显式签名时，杂键
会带 `UserWarning` 被丢弃。

**逃生舱** —— 若确实需要 `**kwargs` 做内层转发，在类上设
`__lighttrain_filtered_kwargs__ = True`；resolver 会按你的显式参数过滤，仍挡住
recipe 侧泄漏。

**eager import 提示** —— 许多研究仓库在顶层 `__init__` 里拖入重型兄弟模块。在 import
前从 `user_modules` 文件预置 stub：

```python
import sys, types
sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
import mamba_ssm   # 现在安全
# 或：先 stub 父包，再直接 import 你需要的子模块，绕过父包 __init__
```

具体的 mamba/tilelang 案例见 [常见问题](troubleshooting.zh-CN.md)。

## 相关

- [扩展](extending.zh-CN.md) —— 完整注册契约
- [常见问题](troubleshooting.zh-CN.md) —— 第三方已知限制
- [reference/registry.zh-CN.md](../reference/registry.zh-CN.md) —— model / objective 注册项
