# 常见问题

> [English](troubleshooting.md) · [文档索引](../README.md)

## 常见报错

| 现象 | 可能原因 / 修法 |
| ---- | --------------- |
| `recipe is missing 'model:'/'data:'/'optim:' section` | 补上必填节点 —— 见 [配置](../guide/configuration.zh-CN.md)。 |
| 自定义优化器在首次 `ckpt_every` 报 `AttributeError` | wrapper 缺 `state_dict` / `load_state_dict`；继承 `_BaseWrapper` —— 见 [扩展](extending.zh-CN.md)。 |
| 自定义 trainer 收不到 teacher / 第二个模型 | `__init__` 必须声明 `models=` / `optimizers=` 才能收到这套 —— 见 [训练 § 多模型](../concepts/training.zh-CN.md#多模型)。 |
| 模型上出现 `UserWarning: dropped recipe key …` | 跨架构杂键；有显式签名时是预期行为 —— 见 [其他架构](architectures.zh-CN.md)。 |
| epoch 中途 resume 没过 `resume-verify` | bit-exact 校验用 fp32 + 单 worker；否则传 `--tol`。 |
| resume 时某 checkpoint 目录被忽略 | 缺 `manifest.json`（最后写）→ 视为不完整。 |
| loss 是 NaN | 看 `diagnostics/` 下 `nan_hunter` 输出 + `repro.py`；见 [诊断](../operations/diagnostics.zh-CN.md)。 |

## 第三方已知限制

这些在上游包里，不在 lighttrain（复现 Mamba-3 时遇到）：

- **`state-spaces/mamba`** —— `mixer_seq_simple.create_block` 只白名单
  `Mamba1`/`Mamba2`。模块级适配模式（[其他架构](architectures.zh-CN.md)）可绕过：
  自己实例化 `Mamba3`，永不碰 `create_block`。
- **`state-spaces/mamba`** —— Mamba-2 快路径（`use_mem_eff_path=True`）需
  `causal_conv1d` CUDA 扩展。设 `use_mem_eff_path=False` 走纯 Triton 回退
  （开箱即用，约慢 3×）。
- **`tilelang==0.1.8` + `apache-tvm-ffi==0.1.11`** —— MIMO chunk-bwd lowering 时
  在 `NestedLoopChecker` 崩溃。GatedDeltaNet 设 `FLA_TILELANG=0`（Triton 后端）。
  该环境下 Mamba-3 MIMO 无回退 —— 用 SISO 变体。

## eager import 失败

若某研究包因 `__init__` 拖入重型兄弟模块（CUDA C 扩展、可选纯 Python 依赖）而 import
失败，在 import 前预置 stub 模块 —— 见 [其他架构](architectures.zh-CN.md) 的 eager
import 提示。

## 相关

- [诊断](../operations/diagnostics.zh-CN.md) —— failure-first 工具
- [扩展](extending.zh-CN.md) —— 注册契约
