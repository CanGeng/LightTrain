# 分布式训练

> [English](distributed.md) · [文档索引](../README.md)

> **状态：** 仅支持**数据并行**——DDP、FSDP、以及通过 `grad_sync` 的 DeepSpeed
> ZeRO。DDP、FSDP、DeepSpeed ZeRO-2 已在真实单机多卡（NCCL）上验证；多机尚未验证。
> 张量 / 流水线 / 专家 / 序列并行（TP / PP / EP / SP）已**移除**。
>
> **失败模式：** 请求了某个 `grad_sync` 策略但无法构建（名称未注册、缺可选依赖如
> `deepspeed`）会 **fail loud 抛 `ConfigError`**，不会静默回落到单卡。

`parallel:` 块让一次 run 从单卡扩到多卡，**无需改动模型或 trainer 代码**。
不写它等同于 `dp=1`。

## `parallel:` 字段

| 字段 | 默认 | 说明 |
| ---- | ---- | ---- |
| `backend` | `nccl` | CPU / CI 用 `gloo` |
| `dp` | 1 | 数据并行副本数（须等于总 GPU 数） |
| `force_cpu` | false | 所有张量在 CPU；配 `gloo` 做无 GPU 通信测试 |
| `grad_sync` | `{name: noop}` | 梯度同步策略（见下） |

### `grad_sync` 策略

| name | 实现 |
| ---- | ---- |
| `noop` | 单卡直通（默认） |
| `ddp` | `DistributedDataParallel`（额外：`find_unused_parameters`） |
| `fsdp` | `FullyShardedDataParallel`（额外：`sharding_strategy`、`state_dict_type`） |
| `deepspeed` | DeepSpeed ZeRO-1/2/3（需安装 `deepspeed`） |

## 启动

```bash
torchrun --nproc_per_node=N -m lighttrain.cli train -c cfg.yaml
# 多机：加 --nnodes --node_rank --master_addr --master_port
```

## 示例

```yaml
# 单机 DDP（4 GPU）
parallel: { backend: nccl, dp: 4, grad_sync: { name: ddp, find_unused_parameters: false } }
```

```yaml
# FSDP + 梯度累积
parallel: { backend: nccl, dp: 8, grad_sync: { name: fsdp, sharding_strategy: FULL_SHARD, state_dict_type: full } }
trainer:  { accumulate: 4 }
```

```yaml
# gloo + CPU 通信测试（无 GPU）
parallel: { backend: gloo, dp: 4, force_cpu: true, grad_sync: { name: ddp } }
engine:   { mixed_precision: "no" }
```

完整 recipe 示例见
[`recipes/`](../../recipes)。

## 相关

- [reference/registry.zh-CN.md](../reference/registry.zh-CN.md) —— 分布式策略注册项
