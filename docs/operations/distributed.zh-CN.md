# 分布式训练

> [English](distributed.md) · [文档索引](../README.md)

> **状态：** DDP/FSDP/ZeRO/TP/PP/SP/EP 已实现，并通过基于 CPU 多进程（gloo）的
> spawn 测试。**尚未**在多机 GPU 集群上做规模验证。生产环境请自行评估风险。

`parallel:` 块让一次 run 从单卡扩到多卡，**无需改动模型或 trainer 代码**。
不写它等同于 `dp=tp=pp=ep=1`。

## `parallel:` 字段

| 字段 | 默认 | 说明 |
| ---- | ---- | ---- |
| `backend` | `nccl` | CPU / CI 用 `gloo` |
| `dp` | 1 | 数据并行副本数 |
| `tp` | 1 | 张量并行分片（TP×DP×PP = 总 GPU 数） |
| `pp` | 1 | 流水线阶段数 |
| `ep` | 1 | 专家并行大小；须整除 `dp` |
| `sp` | false | 序列并行（与 TP 配合） |
| `force_cpu` | false | 所有张量在 CPU；配 `gloo` 做无 GPU 通信测试 |
| `grad_sync` | `{name: noop}` | 梯度同步策略（见下） |
| `tensor_parallel` | null | TP 手术方案 |
| `pipeline` | null | PP 调度 |

### `grad_sync` 策略

| name | 实现 |
| ---- | ---- |
| `noop` | 单卡直通（默认） |
| `ddp` | `DistributedDataParallel`（额外：`find_unused_parameters`） |
| `fsdp` | `FullyShardedDataParallel`（额外：`sharding_strategy`、`state_dict_type`） |
| `deepspeed` | DeepSpeed ZeRO-1/2/3 |

### 张量 / 流水线块

```yaml
tensor_parallel:
  auto_plan_for: llama        # 内置：llama / gpt2 / mistral
  # 或手动：
  # plan:
  #   - { path: "model.layers.*.self_attn.q_proj", style: colwise }
  #   - { path: "model.layers.*.self_attn.o_proj", style: rowwise }

pipeline:
  n_microbatches: 8
  schedule: 1f1b              # 1f1b / gpipe / interleaved_1f1b
  stage_spec:
    - { layers: "model.embed_tokens,model.layers.0-15" }
    - { layers: "model.layers.16-31,model.norm,lm_head" }
```

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
[`plugins/distributed/recipes/`](../../plugins/distributed/recipes)。

## 相关

- [架构 § 初始化顺序](../concepts/architecture.zh-CN.md) —— 为何 TP/SP/EP 先于 FSDP/DDP
- [reference/registry.zh-CN.md](../reference/registry.zh-CN.md) —— 分布式策略注册项
