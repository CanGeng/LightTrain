# Distributed training

> [中文版](distributed.zh-CN.md) · [Docs index](../README.md)

> **Status:** Only **data parallelism** is supported — DDP, FSDP, and DeepSpeed
> ZeRO via the `grad_sync` strategy. DDP and FSDP have been validated on a real
> single-node multi-GPU box (NCCL); DeepSpeed ZeRO and multi-node have not.
> Tensor / pipeline / expert / sequence parallelism (TP / PP / EP / SP) were
> **removed**.
>
> **Fail mode:** a `grad_sync` strategy that is requested but cannot be built
> (unregistered name, missing optional dependency such as `deepspeed`) raises a
> `ConfigError` rather than silently falling back to single-GPU.

The `parallel:` block scales a run from single-GPU to multi-GPU **without
changing model or trainer code**. Omitting it is equivalent to `dp=1`.

## `parallel:` fields

| Field | Default | Notes |
| ----- | ------- | ----- |
| `backend` | `nccl` | `gloo` for CPU / CI |
| `dp` | 1 | data-parallel replicas (must equal the total GPU count) |
| `force_cpu` | false | all tensors on CPU; pair with `gloo` for GPU-free comm tests |
| `grad_sync` | `{name: noop}` | gradient-sync strategy (below) |

### `grad_sync` strategies

| name | implementation |
| ---- | -------------- |
| `noop` | single-GPU passthrough (default) |
| `ddp` | `DistributedDataParallel` (extra: `find_unused_parameters`) |
| `fsdp` | `FullyShardedDataParallel` (extra: `sharding_strategy`, `state_dict_type`) |
| `deepspeed` | DeepSpeed ZeRO-1/2/3 (requires the `deepspeed` package) |

## Launching

```bash
torchrun --nproc_per_node=N -m lighttrain.cli train -c cfg.yaml
# multi-node: add --nnodes --node_rank --master_addr --master_port
```

## Examples

```yaml
# single-node DDP (4 GPUs)
parallel: { backend: nccl, dp: 4, grad_sync: { name: ddp, find_unused_parameters: false } }
```

```yaml
# FSDP + grad accumulation
parallel: { backend: nccl, dp: 8, grad_sync: { name: fsdp, sharding_strategy: FULL_SHARD, state_dict_type: full } }
trainer:  { accumulate: 4 }
```

```yaml
# gloo + CPU comm test (no GPU)
parallel: { backend: gloo, dp: 4, force_cpu: true, grad_sync: { name: ddp } }
engine:   { mixed_precision: "no" }
```

Full recipe examples live under
[`recipes/`](../../recipes).

## See also

- [reference/registry.md](../reference/registry.md) — distributed strategy entries
