# Distributed training

> [中文版](distributed.zh-CN.md) · [Docs index](../README.md)

> **Status:** DDP/FSDP/ZeRO/TP/PP strategies are implemented and unit-tested via
> CPU-based multiprocess (gloo) spawn tests. They have **not** been validated at
> scale on multi-node GPU clusters. Use at your own risk for production.
>
> **Known limitation (SP/EP):** the `sequence_parallel` / `expert_parallel`
> strategies are registered, but the train runtime's strategy selector only wires
> `tensor_parallel` — `parallel.sp` / `parallel.ep` do **not** yet select them,
> and EP is still a skeleton (no all-to-all). They are not usable from a recipe
> at present. See the v0.2.3 changelog "Known issues".

The `parallel:` block scales a run from single-GPU to multi-GPU **without
changing model or trainer code**. Omitting it is equivalent to `dp=tp=pp=ep=1`.

## `parallel:` fields

| Field | Default | Notes |
| ----- | ------- | ----- |
| `backend` | `nccl` | `gloo` for CPU / CI |
| `dp` | 1 | data-parallel replicas |
| `tp` | 1 | tensor-parallel shards (TP×DP×PP = total GPUs) |
| `pp` | 1 | pipeline stages |
| `ep` | 1 | expert-parallel size; must divide `dp` |
| `sp` | false | sequence parallelism (pairs with TP) |
| `force_cpu` | false | all tensors on CPU; pair with `gloo` for GPU-free comm tests |
| `grad_sync` | `{name: noop}` | gradient-sync strategy (below) |
| `tensor_parallel` | null | TP surgery plan |
| `pipeline` | null | PP schedule |

### `grad_sync` strategies

| name | implementation |
| ---- | -------------- |
| `noop` | single-GPU passthrough (default) |
| `ddp` | `DistributedDataParallel` (extra: `find_unused_parameters`) |
| `fsdp` | `FullyShardedDataParallel` (extra: `sharding_strategy`, `state_dict_type`) |
| `deepspeed` | DeepSpeed ZeRO-1/2/3 |

### Tensor / pipeline blocks

```yaml
tensor_parallel:
  auto_plan_for: llama        # built-in: llama / gpt2 / mistral
  # or manual:
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

- [Architecture § init order](../concepts/architecture.md) — why TP/SP/EP precede FSDP/DDP
- [reference/registry.md](../reference/registry.md) — distributed strategy entries
