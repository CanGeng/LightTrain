# 命令行参考

> [English](cli.md) · [文档索引](../README.md)

入口（来自 `pyproject.toml`）：`lighttrain = "lighttrain.cli._app:app"`。

全局选项：`--version`、`--quiet`、`--verbose`。

## 命令

| 命令 | 说明 |
| ---- | ---- |
| `init <path> [--force]` | 脚手架：带注释的 `cfg.yaml` + README + `runs/` + `artifacts/` |
| `dry-run -c <cfg> [--build]` | 解析并打印配置；`--build` 还会构造模型（校验 `model_profiles` 选择器） |
| `train -c <cfg> [OVERRIDES…] [--eval] [--output-summary f.json]` | 完整训练循环；当训练数据引用 PrepGraph 输出（`data.source: prep_graph:<terminal>` 或 `data.name: prep_graph`）时自动跑 PrepGraph |
| `resume --run <dir> [-c cfg] [--mode functional\|exact]` | 从 run 目录恢复（默认用其 `config.snapshot.yaml`） |
| `resume-verify -c <cfg> --phase1-steps N --phase2-steps M [--tol 1e-2]` | 逐步对比 loss，校验 resume == 单趟训练 |
| `overfit -c <cfg> --n N` | 在 N 个 batch 上过拟合（冒烟测试） |
| `inspect-data -c <cfg> [--n 32] [--decoded]` | 解码后的 batch 预览、长度直方图、label mask 覆盖率 |
| `prep -c <cfg> [--dry-run] [--workers N] [--pool thread\|process]` | 只跑 PrepGraph 数据流水线 |
| `prep-graph -c <cfg> [--out g.dot]` | 渲染 PrepGraph DAG（Mermaid / DOT） |
| `prep-status -c <cfg>` | 各 prep 节点缓存状态 |
| `prep-clean -c <cfg> [--orphans] [--dry-run]` | 清理缓存的 prep 产物 |
| `produce-artifact -c <cfg>` | 运行 `artifacts:` 块里的 `ArtifactProducer` |
| `eval -c <cfg> [--checkpoint <dir>] [--json out.json] [--max-batches N]` | 困惑度 + EvalSuite 指标 |
| `estimate -c <cfg> [--json f.json]` | 可训练参数量、显存上界、tokens/s 估算 |
| `regression-gate -c <cfg> --metric <name> --threshold <f> [--op <]` | CI 门控；失败退出码 1 |
| `sweep -c <cfg> -s <sweep.yaml> [--strategy grid\|random\|optuna]` | 超参扫描 |
| `compare <run_a> <run_b> … [--metric M] [--output f.md\|f.json] [--png p.png]` | 配置 diff + 指标表 + fork 谱系 |
| `fork --from <ckpt> -c <cfg>` | 从 checkpoint 分叉并记录 lineage |
| `doctor --run <dir>` | 对一次 run 的聚合诊断 |
| `freeze-step --run <dir> --step N` | 捕获单步重放包 |
| `replay-step <bundle.zip>` | 重放一个冻结步包 |
| `replay --run <dir>` | 重放最近的崩溃包 / 冻结步 |
| `profile -c <cfg> [--steps N]` | `torch.profiler` chrome trace |
| `lineage tag/untag/pin/invalidate/gc/prune-orphans/graph` | SQLite lineage 操作 |
| `migrate config [--to-profiles] / artifact-header / checkpoint` | schema 迁移（写 `.pre-migration-bak` 备份） |
| `convert-checkpoint --from <fmt> --to <fmt> --path <ckpt> [--out <out>]` | `.pt` / `.safetensors` / HF 互转 |
| `export --to <fmt> --ckpt <step_dir> --out <out> [-c <cfg>]` | 导出权重；`hf`/`gguf` 路径需 `-c`；`gguf` 还需 PATH 上有 llama.cpp |

## Override 语法

`train`（及大多数读配置的命令）接受 OmegaConf 风格的位置 override：

```bash
lighttrain train -c cfg.yaml trainer.max_steps=5000           # 覆盖字段
lighttrain train -c cfg.yaml optim.lr=1e-3 trainer.grad_clip=0.5
lighttrain train -c cfg.yaml model=mamba2                     # 选 model profile
lighttrain train -c cfg.yaml model_profiles.default.d_model=256
lighttrain train -c cfg.yaml "exp=my_exp_v2"                  # 字符串加引号
lighttrain train -c cfg.yaml "++trainer.beta=0.1"            # ++ 强制新增键
lighttrain train -c cfg.yaml "~scheduler"                     # ~ 删除键
```

## 相关

- [配置](configuration.zh-CN.md) —— recipe 字段含义
- [诊断](../operations/diagnostics.zh-CN.md) —— `doctor` / `replay` / `freeze-step`
- [数据与 PrepGraph](../concepts/data-prepgraph.zh-CN.md) —— `prep*` 命令
