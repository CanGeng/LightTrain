# 数据与 PrepGraph

> [English](data-prepgraph.md) · [文档索引](../README.md)

## simple 数据模块

多数 run 用 `data: { name: simple, ... }`，把 dataset + tokenizer + collator +
sampler 包成 DataLoader：

```yaml
data:
  name: simple
  dataset:   { name: line_file_text, path: corpus.txt, max_len: 256 }
  tokenizer: { name: byte }
  collator:  { name: causal_lm, max_len: 256 }
  sampler:   { name: shuffle, seed: ${seed} }
  batch_size: 4
  num_workers: 0
```

内置（完整清单见 [reference/registry.zh-CN.md](../reference/registry.zh-CN.md)）：

- **dataset**：`line_file_text`、`preference_jsonl`、`artifact_joined`
- **collator**：`causal_lm`、`preference`、`multimodal`
- **sampler**：`shuffle`、`sequential`、`length_grouped`、`curriculum`、
  `stateful_resumable`
- **tokenizer**：`byte`（vocab 260）

## PrepGraph —— 内容寻址的数据预处理

PrepGraph 是预处理节点的 DAG。每个节点 fingerprint =
`sha256(node.kind + node.schema_kind + SCHEMA_VERSION[schema_kind] +
code_version + 规范化 config + 排序后的 upstream_fps)`；结果原子落盘，
`MANIFEST_COMPLETE.json` 最后写。输入与 config 不变的节点直接复用缓存；改动上游
只重算受影响的子树。

缓存根目录因入口而异：独立 `lighttrain prep` 默认落到
`runs/<exp>/prep/<kind>/<name>/<fp>/`；训练内嵌的 `prep_graph` DataModule 默认落到
`<run_dir>/prep/...`。

训练数据引用 PrepGraph 输出（`data.source: prep_graph:<terminal>` 或
`data.name: prep_graph`）时 `lighttrain train` 会自动跑 PrepGraph，很少需要单独 `prep`。

节点种类：`load`、`tokenize`、`chunk`、`pack`、`mix`、`join`、`index`、
`validate`、`materialize`。

### 打包策略（`pack` 节点）

`strategy:` 决定文档如何填充上下文窗口，各自吐 `truncation_rate` /
`token_utilization` 指标：

- `concat_chunk`（默认）—— 无 padding 基线
- `next_fit` —— 贪心 pad-flush
- `best_fit` —— best-fit-decreasing 装箱（opt-in，更少截断）

### 命令

```bash
lighttrain prep        -c cfg.yaml [--dry-run] [--workers N] [--pool thread|process]
lighttrain prep-graph  -c cfg.yaml [--out g.dot]   # 渲染 DAG
lighttrain prep-status -c cfg.yaml                 # 各节点缓存状态
lighttrain prep-clean  -c cfg.yaml [--orphans]
lighttrain inspect-data -c cfg.yaml --n 4 --decoded
```

## 恢复与数据位置

checkpoint 保存 sampler 状态与已消费 batch 数。用 `simple` DataModule 且 sampler 支持
seek（`sequential` / `shuffle` / `stateful_resumable`）时，resume 能在 epoch 中途逐步
精确定位 sampler（不受 DataLoader 预取深度影响）。`PrepGraphDataModule` 与
`length_grouped` / `curriculum` sampler **未**实现按 consumed-index 的 `seek()`，那里的
resume 是*功能性*恢复（状态已恢复）而非逐步精确。见
[诊断](../operations/diagnostics.zh-CN.md) 与 [CLI](../guide/cli.zh-CN.md) 里的 `resume-verify`。

## 相关

- [配置](../guide/configuration.zh-CN.md) —— `data:` schema
- [扩展](../extending/extending.zh-CN.md) —— 写自定义 dataset / collator / prep 节点
- [reference/registry.zh-CN.md](../reference/registry.zh-CN.md) —— 所有数据组件
