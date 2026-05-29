# lighttrain 实战运行手册 (User Guide)

> 本手册帮助你从零开始完成一次完整训练：准备文件 → 编写配置 → 启动命令 → 查看产出。
> 所有内容均来自代码库，无臆造。

---

## 目录

1. [一次完整训练的准入与产出 (I/O)](#1-io)
2. [命令行工具 (CLI) 指南](#2-cli)
3. [配置文件 (YAML) 编写指南](#3-yaml)
4. [分布式训练（parallel:）](#4-parallel)
5. [程序的默认防呆行为](#5-default-behaviors)
6. [内部组件的耦合工作流](#6-under-the-hood)

---

## 1. I/O

### 1.1 需要准备什么

| 文件 | 是否必须 | 说明 |
|------|---------|------|
| YAML 配置文件 | **必须** | 需包含 `model:`、`data:`、`optim:` 三个节点 |
| 训练数据文件 | **必须** | 格式取决于 dataset 类型（见下） |
| 预训练权重 | 可选 | 用 `hf_causal` 时填写 HF 模型名或本地路径 |

**数据格式对应关系：**

| dataset 名称 | 数据格式 | 示例路径写法 |
|-------------|---------|------------|
| `line_file_text` | 每行一条文本的 `.txt` 文件 | `path: data/corpus.txt` |
| `preference_jsonl` | 每行含 `chosen_input_ids` + `rejected_input_ids` + `labels` 的 JSONL | `path: data/prefs.jsonl` |
| `prep_graph` (SFT) | 原始 JSONL，经 PrepGraph 预处理后自动缓存 | `source: jsonl:data/chat.jsonl` |

**最小配置骨架：**

```yaml
model:
  name: tiny_lm          # 或 hf_causal 等
  ...

data:
  name: simple
  dataset:
    name: line_file_text
    path: /path/to/corpus.txt
    max_len: 128
  batch_size: 4

optim:
  name: adamw
  lr: 3e-4
```

---

### 1.2 训练结束后产出什么

运行目录格式：`<run_root>/<exp_slug>/<YYYYMMDD-HHMMSS>-<slug>-<hash8>/`

默认 `run_root = "runs"`，`exp = "default"`。

```
runs/
└── tiny_pretrain/
    └── 20260527-084145-tiny_pretrain-cf8e566b/
        │
        ├── config.snapshot.yaml      # 你传入的原始 YAML（未解析）
        ├── config.resolved.yaml      # 合并 defaults、插值后的最终配置
        ├── env.json                  # 运行环境元信息（Python 版本、GPU 型号、git sha…）
        │
        ├── logs/
        │   └── metrics.jsonl         # 每步指标，JSON Lines 格式（启用 jsonl logger 时）
        │
        ├── checkpoints/
        │   ├── step_500/
        │   │   ├── model.safetensors # 模型权重（safetensors 格式）
        │   │   ├── optimizer.pt      # 优化器状态
        │   │   ├── scheduler.pt      # 学习率调度器状态
        │   │   ├── trainer.pt        # 训练器状态（step / epoch / global_step）
        │   │   ├── rng.pt            # 完整随机数状态（Python / NumPy / torch / CUDA）
        │   │   ├── data_module.pt    # 数据加载器状态（采样器位置，用于断点续训）
        │   │   └── manifest.json     # 最后写入；文件存在 = checkpoint 完整
        │   ├── step_1000/
        │   │   └── ...
        │   ├── last.json             # 指向最新 checkpoint 的指针
        │   └── best.json            # 最佳 checkpoint 指针（需启用 best_ckpt callback）
        │
        ├── tb_logs/                  # TensorBoard 事件文件（启用 tensorboard logger 时）
        ├── lineage.sqlite            # 数据血缘数据库（PrepGraph 场景）
        ├── frozen_steps/             # 冻结步快照（lab 模式自动每 1000 步保存一次）
        └── diagnostics/              # 崩溃现场 / OOM 报告（lab 模式）
```

> **关键约定**：`manifest.json` 是 checkpoint 目录中最后写入的文件。只有它存在，该 checkpoint 才被认为完整。断点续训时，程序会自动跳过不完整的目录。

> **`last.json` / `best.json` schema**：两个指针文件都形如
> `{"target": "<step_N>"}`，其中 `<step_N>` 是 `checkpoints/` 下的**目录名**（不是绝对路径）。
> 解析方式：`<run_dir>/checkpoints/<target>`。下游 launcher 直接读 `target`
> 拼出 checkpoint 目录即可，不要假设有 `path` 字段。

---

## 2. CLI 指南

### 2.1 安装与入口

```bash
pip install -e .
lighttrain --help
```

入口定义于 `pyproject.toml`：

```toml
[project.scripts]
lighttrain = "lighttrain.cli._app:app"
```

**全局选项：**

| 选项 | 说明 |
|------|------|
| `--version` | 显示版本并退出 |
| `--quiet` | 抑制非必要输出 |
| `--verbose` | 详细输出 |

---

### 2.2 核心命令

#### `train` — 启动训练

```bash
lighttrain train -c <config.yaml> [OVERRIDES...]
```

| 参数 | 短选项 | 类型 | 默认 | 说明 |
|------|-------|------|------|------|
| `--config` | `-c` | Path | **必填** | YAML 配置文件路径 |
| `OVERRIDES` | — | `key=value` 位置参数 | 无 | OmegaConf 风格覆盖（见下） |
| `--mode` | — | `lab\|prod` | 配置内值 | 覆盖 mode 字段 |
| `--print-config` | — | bool | False | 打印解析后的配置后退出，不执行训练 |
| `--no-cache` | — | bool | False | 禁用所有缓存 |
| `--allow-stale-artifact` | — | bool | False | 跳过 artifact 头部版本检查 |

**Override 语法**（支持任意 YAML 标量）：

```bash
# 覆盖单个字段
lighttrain train -c pretrain_causal.yaml trainer.max_steps=5000

# 覆盖多个字段
lighttrain train -c pretrain_causal.yaml optim.lr=1e-3 trainer.grad_clip=0.5

# 插值引用（字符串用引号）
lighttrain train -c pretrain_causal.yaml "exp=my_exp_v2"

# 强制新增键（++ 前缀）
lighttrain train -c pretrain_causal.yaml "++trainer.beta=0.1"

# 删除键（~ 前缀）
lighttrain train -c pretrain_causal.yaml "~scheduler"
```

---

#### `resume` — 断点续训

```bash
lighttrain resume --run runs/tiny_pretrain/20260527-... [-c new_config.yaml] [--mode functional]
```

| 参数 | 短选项 | 类型 | 默认 | 说明 |
|------|-------|------|------|------|
| `--run` | — | Path | **必填** | 现有 run 目录 |
| `--config` | `-c` | Path | None | 不填则使用 run_dir 内的 `config.snapshot.yaml` |
| `--mode` | — | str | `"functional"` | `functional`（允许代码变化）或 `exact`（严格复现） |

---

#### `dry-run` — 验证配置不训练

```bash
lighttrain dry-run -c config.yaml [OVERRIDES...]
```

验证配置文件能否正确解析并实例化所有组件，不执行任何训练步骤。

---

#### `overfit` — 过拟合冒烟测试

```bash
lighttrain overfit -c config.yaml [--n 200] [OVERRIDES...]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--n` | `200` | 运行步数（应能在单 batch 上过拟合） |

---

#### `inspect-data` — 查看数据样本

```bash
lighttrain inspect-data -c config.yaml [--n 32] [--decoded]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--n` | `32` | 查看样本数量 |
| `--decoded` | False | 显示 tokenizer 解码后的文本 |

---

#### `prep` — 运行数据预处理流水线 (PrepGraph)

```bash
lighttrain prep -c config.yaml [--dry-run] [--workers 4] [--pool thread]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--dry-run` | False | 只计算 fingerprint，不写磁盘 |
| `--workers` | `1` | 并行 worker 数 |
| `--pool` | `"thread"` | `thread` 或 `process` |

> `prep` 命令需要配置文件中包含 `prep_graph:` 节点。

---

#### `eval` — 评估模型（困惑度）

```bash
lighttrain eval -c config.yaml [--checkpoint runs/.../step_1000] [--json report.json] [--max-batches 50]
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--checkpoint` | None | 指定 checkpoint 目录；不填则使用 `last.json` 指向的最新 ckpt |
| `--json` | None | 将评估报告写为 JSON |
| `--max-batches` | `0` | 限制评估批次数（0 = 全量） |

---

#### `estimate` — 估算训练资源消耗

```bash
lighttrain estimate -c config.yaml [--json estimate.json]
```

---

#### `export` — 导出模型

```bash
lighttrain export --to hf --ckpt runs/.../step_1000 --out ./exported_model -c config.yaml
```

| 参数 | 说明 |
|------|------|
| `--to` | 目标格式：`safetensors` / `hf` / `gguf` |
| `--ckpt` | checkpoint 目录（`step_<N>/`） |
| `--out` | 输出路径或目录 |
| `-c/--config` | YAML 配置（`hf` / `gguf` 导出时需要） |

---

#### `doctor` — 诊断运行目录

```bash
lighttrain doctor --run runs/tiny_pretrain/20260527-...
```

---

#### 其他实用命令

| 命令 | 功能 |
|------|------|
| `prep-graph -c <cfg> [--out graph.mmd]` | 渲染 PrepGraph DAG 结构（Mermaid 或 DOT 格式） |
| `prep-status -c <cfg>` | 查看各 PrepGraph 节点缓存状态 |
| `prep-clean -c <cfg> [--orphans] [--dry-run]` | 清理过期缓存 |
| `fork --from <ckpt> -c <cfg>` | 从 checkpoint fork 一个新实验 |
| `replay --run <dir>` | 重放崩溃现场的最后一步 |
| `freeze-step --run <dir> --step <n>` | 手动冻结某步快照（用于调试） |
| `replay-step <bundle.zip>` | 重放一个冻结步快照 |
| `profile -c <cfg> [--steps 50]` | 运行 PyTorch Profiler |
| `regression-gate -c <cfg> --metric loss --threshold 2.5 --op <` | CI 回归检查门控 |
| `convert-checkpoint --from pt --to safetensors --path <ckpt>` | 转换 checkpoint 格式 |
| `sweep -c <cfg> -s sweep_spec.yaml [--strategy grid]` | 超参数扫描 |
| `compare <run1> <run2> [--png chart.png]` | 对比多次实验曲线 |
| `init <project_dir>` | 初始化新项目目录 |

---

## 3. YAML 配置文件编写指南

### 3.1 必填节点

以下三个节点**缺少任何一个**都会在启动时立即报错：

| 节点 | 报错信息 |
|------|---------|
| `model:` | `RuntimeError: recipe is missing 'model:' section` |
| `data:` | `RuntimeError: recipe is missing 'data:' section` |
| `optim:` | `RuntimeError: recipe is missing 'optim:' section` |

---

### 3.2 全字段参考表（含默认值）

#### 根节点（RootConfig）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `"lab"` 或 `"prod"` | `"lab"` | lab 模式自动挂载诊断 callback；prod 模式精简化 |
| `seed` | int | `42` | 全局随机种子（PyTorch / NumPy / Python random） |
| `exp` | str | `"default"` | 实验名，影响 run_dir 命名 |
| `run_root` | str | `"runs"` | 输出根目录 |
| `run_dir` | str 或 None | None | 手动指定完整输出路径（覆盖 run_root + exp） |
| `user_modules` | list[str] | `[]` | 额外加载的 Python 模块路径（用于注册自定义组件） |

#### `trainer:` 节点（TrainerSection）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"pretrain"` | 训练器类型（可选值见注册表：`dpo`、`grpo`、`ppo`、`orpo` 等） |
| `max_steps` | int | `1000` | 训练总步数 |
| `val_every` | int | `0` | 验证间隔步数（`0` = 不验证） |
| `ckpt_every` | int | `500` | checkpoint 保存间隔 |
| `log_every` | int | `50` | 指标日志间隔 |
| `grad_clip` | float | `1.0` | 梯度裁剪阈值（`torch.nn.utils.clip_grad_norm_`） |
| `accumulate` | int | `1` | 梯度累积步数 |

#### `engine:` 节点（EngineSection）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"standard"` | 引擎实现名称 |
| `mixed_precision` | `"no"` / `"fp16"` / `"bf16"` | `"bf16"` | 混合精度模式 |
| `update_rule.name` | str | `"standard"` | 梯度更新规则（`standard` / `sam` / `mezo` / `rl`） |

#### `data:` 节点内部（SimpleDataModule）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"simple"` | 自动补全，不写则框架填入 |
| `batch_size` | int | `8` | 批大小 |
| `num_workers` | int | `0` | DataLoader worker 数 |
| `pin_memory` | bool | `False` | 是否锁页内存 |
| `drop_last` | bool | `False` | 是否丢弃最后不完整的 batch |

---

### 3.3 组件引用语法

组件（model / optimizer / loss 等）支持两种写法：

**写法 A：registry 短名称**

```yaml
optim:
  name: adamw
  lr: 3e-4
  weight_decay: 0.1
```

**写法 B：直接 import（`_target_`）**

```yaml
optim:
  _target_: torch.optim.AdamW
  lr: 3e-4
  weight_decay: 0.1
```

---

### 3.4 配置组合（defaults）

可以用 `defaults:` 列表合并多个 YAML 文件（类似 Hydra）：

```yaml
defaults:
  - base/model_tiny        # 相对于当前文件的路径（不含 .yaml 后缀）
  - ../shared/adamw_optim

exp: my_finetune_run

trainer:
  max_steps: 3000           # 覆盖 base 中的值
```

---

### 3.5 OmegaConf 插值

```yaml
seed: 1337

scheduler:
  name: warmup_cosine
  total_steps: ${trainer.max_steps}    # 引用同配置中的字段
  warmup_steps: 100

data:
  dataset:
    seed: ${seed}                       # 引用根节点 seed

optim:
  lr: ${oc.env:LR,3e-4}               # 读取环境变量 LR，缺省 3e-4
```

---

### 3.6 完整可运行示例（预训练）

来源：[recipes/pretrain_causal.yaml](recipes/pretrain_causal.yaml)

```yaml
mode: lab
seed: 1337
exp: tiny_pretrain
run_root: runs

model:
  name: tiny_lm
  vocab_size: 260
  d_model: 256
  n_layers: 4
  n_heads: 8
  max_seq_len: 128
  dropout: 0.0

data:
  name: simple
  dataset:
    name: line_file_text
    path: tests/fixtures/tiny_corpus.txt
    max_len: 128
  tokenizer:
    name: byte
  collator:
    name: causal_lm
    max_len: 128
  sampler:
    name: shuffle
    seed: ${seed}
  batch_size: 4
  num_workers: 0

loss:
  name: cross_entropy

optim:
  name: adamw
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1

scheduler:
  name: warmup_cosine
  warmup_steps: 50
  total_steps: ${trainer.max_steps}
  min_lr_ratio: 0.1

engine:
  name: standard
  mixed_precision: bf16

trainer:
  name: pretrain
  max_steps: 2000
  val_every: 0
  ckpt_every: 500
  log_every: 25
  grad_clip: 1.0
  accumulate: 1

callbacks:
  - { name: throughput, window: 50 }
  - { name: nan_skip, max_skips: 10 }
  - { name: best_ckpt, monitor: loss, mode: min }

logger:
  - { name: console, log_every: 25 }
  - { name: jsonl }
  - { name: tensorboard }
```

---

## 4. 分布式训练（`parallel:`）

`parallel:` 块让 lighttrain 从单卡自然扩展到多卡分布式，无需修改模型或 Trainer 代码。
缺省不填 `parallel:` 等同于 `dp=tp=pp=ep=1`（单卡退化模式）。

### 4.1 `parallel:` 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `backend` | `"nccl"` \| `"gloo"` | `"nccl"` | `torch.distributed` 通信后端；gloo 用于 CPU / CI 测试 |
| `dp` | int | 1 | Data Parallel 副本数 |
| `tp` | int | 1 | Tensor Parallel 分片数（TP×DP×PP 须等于总 GPU 数） |
| `pp` | int | 1 | Pipeline Parallel 阶段数 |
| `ep` | int | 1 | Expert Parallel 大小；须整除 `dp`，是 DP 子分组 |
| `sp` | bool | false | 是否启用 Sequence Parallelism（与 TP 配合使用） |
| `force_cpu` | bool | false | 强制所有张量使用 CPU；配合 `backend: gloo` 做多进程通信测试，无需 GPU |
| `grad_sync` | sub-block | `{name: noop}` | 梯度同步策略，见下表 |
| `tensor_parallel` | sub-block \| null | null | TP 手术配置 |
| `pipeline` | sub-block \| null | null | PP 调度配置 |

### 4.2 `grad_sync:` 策略

| `name` | 实现 | 说明 |
|---|---|---|
| `noop` | `NoopGradSyncStrategy` | 单卡直通（默认） |
| `ddp` | `DDPStrategy` | `torch.nn.parallel.DistributedDataParallel` |
| `fsdp` | `FSDPStrategy` | `torch.distributed.fsdp.FullyShardedDataParallel` |
| `deepspeed` | `ZeROStrategy` | DeepSpeed ZeRO-1/2/3 engine |

`ddp` 额外可选字段：`find_unused_parameters: false`
`fsdp` 额外可选字段：`sharding_strategy`, `state_dict_type` (`"full"` 或 `"sharded"`)

### 4.3 `tensor_parallel:` 子块

```yaml
tensor_parallel:
  auto_plan_for: llama    # 内置方案：llama / gpt2 / mistral
  # 或手动指定：
  # plan:
  #   - { path: "model.layers.*.self_attn.q_proj", style: colwise }
  #   - { path: "model.layers.*.self_attn.o_proj", style: rowwise }
```

### 4.4 `pipeline:` 子块

```yaml
pipeline:
  n_microbatches: 8          # 微批次数（建议 ≥ pp 阶段数）
  schedule: 1f1b             # 调度方案：1f1b / gpipe / interleaved_1f1b
  stage_spec:
    - { layers: "model.embed_tokens,model.layers.0-15" }
    - { layers: "model.layers.16-31,model.norm,lm_head" }
```

### 4.5 torchrun 启动格式

```bash
torchrun --nproc_per_node=<N> -m lighttrain.cli train -c <config.yaml>
```

多机多卡时另加 `--nnodes`, `--node_rank`, `--master_addr`, `--master_port`。

### 4.6 典型配置示例

**单机 DDP（4 GPU）**

```yaml
parallel:
  backend: nccl
  dp: 4
  grad_sync:
    name: ddp
    find_unused_parameters: false
```

**FSDP + 梯度累积**

```yaml
parallel:
  backend: nccl
  dp: 8
  grad_sync:
    name: fsdp
    sharding_strategy: FULL_SHARD
    state_dict_type: full
trainer:
  accumulate: 4
```

**gloo + CPU 多进程通信测试（无 GPU）**

```yaml
parallel:
  backend: gloo
  dp: 4
  force_cpu: true
  grad_sync:
    name: ddp
    find_unused_parameters: false
engine:
  mixed_precision: "no"
```

> `force_cpu: true` 会跳过 CUDA DeviceMesh 初始化，所有张量在 CPU 上运行。
> 适用于 CI 环境验证进程组初始化、AllReduce 通信、训练循环不死锁等场景。

完整 recipe 示例见 [`frontier_plugins/distributed/recipes/`](../plugins/distributed/recipes/)。

---

## 5. 默认防呆行为

如果你只提供 `model:` + `data:` + `optim:`，框架会自动填充以下默认值：

| 组件 | 自动 fallback | 代码位置 |
|------|--------------|---------|
| `loss` | `CrossEntropyLoss`（shift logits + CE，适合 Causal-LM） | `_runtime.py:_build_loss` |
| `trainer` | `PretrainTrainer`（max_steps=1000，每 500 步 ckpt） | `_schema.py:TrainerSection` |
| `engine` | `StandardEngine`（bf16 混合精度，标准梯度裁剪） | `_schema.py:EngineSection` |
| `data.name` | 自动补 `"simple"`（即 `SimpleDataModule`） | `_runtime.py:_build_data` |
| `tokenizer`（data 内未指定） | `ByteTokenizer`（UTF-8 字节级，vocab_size=260） | `_module.py` |
| `collator`（data 内未指定） | `CausalLMCollator`（右填充至批内最长） | `_module.py` |
| `scheduler` | 无（lr 全程固定） | `_runtime.py:_build_scheduler` |
| `callbacks` | `[]`（空列表；lab 模式额外自动附加诊断 callbacks） | `_runtime.py:_build_callbacks` |
| `logger` | `[]`（不记录任何日志） | `_runtime.py:_build_logger` |
| device | CUDA 可用时自动选 GPU，否则 CPU | `_runtime.py:_select_device` |

**lab 模式额外自动挂载的 callbacks（无需在配置中声明）：**

| Callback | 触发条件 | 作用 |
|---------|---------|------|
| `InvariantsCallback` | 始终（读 `invariants:` 节点） | 运行内置 invariant 检查（loss_finite 等） |
| `FrozenStepCallback` | lab 模式，每 1000 步 | 冻结步快照，供 `replay` 调试 |
| `FileSignalsCallback` | lab 模式 | 监听 `control/` 目录文件信号（动态调 lr / 暂停） |
| `CallbackIsolationSink` | 始终 | 捕获 callback 异常写入 `diagnostics/callback_failures.jsonl` |

---

## 6. 内部组件初始化顺序

以下是 `lighttrain train -c config.yaml` 执行时内部的完整流程（来源：`lighttrain/cli/_runtime.py:setup_run_from_config`）：

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. 配置加载                                                         │
│     YAML 读取 → defaults 合并 → CLI overrides 注入 → OmegaConf 插值 │
│     → Pydantic 验证 (RootConfig)                                     │
├─────────────────────────────────────────────────────────────────────┤
│  2. 准备阶段                                                         │
│     user_modules 导入（@register 装饰器生效）                        │
│     seed_everything(cfg.seed)                                        │
│     run_dir 创建 + config.snapshot.yaml / env.json 写入             │
├─────────────────────────────────────────────────────────────────────┤
│  3. 组件实例化（顺序严格）                                           │
│                                                                     │
│  [Phase A] 拓扑初始化                                                │
│     parallel_ctx = _init_parallel(cfg)                              │
│       └─ parallel: 缺省或 dp×tp×pp=1 → ParallelContext.single_gpu() │
│       └─ 否则 → ParallelContext.from_env(cfg.parallel)              │
│             init_process_group(backend) + DeviceMesh 构建           │
│     device = parallel_ctx.local_device                              │
│       └─ force_cpu=true → cpu；CUDA 可用 → cuda:{local_rank}        │
│                                                                     │
│  [Phase B] 模型构建 + Tensor/Sequence/Expert 并行手术               │
│     model = build_model(cfg)                                        │
│     mp_strategy = _build_model_parallel_strategy(cfg)               │
│     if mp_strategy:                                                 │
│         model = mp_strategy.apply(model, parallel_ctx)             │
│         (TP/SP/EP 手术在此完成，参数已按 rank 切分)                 │
│                                                                     │
│  [Phase C] Pipeline 分阶段（pp > 1 时）                             │
│     pipeline schedule 构建，模型按 stage_spec 切割到各 pp rank      │
│                                                                     │
│  [Phase D] 梯度同步包装                                             │
│     grad_sync = _build_grad_sync_strategy(cfg)  ← noop/ddp/fsdp/ds │
│     if grad_sync:                                                   │
│         model, optimizer, loader =                                  │
│             grad_sync.prepare(model, optimizer_factory,             │
│                               loader, parallel_ctx, device=device)  │
│         (FSDP 在 prepare 内包装后再调 optimizer_factory)            │
│     else:                                                           │
│         model = model.to(device)                                    │
│         optimizer = build_optimizer(cfg, model)                     │
│                                                                     │
│  [公共] data_module / scheduler / loss / callbacks / logger / ckpt  │
│     data_module = build_data(cfg)                                   │
│       └─ tokenizer → dataset → collator → sampler → DataLoader     │
│     scheduler = build_scheduler(cfg, optimizer)                     │
│     loss_fn   = build_loss(cfg)      ← 默认 CrossEntropyLoss       │
│     callbacks = build_callbacks(cfg)                                │
│     logger    = build_logger(cfg, run_dir)                          │
│     ckpt_manager = CheckpointManager(run_dir)                       │
├─────────────────────────────────────────────────────────────────────┤
│  4. 引擎组装                                                         │
│     update_rule = StandardUpdateRule(grad_clip, accumulate)         │
│       （RL trainer 自持 RLUpdateRule，跳过 model(**batch) forward） │
│     accelerator = build_accelerator(mixed_precision)                │
│     engine = StandardEngine(update_rule, loss_fn, accelerator)      │
│     ctx.parallel_ctx = parallel_ctx                                 │
│     ctx.grad_sync    = grad_sync                                    │
├─────────────────────────────────────────────────────────────────────┤
│  5. Trainer 装配                                                     │
│     trainer = Trainer(                                              │
│         engine, data_module, optimizer, scheduler,                  │
│         callbacks, logger, ckpt_manager, max_steps                  │
│     )                                                               │
│     _auto_attach_m4_callbacks(cfg, trainer)  ← lab 模式诊断 hooks  │
├─────────────────────────────────────────────────────────────────────┤
│  6. 训练循环                                                         │
│     trainer.fit()                                                   │
│       on_train_start                                                │
│       while step < max_steps:                                       │
│         batch = next(data_loader)                                   │
│         engine.step(batch, ctx)                                     │
│           └─ update_rule.step(model, batch, ctx)                   │
│               ├─ [StandardUpdateRule] model(**batch) → loss_fn → loss │
│               └─ [RLUpdateRule] 跳过 forward，ctx.loss_fn 直接得 loss │
│               └─ grad_sync.backward(loss, model)    ← 含 no_sync   │
│               └─ grad_sync.clip_grad_norm(...)                      │
│               └─ grad_sync.optimizer_step(optimizer, model)         │
│               └─ scheduler.step()                                   │
│         [每 log_every 步，rank-0] logger.log_scalars(metrics)      │
│         [每 ckpt_every 步，rank-0] ckpt_manager.save(step, state)  │
│       on_train_end → logger.flush()                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**设计要点：**
- **并行拓扑先于一切**：`parallel_ctx` 在 model 构建前就绪，`device` 从它取，避免任何组件在错误设备上初始化。
- **TP/SP/EP 必须在 FSDP/DDP 包装之前**：张量并行手术改变参数形状，sharding 必须看到切分后的形状。
- **FSDP optimizer 后置**：`grad_sync.prepare()` 接受 `optimizer_factory: Callable`，在模型包装完毕后才调用，符合 FSDP 要求（先 wrap 再 build optimizer）。
- **rank-0 gating**：`_is_main_process` 门控所有 IO（日志、checkpoint、crash bundle），多卡下只有 global rank 0 写出。
- `optimizer.build(model)` 发生在 model 构建**之后**，确保 optimizer 能看到完整的参数组。
- `loss_fn` 作为独立组件传入 `engine`，而不是写死在 trainer 里，因此可以通过配置随时替换。
- `manifest.json` 永远最后写入：中途崩溃不会产生被误读的"半截 checkpoint"。
