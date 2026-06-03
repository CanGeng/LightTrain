# 快速开始

> [English](getting-started.md) · [文档索引](../README.md)

## 安装

```bash
git clone <this-repo> lighttrain && cd lighttrain
pip install -e .
```

可选 extras：

```bash
pip install -e ".[peft]"          # LoRA / IA³ / AdaLoRA 适配器
pip install -e ".[peft,quant]"    # + bitsandbytes 4-bit（仅 Linux + CUDA）
```

## 60 秒上手

```bash
lighttrain init my_project        # 脚手架（recipe + README + 目录）
cd my_project
lighttrain dry-run -c cfg.yaml    # 解析并打印配置，不训练
lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50 步冒烟
```

生成的 `cfg.yaml` 在旁边放一个 `corpus.txt`（每行一条样本）后即可跑——`lighttrain
init` 只生成 recipe，不生成语料。它用 tiny_lm + byte 分词器，并带有大量教程式注释——
取消注释里面的可选块（`models:`、`parallel:`、`prep_graph:`、PEFT 等）即可逐步扩展。

## 你需要准备什么

| 文件 | 必须 | 说明 |
| ---- | ---- | ---- |
| YAML recipe | **是** | 必须声明模型、数据、优化器（见 [配置](configuration.zh-CN.md#必填节点)） |
| 训练数据 | **是** | 格式取决于 dataset（见下） |
| 预训练权重 | 可选 | 仅 `hf_causal` 需要（HF 名称或本地路径） |

数据格式：

| dataset | 格式 | path 写法 |
| ------- | ---- | --------- |
| `line_file_text` | 每行一条文本 `.txt` | `path: data/corpus.txt` |
| `preference_jsonl` | 含 `chosen_*` / `rejected_*` / `labels` 的 JSONL | `path: data/prefs.jsonl` |
| `prep_graph`（SFT） | 原始 JSONL，经 PrepGraph 预处理并缓存 | `source: jsonl:data/chat.jsonl` |

最小 recipe 骨架：

```yaml
model: demo                  # 按名字选一个 profile
model_profiles:
  demo:
    name: tiny_lm
    vocab_size: 260
    d_model: 256
    n_layers: 4
    n_heads: 4
    max_seq_len: 256

data:
  name: simple
  dataset: { name: line_file_text, path: corpus.txt, max_len: 256 }
  batch_size: 4

optim:
  name: adamw
  lr: 3.0e-4
```

其余（loss、trainer、engine、scheduler、tokenizer、collator）都有合理默认值——
见 [配置](configuration.zh-CN.md) 的 fallback 表。

## 训练产出什么

`runs/<exp>/<时间戳>-<slug>-<hash>/` 下的自包含运行胶囊：

```
config.snapshot.yaml   # 你传入的原始 YAML
config.resolved.yaml   # 合并 / override / 插值后的最终配置
env.json               # python / torch / CUDA / git sha / argv
logs/metrics.jsonl     # 每步指标（jsonl logger）
checkpoints/
  step_500/{model.safetensors, optimizer.pt, scheduler.pt, rng.pt, manifest.json}
  last.json            # {"target": "step_500"} —— 指针，按 checkpoints/<target> 解析
  best.json
```

`manifest.json` **最后写入**：缺它的 checkpoint 目录视为不完整，resume 时跳过。

## 下一步

- [命令行参考](cli.zh-CN.md) —— 所有命令与参数
- [配置](configuration.zh-CN.md) —— 完整 YAML schema
- [训练范式](../concepts/training.zh-CN.md) —— SFT、偏好、RL、蒸馏
- [配方索引](../extending/recipes.zh-CN.md) —— 挑一个内置 recipe 起步
