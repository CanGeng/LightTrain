# Getting started

> [中文版](getting-started.zh-CN.md) · [Docs index](../README.md)

## Install

```bash
git clone <this-repo> lighttrain && cd lighttrain
pip install -e .
```

Optional extras:

```bash
pip install -e ".[peft]"          # LoRA / IA³ / AdaLoRA adapters
pip install -e ".[peft,quant]"    # + bitsandbytes 4-bit (Linux + CUDA only)
```

## 60-second quickstart

```bash
lighttrain init my_project        # scaffold a project (recipe + README + dirs)
cd my_project
lighttrain dry-run -c cfg.yaml    # resolve & print the config, no training
lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50-step smoke run
```

The generated `cfg.yaml` is fully runnable out of the box (tiny_lm + byte
tokenizer) and is heavily commented as a living tutorial — uncomment the
optional blocks (`models:`, `parallel:`, `prep_graph:`, PEFT, …) to grow it.

## What you provide

| File | Required | Notes |
| ---- | -------- | ----- |
| YAML recipe | **yes** | must contain `model:`, `data:`, `optim:` |
| training data | **yes** | format depends on the dataset (below) |
| pretrained weights | optional | only for `hf_causal` (HF name or local path) |

Dataset formats:

| dataset | format | path example |
| ------- | ------ | ------------ |
| `line_file_text` | one example per line `.txt` | `path: data/corpus.txt` |
| `preference_jsonl` | JSONL with `chosen_*` / `rejected_*` / `labels` | `path: data/prefs.jsonl` |
| `prep_graph` (SFT) | raw JSONL, preprocessed & cached by PrepGraph | `source: jsonl:data/chat.jsonl` |

Minimal recipe skeleton:

```yaml
model: demo                  # select a profile by name
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

Everything else (loss, trainer, engine, scheduler, tokenizer, collator) has a
sensible default — see [Configuration](configuration.md) for the fallback table.

## What you get

A self-contained run capsule under `runs/<exp>/<ts>-<slug>-<hash>/`:

```
config.snapshot.yaml   # the YAML exactly as supplied
config.resolved.yaml   # after merge / overrides / interpolation
env.json               # python / torch / CUDA / git sha / argv
logs/metrics.jsonl     # per-step metrics (jsonl logger)
checkpoints/
  step_500/{model.safetensors, optimizer.pt, scheduler.pt, rng.pt, manifest.json}
  last.json            # {"target": "step_500"}  — pointer, resolve as checkpoints/<target>
  best.json
```

`manifest.json` is written **last**: a checkpoint dir without it is treated as
incomplete and skipped on resume.

## Next steps

- [CLI reference](cli.md) — all commands and flags
- [Configuration](configuration.md) — the full YAML schema
- [Training paradigms](../concepts/training.md) — SFT, preference, RL, distillation
- [Recipe index](../extending/recipes.md) — pick a bundled recipe to start from
