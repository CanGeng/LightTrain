# CLI reference

> [中文版](cli.zh-CN.md) · [Docs index](../README.md)

Entry point (from `pyproject.toml`): `lighttrain = "lighttrain.cli._app:app"`.

Global options: `--version`, `--quiet`, `--verbose`.

## Commands

| Command | Description |
| ------- | ----------- |
| `init <path> [--force]` | Scaffold a project: commented `cfg.yaml` + README + `runs/` + `artifacts/` |
| `dry-run -c <cfg> [--build]` | Resolve & print the config; `--build` also constructs the model (verifies the `model_profiles` selector) |
| `train -c <cfg> [OVERRIDES…] [--eval] [--output-summary f.json]` | Full training loop; auto-runs PrepGraph if `prep_graph:` is set |
| `resume --run <dir> [-c cfg] [--mode functional\|exact]` | Resume from a run dir (defaults to its `config.snapshot.yaml`) |
| `resume-verify -c <cfg> --phase1-steps N --phase2-steps M [--tol 1e-2]` | Verify resume == single pass by comparing step-aligned losses |
| `overfit -c <cfg> --n N` | Overfit on N batches (smoke test) |
| `inspect-data -c <cfg> [--n 32] [--decoded]` | Decoded batch preview, length histogram, label-mask coverage |
| `prep -c <cfg> [--dry-run] [--workers N] [--pool thread\|process]` | Run the PrepGraph data pipeline only |
| `prep-graph -c <cfg> [--out g.dot]` | Render the PrepGraph DAG (Mermaid / DOT) |
| `prep-status -c <cfg>` | Cache status per prep node |
| `prep-clean -c <cfg> [--orphans] [--dry-run]` | Remove cached prep artefacts |
| `produce-artifact -c <cfg>` | Run an `ArtifactProducer` from the `artifacts:` block |
| `eval -c <cfg> [--checkpoint <dir>] [--json out.json] [--max-batches N]` | Perplexity + EvalSuite metrics |
| `estimate -c <cfg> [--json f.json]` | Trainable params, memory bound, tokens/s estimate |
| `regression-gate -c <cfg> --metric <name> --threshold <f> [--op <]` | CI gate; exits 1 on failure |
| `sweep -c <cfg> -s <sweep.yaml> [--strategy grid\|random\|optuna]` | Hyperparameter sweep |
| `compare <run_a> <run_b> … [--metric M] [--output f.md\|f.json] [--png p.png]` | Config diff + metric table + fork ancestry |
| `fork --from <ckpt> -c <cfg>` | Branch from a checkpoint with lineage |
| `doctor --run <dir>` | Aggregated diagnostics over a run |
| `freeze-step --run <dir> --step N` | Capture a single-step replay bundle |
| `replay-step <bundle.zip>` | Replay a frozen step bundle |
| `replay --run <dir>` | Replay the latest crash bundle / frozen step |
| `profile -c <cfg> [--steps N]` | `torch.profiler` chrome trace |
| `lineage tag/untag/pin/invalidate/gc/prune-orphans/graph` | SQLite lineage operations |
| `migrate config [--to-profiles] / artifact-header / checkpoint` | Schema migrations (writes `.pre-migration-bak`) |
| `convert-checkpoint --from <fmt> --to <fmt> --path <ckpt> [--out <out>]` | Convert `.pt` / `.safetensors` / HF |
| `export --to <fmt> --ckpt <step_dir> --out <out> [-c <cfg>]` | Export weights; `hf`/`gguf` need `-c`; `gguf` needs llama.cpp on PATH |

## Override syntax

`train` (and most config-taking commands) accept OmegaConf-style positional
overrides:

```bash
lighttrain train -c cfg.yaml trainer.max_steps=5000           # set a field
lighttrain train -c cfg.yaml optim.lr=1e-3 trainer.grad_clip=0.5
lighttrain train -c cfg.yaml model=mamba2                     # pick a model profile
lighttrain train -c cfg.yaml model_profiles.default.d_model=256
lighttrain train -c cfg.yaml "exp=my_exp_v2"                  # quote strings
lighttrain train -c cfg.yaml "++trainer.beta=0.1"            # ++ force-adds a new key
lighttrain train -c cfg.yaml "~scheduler"                     # ~ deletes a key
```

## See also

- [Configuration](configuration.md) — what the recipe fields mean
- [Diagnostics](../operations/diagnostics.md) — `doctor` / `replay` / `freeze-step` in context
- [Data & PrepGraph](../concepts/data-prepgraph.md) — the `prep*` commands
