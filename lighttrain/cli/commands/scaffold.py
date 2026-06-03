"""Project scaffolding command: init."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from lighttrain.cli._context import console

_INIT_RECIPE = '''\
# =============================================================================
# lighttrain recipe — runnable out of the box, and a guided tour.
#
#   lighttrain dry-run -c cfg.yaml                      # resolve & print, no train
#   lighttrain train   -c cfg.yaml ++trainer.max_steps=50   # 50-step smoke run
#
# The ACTIVE part below (model/data/optim + sensible defaults) trains a tiny
# byte-level LM immediately — just drop a corpus at ``corpus.txt`` (one example
# per line). Everything under "OPTIONAL" is commented out: uncomment a block to
# grow into SFT / preference / RL / distillation / PEFT / distributed.
#
# Docs: docs/getting-started.md · docs/configuration.md · docs/recipes.md
# =============================================================================

mode: lab                 # lab = auto-attach diagnostics; prod = lean
seed: 1337
exp: demo                 # names the run dir: runs/<exp>/<ts>-<slug>-<hash>/
run_root: runs

# --- model -------------------------------------------------------------------
# The model is a config group: declare named profiles and select one with
# ``model: <name>`` (override on the CLI with ``model=big``). For a frozen
# teacher / multi-model, use ``models:`` instead (see OPTIONAL C).
model: demo
model_profiles:
  demo:
    name: tiny_lm         # built-ins: tiny_lm | hf_causal | tiny_rwkv | tiny_mamba | jepa
    vocab_size: 260       # byte tokenizer = 256 bytes + PAD/BOS/EOS/UNK
    d_model: 256
    n_layers: 4
    n_heads: 4
    max_seq_len: 256
    dropout: 0.0
    # tie_weights: true

# --- data --------------------------------------------------------------------
data:
  name: simple            # simple | prep_graph (DAG-based prep, OPTIONAL E)
  dataset:
    name: line_file_text  # line_file_text | preference_jsonl | artifact_joined
    path: corpus.txt      # <-- put one training example per line here
    max_len: 256
  tokenizer:
    name: byte            # byte (no external deps); hf tokenizers via processors
  collator:
    name: causal_lm       # causal_lm | preference | multimodal
    max_len: 256
  sampler:
    name: shuffle         # shuffle | sequential | length_grouped | curriculum | stateful_resumable
    seed: ${seed}
  batch_size: 4
  num_workers: 0
  # pin_memory: false
  # drop_last: false

# --- loss --------------------------------------------------------------------
# The algorithm lives here, not in a separate trainer (DPO/PPO/… are losses).
loss:
  name: cross_entropy     # ce | mlm | z_loss | composite | dpo | ppo_surrogate | grpo | kl_topk | ...

# --- optimizer ---------------------------------------------------------------
optim:
  name: adamw             # adamw | lion
  lr: 3.0e-4
  betas: [0.9, 0.95]
  weight_decay: 0.1
  # param_groups:                       # regex-based, first-match-wins
  #   - { pattern: ".*bias|.*norm.*", weight_decay: 0.0 }
  #   - { pattern: "attn|mlp", min_ndim: 2, module_type: Linear, weight_decay: 0.1 }

# --- scheduler ---------------------------------------------------------------
scheduler:
  name: warmup_cosine     # constant | linear | warmup_cosine | wsd
  warmup_steps: 50
  total_steps: ${trainer.max_steps}
  min_lr_ratio: 0.1

# --- engine ------------------------------------------------------------------
engine:
  name: standard          # standard | layer_offload (large models on small GPUs)
  mixed_precision: bf16   # no | fp16 | bf16
  # update_rule: { name: standard }     # standard | sam | mezo | rl

# --- trainer -----------------------------------------------------------------
trainer:
  name: pretrain          # pretrain | preference | reward_model | ppo | grpo | <your own>
  max_steps: 200
  val_every: 0
  ckpt_every: 100
  log_every: 25
  grad_clip: 1.0
  accumulate: 1

# --- callbacks ---------------------------------------------------------------
callbacks:
  - { name: throughput, window: 25 }
  - { name: nan_skip, max_skips: 10 }
  - { name: best_ckpt, monitor: loss, mode: min }
  - { name: lineage_recorder }          # writes lineage.sqlite
  # - { name: ema, decay: 0.999 }
  # - { name: early_stop, monitor: loss, patience: 5 }

# --- logger ------------------------------------------------------------------
logger:
  - { name: console, log_every: 25 }
  - { name: jsonl }
  # - { name: tensorboard }

# --- invariants (failure-first; lab mode runs these) -------------------------
# invariants:
#   - { name: loss_finite, action: abort }
#   - { name: grad_norm_bounded, max: 1000, action: warn }


# =============================================================================
# OPTIONAL blocks — uncomment one to switch paradigm. See docs/training.md.
# =============================================================================

# --- A. SFT over chat data (PrepGraph) ---------------------------------------
# Replace data: with a prep_graph data module; see recipes/sft_chat.yaml.

# --- B. Model variants (model_profiles) --------------------------------------
# Swap on the CLI with `model=big`; tweak with `model_profiles.base.d_model=384`.
# model: base
# model_profiles:
#   base: { name: tiny_lm, d_model: 256, n_layers: 4, n_heads: 8, max_seq_len: 256 }
#   big:  { name: tiny_lm, d_model: 512, n_layers: 8, n_heads: 8, max_seq_len: 256 }

# --- C. Multi-model: frozen teacher + trainable student ----------------------
# A custom trainer reads self.models["teacher"] / self.optimizers["student"].
# See examples/online_distill.py and recipes/online_distill_demo.yaml.
# models:
#   student: { spec: { name: tiny_lm, d_model: 128, n_layers: 4, n_heads: 4 }, trainable: true,  optimizer: main }
#   teacher: { spec: { name: tiny_lm, d_model: 128, n_layers: 4, n_heads: 4 }, trainable: false, checkpoint: path/to/teacher }
# optimizers:
#   main: { name: adamw, lr: 1.0e-3 }

# --- D. Online RL (PPO / GRPO) ----------------------------------------------
# trainer: { name: ppo, rollout_steps: 32, ppo_epochs: 4, rollout_backend: hf_generate, temperature: 1.0, top_p: 1.0 }
# loss:    { name: ppo_surrogate, clip_eps: 0.2 }
# judge:   { name: verifier, verify_pattern: "\\\\d+" }   # judge -> reward_fn via reward_adapter

# --- E. Preference (DPO / IPO / SimPO / ORPO / KTO) --------------------------
# trainer: { name: preference, ref_namespace: ref }
# loss:    { name: dpo, beta: 0.1 }

# --- F. PEFT (LoRA / QLoRA) --------------------------------------------------
# pip install -e ".[peft]"  (QLoRA also needs ".[quant]", Linux+CUDA)
# model: lora
# model_profiles:
#   lora:
#     name: lora
#     base: { name: hf_causal, pretrained: meta-llama/Llama-3.2-1B }
#     r: 8
#     target_modules: [q_proj, v_proj]

# --- G. Distributed (no model/trainer changes needed) ------------------------
# Launch with: torchrun --nproc_per_node=N -m lighttrain.cli train -c cfg.yaml
# parallel:
#   backend: nccl          # gloo for CPU/CI
#   dp: 4
#   grad_sync: { name: ddp, find_unused_parameters: false }   # ddp | fsdp | deepspeed

# --- H. PrepGraph data pipeline ----------------------------------------------
# data: { name: prep_graph }
# prep_graph:
#   nodes:
#     - { name: load,     kind: load,     config: { source: "jsonl:data/chat.jsonl" } }
#     - { name: tok,      kind: tokenize, inputs: [load] }
#     - { name: packed,   kind: pack,     inputs: [tok], config: { strategy: best_fit, seq_len: 256 } }
'''

_INIT_README = """\
# lighttrain project

Generated by `lighttrain init`. `cfg.yaml` is a fully runnable recipe (tiny_lm +
byte tokenizer + warmup_cosine) **and** a guided tour: the active blocks train
immediately, and the commented `OPTIONAL` blocks (A–H) switch the recipe into
SFT / preference / online RL / distillation / PEFT / distributed / PrepGraph.

## Quickstart

1. Drop a corpus at `corpus.txt` (one example per line).
2. `lighttrain dry-run -c cfg.yaml` — validate the recipe without training.
3. `lighttrain train -c cfg.yaml ++trainer.max_steps=50` — 50-step smoke run.

Outputs land in `runs/<exp>/<ts>-<slug>-<hash>/` (config snapshot, `env.json`,
`logs/metrics.jsonl`, `checkpoints/`).

## Grow the recipe

Uncomment one `OPTIONAL` block in `cfg.yaml`:

| Block | Paradigm |
| ----- | -------- |
| B | model variants (`model_profiles`) |
| C | multi-model (frozen teacher + student) |
| D | online RL (PPO / GRPO) |
| E | preference (DPO / IPO / SimPO / ORPO / KTO) |
| F | PEFT (LoRA / QLoRA) |
| G | distributed (DDP / FSDP / ZeRO) |
| H | PrepGraph data pipeline |

## Docs

See the framework's `docs/` directory — start at `docs/getting-started.md`,
`docs/configuration.md`, and `docs/recipes.md`.
"""


def init_cmd(
    path: Path = typer.Argument(..., help="Target directory (created if absent)."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Generate a minimal recipe + run-dir skeleton."""
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not force:
        console.print(f"[red]target {path} is not empty (pass --force to overwrite)[/]")
        raise typer.Exit(code=1)
    path.mkdir(parents=True, exist_ok=True)
    (path / "cfg.yaml").write_text(_INIT_RECIPE, encoding="utf-8")
    (path / "README.md").write_text(_INIT_README, encoding="utf-8")
    (path / "runs").mkdir(exist_ok=True)
    (path / "artifacts").mkdir(exist_ok=True)

    table = Table(title="lighttrain init")
    table.add_column("file", style="cyan")
    table.add_column("status", style="green")
    table.add_row(str(path / "cfg.yaml"), "created")
    table.add_row(str(path / "README.md"), "created")
    table.add_row(str(path / "runs/"), "created")
    table.add_row(str(path / "artifacts/"), "created")
    console.print(table)
    console.print(f"[green]initialized lighttrain project at {path}[/]")
