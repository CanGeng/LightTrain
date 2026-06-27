# nanoGPT — lighttrain port

Faithful port of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
using lighttrain as the training framework.

## What's here

```
examples/nanoGPT/
  model.py          # NanoGPT model (@register "nano_gpt") — identical arch to original
  components.py     # BinaryMemmapDataset + StackCollator (@register via user_modules)
  sample.py         # Text generation (from checkpoint or pretrained GPT-2)
  data/
    shakespeare_char/prepare.py   # char-level tokenization → train.bin / val.bin
    shakespeare/prepare.py        # GPT-2 BPE tokenization → train.bin / val.bin
    openwebtext/prepare.py        # full OpenWebText → train.bin / val.bin (~54 GB)
  configs/
    train_shakespeare_char.yaml   # 6-layer char-level (10.65M), ~5 min on GPU
    train_gpt2.yaml               # GPT-2 124M on OpenWebText, ~4 days A100
    finetune_shakespeare.yaml     # fine-tune pretrained GPT-2 on Shakespeare
```

## Quickstart — Shakespeare char-level

```bash
# 1. Prepare data (downloads ~1 MB)
python examples/nanoGPT/data/shakespeare_char/prepare.py

# 2. Train (val loss ~1.47 after 5000 steps)
lighttrain train -c examples/nanoGPT/configs/train_shakespeare_char.yaml

# 3. Sample from the trained model
python examples/nanoGPT/sample.py \
    --from_ckpt runs/nanogpt_shakespeare_char/checkpoints/best \
    --meta_path examples/nanoGPT/data/shakespeare_char/meta.pkl \
    --num_samples 3 --max_new_tokens 200
```

## Configs

| Config | Dataset | Params | Notes |
|--------|---------|--------|-------|
| `train_shakespeare_char.yaml` | Shakespeare (char) | 10.65M | Trains in minutes |
| `train_gpt2.yaml` | OpenWebText | 124M | ~4 days on A100 |
| `finetune_shakespeare.yaml` | Shakespeare (BPE) | 124M | Fine-tunes `gpt2` weights |

## GPT-2 fine-tuning

```bash
python examples/nanoGPT/data/shakespeare/prepare.py
lighttrain train -c examples/nanoGPT/configs/finetune_shakespeare.yaml
```

## Generate from pretrained GPT-2

```bash
python examples/nanoGPT/sample.py --from_pretrained gpt2 \
    --prompt "To be or not to be" --num_samples 3 --max_new_tokens 200
```

## Architecture

`NanoGPT` is identical to the original: pre-norm GPT-2 transformer with
optional-bias LayerNorm, fused QKV projection (`c_attn` naming for weight
compatibility with OpenAI checkpoints), Flash Attention (PyTorch ≥ 2.0),
weight tying, and the GPT-2 Conv1D→Linear weight transposition for
`from_pretrained`. Registers as `nano_gpt` in lighttrain's model registry.

## Differences from the original

- Training is driven by `lighttrain train -c ...` (YAML config) instead of `train.py`
- DDP: use `torchrun --nproc_per_node=N -m lighttrain.cli train -c ...`
- Checkpoints are saved in lighttrain's capsule format (`model.safetensors`)
- `sample.py` reads both lighttrain capsules and the original `.pt` format
