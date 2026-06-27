"""Generate text from a trained nanoGPT checkpoint or a pretrained GPT-2 model.

Equivalent to nanoGPT's sample.py — run from the project root:

    # From a lighttrain checkpoint:
    python examples/nanoGPT/sample.py --from_ckpt runs/nanogpt_shakespeare_char/checkpoints/best

    # From a pretrained GPT-2:
    python examples/nanoGPT/sample.py --from_pretrained gpt2 --prompt "To be or not"

    # Character-level (custom stoi/itos from prepare.py meta.pkl):
    python examples/nanoGPT/sample.py --from_ckpt runs/nanogpt_shakespeare_char/checkpoints/best \\
        --meta_path examples/nanoGPT/data/shakespeare_char/meta.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

# Allow running from repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanoGPT.model import NanoGPT


def _load_from_ckpt(ckpt_path: str, device: str) -> tuple[NanoGPT, dict]:
    """Load model from a lighttrain checkpoint capsule (directory or .pt file)."""
    p = Path(ckpt_path)
    # Lighttrain checkpoint capsule: directory with model.safetensors or model.pt
    if p.is_dir():
        for cand in ("model.safetensors", "model.pt"):
            if (p / cand).exists():
                weight_file = p / cand
                break
        else:
            raise FileNotFoundError(f"No model.safetensors or model.pt in {p}")
        # Try to read config from the capsule's metadata
        config_file = p / "config.json"
        model_args: dict = {}
        if config_file.exists():
            import json
            model_args = json.loads(config_file.read_text())
    else:
        weight_file = p
        model_args = {}

    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(weight_file))
    else:
        obj = torch.load(str(weight_file), map_location=device, weights_only=False)
        sd = obj.get("model", obj) if isinstance(obj, dict) else obj

    # Strip compile prefix
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    # Infer arch from state-dict shapes
    if not model_args:
        wpe = sd.get("transformer.wpe.weight", sd.get("wpe.weight"))
        wte = sd.get("transformer.wte.weight", sd.get("wte.weight"))
        n_layer = sum(1 for k in sd if k.endswith(".attn.c_attn.weight"))
        model_args = dict(
            block_size=wpe.shape[0] if wpe is not None else 1024,
            vocab_size=wte.shape[0] if wte is not None else 50304,
            n_layer=n_layer or 12,
            n_head=int(wte.shape[1] ** 0.5) if wte is not None else 12,  # approx
            n_embd=wte.shape[1] if wte is not None else 768,
        )

    model = NanoGPT(**model_args)
    model.load_state_dict(sd, strict=False)
    return model, model_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a nanoGPT model")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_ckpt", help="Path to a lighttrain checkpoint directory or .pt file")
    src.add_argument("--from_pretrained", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                     help="Initialize from OpenAI GPT-2 weights")
    parser.add_argument("--prompt", default="\n", help="Prompt string (or 'FILE:path.txt')")
    parser.add_argument("--meta_path", help="Path to meta.pkl for char-level encode/decode")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if args.device == "cpu" else torch.amp.autocast(
        device_type="cuda", dtype=ptdtype)

    # ---- Load model ----
    if args.from_pretrained:
        model = NanoGPT(pretrained=args.from_pretrained)
    else:
        model, _ = _load_from_ckpt(args.from_ckpt, args.device)

    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    # ---- Encoder/decoder ----
    if args.meta_path and os.path.exists(args.meta_path):
        with open(args.meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]                 # noqa: E731
        decode = lambda ids: "".join(itos[i] for i in ids)      # noqa: E731
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})  # noqa: E731
        decode = lambda ids: enc.decode(ids)                                 # noqa: E731

    # ---- Prompt ----
    prompt = args.prompt
    if prompt.startswith("FILE:"):
        prompt = Path(prompt[5:]).read_text(encoding="utf-8")
    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    # ---- Generate ----
    with torch.no_grad(), ctx:
        for _ in range(args.num_samples):
            y = model.generate(x, args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k)
            print(decode(y[0].tolist()))
            print("---------------")


if __name__ == "__main__":
    main()
