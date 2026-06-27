"""``--remap-embed`` path: slice ``embed_tokens`` / ``lm_head`` weights and
rewrite ``config.json`` / ``generation_config.json`` to match the pruned
vocab.

Ports ``voca-prune/model_save.py:replace_embed_and_lm_heads`` and the config
update block of ``voca-prune/main.py``. Uses ``safetensors`` for in-place
shard-by-shard rewrites (no model load into memory — saving peak RAM) and
``transformers.AutoConfig`` for the config.json rewrite.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file


def remap_embed_and_lm_heads(
    model_dir: Path,
    new_model_dir: Path,
    mapping_new2old: list[int],
) -> None:
    """Slice ``embed_tokens.weight`` / ``lm_head.weight`` from every
    ``.safetensors`` shard into the pruned vocab; copy other tensors and the
    ``model.safetensors.index.json`` verbatim.
    """
    new_model_dir.mkdir(parents=True, exist_ok=True)

    index_filename = "model.safetensors.index.json"
    src_index = model_dir / index_filename
    if src_index.exists():
        shutil.copy2(src_index, new_model_dir / index_filename)

    index = torch.tensor(mapping_new2old, dtype=torch.long)
    for f in os.listdir(model_dir):
        if not f.endswith(".safetensors"):
            continue
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_dir / f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                if "embed_tokens.weight" in k or "lm_head.weight" in k:
                    t = t[index].clone()
                tensors[k] = t
        save_file(tensors, str(new_model_dir / f))


def remap_config_and_generation(
    old_model_dir: Path,
    new_model_dir: Path,
    mapping_new2old: list[int],
) -> None:
    """Write ``config.json`` (vocab_size + every ``*_token_id`` remapped)
    and ``generation_config.json`` (same, supports ``eos_token_id`` as int
    or list) into ``new_model_dir``.

    A ``*_token_id`` whose old id is **not** in the pruned vocab is a
    pruned special token — the field is set to ``None`` and a warning is
    emitted (the user almost certainly wanted that special retained; add
    it to the corpus and re-prune).
    """
    from transformers import AutoConfig

    old2new: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(mapping_new2old)}

    new_config = AutoConfig.from_pretrained(str(old_model_dir), trust_remote_code=True)
    new_config.vocab_size = len(mapping_new2old)
    for key, old_id in list(new_config.to_dict().items()):
        if "token_id" not in key or not isinstance(old_id, int):
            continue
        if old_id in old2new:
            setattr(new_config, key, old2new[old_id])
        else:
            import warnings

            warnings.warn(
                f"config key '{key}' (token_id={old_id}) was pruned; set to None",
                stacklevel=2,
            )
            setattr(new_config, key, None)
    new_config.save_pretrained(str(new_model_dir))

    gen_path = old_model_dir / "generation_config.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        for key, v in list(gen.items()):
            if "token_id" not in key:
                continue
            if isinstance(v, int):
                if v in old2new:
                    gen[key] = old2new[v]
            elif isinstance(v, list):
                gen[key] = [old2new[i] for i in v if i in old2new]
        (new_model_dir / "generation_config.json").write_text(
            json.dumps(gen, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = ["remap_config_and_generation", "remap_embed_and_lm_heads"]
