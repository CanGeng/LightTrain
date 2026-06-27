"""Shared training utilities — lighttrain-compatible port of MiniMind's trainer_utils.

Changes from original:
  - sys.path manipulation replaced by importlib-based model import
  - init_model uses path relative to this file for default tokenizer_path
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer

# Load model_minimind from the sibling model/ directory.
_mm_root = str(Path(__file__).resolve().parents[1])
if _mm_root not in sys.path:
    sys.path.insert(0, _mm_root)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str) -> None:
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed_mode() -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def lm_checkpoint(lm_config: MiniMindConfig, weight: str = "full_sft",
                  model=None, optimizer=None, epoch: int = 0, step: int = 0,
                  wandb=None, save_dir: str = "checkpoints", **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, "_orig_mod", raw)
        sd = {k: v.half().cpu() for k, v in raw.state_dict().items()}
        tmp = ckp_path + ".tmp"
        torch.save(sd, tmp)
        os.replace(tmp, ckp_path)
        wandb_id = None
        if wandb:
            run = getattr(wandb, "get_run", lambda: None)()
            wandb_id = getattr(run, "id", None) if run else getattr(wandb, "id", None)
        resume_data = {"model": sd, "optimizer": optimizer.state_dict(), "epoch": epoch,
                       "step": step,
                       "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                       "wandb_id": wandb_id}
        for k, v in kwargs.items():
            if v is not None:
                r = v.module if isinstance(v, DistributedDataParallel) else v
                r = getattr(r, "_orig_mod", r)
                resume_data[k] = r.state_dict() if hasattr(r, "state_dict") else v
        tmp = resume_path + ".tmp"
        torch.save(resume_data, tmp)
        os.replace(tmp, resume_path)
        del sd, resume_data
        torch.cuda.empty_cache()
    else:
        if os.path.exists(resume_path):
            ckp = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp.get("world_size", 1)
            cur_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != cur_ws:
                ckp["step"] = ckp["step"] * saved_ws // cur_ws
                Logger(f"GPU数量变化({saved_ws}→{cur_ws})，step已自动转换为{ckp['step']}")
            return ckp
        return None


def init_model(lm_config: MiniMindConfig, from_weight: str = "none",
               tokenizer_path: str | None = None, save_dir: str = "out",
               device: str = "cuda"):
    if tokenizer_path is None:
        tokenizer_path = str(Path(__file__).resolve().parents[1] / "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config).to(device)
    if from_weight != "none":
        moe = "_moe" if lm_config.use_moe else ""
        ckp = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe}.pth"
        if os.path.exists(ckp):
            sd = torch.load(ckp, map_location=device)
            model.load_state_dict(sd, strict=False)
            Logger(f"Loaded weights from {ckp}")
    return model, tokenizer


class LMForRewardModel(torch.nn.Module):
    """Sequence-level scalar reward head on top of a causal LM."""
    def __init__(self, model: MiniMindForCausalLM) -> None:
        super().__init__()
        self.model = model
        self.v_head = torch.nn.Linear(model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden, *_ = self.model.model(input_ids, attention_mask=attention_mask)
        last = hidden[:, -1, :]
        return self.v_head(last).squeeze(-1)


class SkipBatchSampler(Sampler):
    """Wrap a sampler and skip the first ``skip`` batches (for resume)."""
    def __init__(self, source, batch_size: int, skip: int = 0) -> None:
        self.source = source
        self.batch_size = int(batch_size)
        self.skip = int(skip)

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.source:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip:
                    skipped += 1
                else:
                    yield batch
                batch = []
        if batch and skipped >= self.skip:
            yield batch

    def __len__(self) -> int:
        n = len(self.source) // self.batch_size
        return max(0, n - self.skip)
