"""Demo-only lab components for the alt-objective / local-learning recipes.

DEMO ONLY. These tiny synthetic datasets / collators / model are registered via
a recipe's ``user_modules:`` so the R8–R11 demo recipes (``diffusion_eps``,
``jepa``, ``ff_demo``, ``pcn_demo``, ``mezo_sft``) run end-to-end without any
external data. They are **not** core capabilities and are **not** guaranteed to
exist once the package is installed — do not depend on them outside the bundled
recipes.

Registered components:
    model     mlp_toy           — plain MLP for Forward-Forward / PCN
    dataset   synthetic_binary  + collator x_labels   → batch["x"], one-hot labels
    dataset   synthetic_signal  + collator signal     → batch["x"] (B, C, L)  (diffusion)
    dataset   synthetic_patches + collator patches    → batch["patches"] (B, N, D)  (JEPA)
    dataset   sft_jsonl                                → causal-LM samples from JSONL

Note on signatures: each component explicitly accepts the kwargs the data module
injects (``tokenizer`` into datasets, ``pad_id`` into collators) even when unused,
so resolution doesn't drop-and-warn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain import register
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Toy model — Forward-Forward / PCN (a plain stack of nn.Linear layers)
# ---------------------------------------------------------------------------

@register("model", "mlp_toy")
class MLPToy(nn.Module):
    """``num_layers`` Linear layers: input_dim → hidden_dim … → output_dim.

    ForwardForwardUpdateRule reads ``model.layers``; PCNUpdateRule collects the
    ``nn.Linear`` modules — neither calls ``forward``, but it's provided for
    completeness (chained with ReLU, returns ``{"logits": ...}``).
    """

    def __init__(
        self,
        *,
        input_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(hidden_dim)] * (int(num_layers) - 1) + [int(output_dim)]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, **batch: Any) -> ModelOutput:
        h = batch.get("x", batch.get("input_ids"))
        h = h.float()  # type: ignore[union-attr]
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < last:
                h = torch.relu(h)
        return ModelOutput(outputs={"logits": h})


# ---------------------------------------------------------------------------
# Synthetic binary classification — Forward-Forward / PCN
# ---------------------------------------------------------------------------

@register("dataset", "synthetic_binary")
class SyntheticBinaryDataset:
    """Two-class linearly-separable Gaussian features."""

    def __init__(
        self,
        *,
        input_dim: int = 16,
        num_samples: int = 256,
        seed: int = 0,
        tokenizer: Any = None,  # injected by the data module; unused
    ) -> None:
        g = torch.Generator().manual_seed(int(seed))
        self.x = torch.randn(int(num_samples), int(input_dim), generator=g)
        w = torch.randn(int(input_dim), generator=g)
        self.y = (self.x @ w > 0).long()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"x": self.x[idx], "label": int(self.y[idx])}


@register("collator", "x_labels")
class XLabelsCollator:
    """Stack features → ``x`` (B, D); one-hot targets → ``labels`` (B, C).

    The one-hot shape matters: PCNUpdateRule only clamps the supervised top layer
    when ``labels.shape == top_activation.shape`` (i.e. (B, output_dim)).
    """

    def __init__(
        self,
        *,
        input_dim: int = 16,  # noqa: ARG002  (kept for recipe symmetry)
        num_classes: int = 2,
        pad_id: int | None = None,  # injected by the data module; unused
    ) -> None:
        self.num_classes = int(num_classes)

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        x = torch.stack([torch.as_tensor(s["x"], dtype=torch.float) for s in samples])
        y = torch.tensor([int(s["label"]) for s in samples], dtype=torch.long)
        labels = F.one_hot(y, num_classes=self.num_classes).float()
        return {"x": x, "labels": labels}


# ---------------------------------------------------------------------------
# Synthetic 1-D signals — diffusion (DiffusionObjective reads batch["x"])
# ---------------------------------------------------------------------------

@register("dataset", "synthetic_signal")
class SyntheticSignalDataset:
    """Random 1-D signals generated on the fly (no real data)."""

    def __init__(
        self,
        *,
        signal_len: int = 32,
        num_samples: int = 512,
        seed: int = 0,
        tokenizer: Any = None,  # injected; unused
    ) -> None:
        g = torch.Generator().manual_seed(int(seed))
        self.sig = torch.randn(int(num_samples), int(signal_len), generator=g)

    def __len__(self) -> int:
        return self.sig.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"x": self.sig[idx]}


@register("collator", "signal")
class SignalCollator:
    """Stack signals into ``x`` of shape (B, channels, signal_len)."""

    def __init__(
        self,
        *,
        signal_len: int = 32,  # noqa: ARG002
        channels: int = 1,
        pad_id: int | None = None,  # injected; unused
    ) -> None:
        self.channels = int(channels)

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        x = torch.stack([torch.as_tensor(s["x"], dtype=torch.float) for s in samples])
        return {"x": x.view(x.shape[0], self.channels, -1)}


# ---------------------------------------------------------------------------
# Synthetic patch sequences — JEPA (JEPAObjective reads batch["patches"])
# ---------------------------------------------------------------------------

@register("dataset", "synthetic_patches")
class SyntheticPatchesDataset:
    """Random patch sequences of shape (num_patches, patch_dim)."""

    def __init__(
        self,
        *,
        num_patches: int = 16,
        patch_dim: int = 16,
        num_samples: int = 256,
        seed: int = 0,
        tokenizer: Any = None,  # injected; unused
    ) -> None:
        g = torch.Generator().manual_seed(int(seed))
        self.p = torch.randn(int(num_samples), int(num_patches), int(patch_dim), generator=g)

    def __len__(self) -> int:
        return self.p.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"patches": self.p[idx]}


@register("collator", "patches")
class PatchesCollator:
    """Stack patch grids into ``patches`` of shape (B, N, D)."""

    def __init__(
        self,
        *,
        num_patches: int = 16,  # noqa: ARG002
        patch_dim: int = 16,  # noqa: ARG002
        pad_id: int | None = None,  # injected; unused
    ) -> None:
        pass

    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        patches = torch.stack(
            [torch.as_tensor(s["patches"], dtype=torch.float) for s in samples]
        )
        return {"patches": patches}


# ---------------------------------------------------------------------------
# SFT JSONL — MeZO (causal-LM samples with prompt tokens masked from the loss)
# ---------------------------------------------------------------------------

@register("dataset", "sft_jsonl")
class SFTJsonlDataset:
    """Read ``{"prompt", "completion"}`` JSONL → tokenized causal-LM samples.

    ``input_ids`` = encode(prompt) + encode(completion); ``labels`` mirror them
    with the prompt positions set to ``-100`` (loss on the completion only).
    """

    def __init__(
        self,
        *,
        path: str | Path,
        tokenizer: Any,
        max_len: int = 128,
        encoding: str = "utf-8",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.max_len = int(max_len)
        self.samples: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding=encoding, errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            p_ids = list(tokenizer.encode(str(obj["prompt"])))
            c_ids = list(tokenizer.encode(str(obj["completion"])))
            ids = (p_ids + c_ids)[: self.max_len]
            labels = ([-100] * len(p_ids) + c_ids)[: self.max_len]
            if not ids:
                continue
            self.samples.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels,
            })
        if not self.samples:
            raise ValueError(f"No usable lines in {self.path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[int(idx)]
