"""Minimal DataModule.

Wires a dataset / collator / sampler into a torch DataLoader pair.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from lighttrain.config._resolver import resolve as _resolve
from lighttrain.registry import register
from .tokenizers import PAD_ID


@register("data_module", "simple")
class SimpleDataModule:
    """Single-dataset module reading from a plain dataset spec."""

    def __init__(
        self,
        *,
        dataset: Mapping[str, Any] | Any,
        tokenizer: Mapping[str, Any] | Any | None = None,
        collator: Mapping[str, Any] | Any | None = None,
        sampler: Mapping[str, Any] | Any | None = None,
        val_dataset: Mapping[str, Any] | Any | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)

        self.tokenizer = _maybe_resolve(tokenizer, "tokenizer")
        if self.tokenizer is None:
            from .tokenizers import ByteTokenizer

            self.tokenizer = ByteTokenizer()

        self.dataset = _resolve_dataset(dataset, tokenizer=self.tokenizer)
        self.val_dataset = (
            _resolve_dataset(val_dataset, tokenizer=self.tokenizer)
            if val_dataset is not None
            else None
        )

        self.collator = _maybe_resolve(
            collator, "collator", default_kwargs={"pad_id": getattr(self.tokenizer, "pad_id", PAD_ID)}
        )
        if self.collator is None:
            from .collators import CausalLMCollator

            self.collator = CausalLMCollator(pad_id=getattr(self.tokenizer, "pad_id", PAD_ID))

        self._sampler_spec = sampler
        self._train_sampler = _maybe_resolve_sampler(sampler, dataset=self.dataset)

    def train_loader(self) -> DataLoader:
        sampler = self._train_sampler
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False if sampler is not None else True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collator,
        )

    def val_loader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def predict_loader(self) -> DataLoader | None:
        return None

    def seek(self, epoch: int, consumed_batches: int) -> None:
        """Position the train sampler for mid-epoch resume (BUG-1).

        Translates the trainer's authoritative consumed-*batch* count into a
        consumed-*index* count (``× batch_size``) and delegates to the
        sampler's ``seek``. Prefetch-independent: the count comes from the
        training loop, not the sampler's yield position.
        """
        sampler = self._train_sampler
        if sampler is None or not hasattr(sampler, "seek"):
            return
        sampler.seek(int(epoch), int(consumed_batches) * self.batch_size)

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        if self._train_sampler is not None and hasattr(
            self._train_sampler, "state_dict"
        ):
            sd["sampler"] = self._train_sampler.state_dict()
        return sd

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if self._train_sampler is not None and "sampler" in sd and hasattr(
            self._train_sampler, "load_state_dict"
        ):
            self._train_sampler.load_state_dict(sd["sampler"])


def _maybe_resolve(
    spec: Mapping[str, Any] | Any | None,
    category: str,
    *,
    default_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        return spec
    spec = dict(spec)
    if default_kwargs:
        for k, v in default_kwargs.items():
            spec.setdefault(k, v)
    return _resolve(spec, category=category)


def _resolve_dataset(spec: Any, *, tokenizer: Any) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    spec = dict(spec)
    spec.setdefault("tokenizer", tokenizer)
    if "name" in spec or "_target_" in spec:
        return _resolve(spec, category="dataset")
    raise ValueError("Dataset spec needs `name` or `_target_`.")


def _maybe_resolve_sampler(
    spec: Mapping[str, Any] | Any | None, *, dataset: Any
) -> Any:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        spec = dict(spec)
        spec.setdefault("dataset", dataset)
        return _resolve(spec, category="sampler")
    return spec


__all__ = ["SimpleDataModule"]
