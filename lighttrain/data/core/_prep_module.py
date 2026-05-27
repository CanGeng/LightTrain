"""PrepGraph-backed DataModule.

Builds the graph from ``cfg.prep_graph`` (handed in via the runtime), runs
or recovers it via :class:`PrepRunner`, and mounts terminal nodes as
training / validation datasets.

Public surface mirrors :class:`SimpleDataModule`:
``train_loader`` / ``val_loader`` / ``state_dict`` / ``load_state_dict``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch.utils.data import DataLoader

from ...config._resolver import resolve as _resolve
from ...prepgraph.dag import PrepGraph
from ...prepgraph.runner import PrepRunner
from ...registry import register
from .collators import CausalLMCollator
from .tokenizers import PAD_ID


def _normalize_terminal(spec: Any) -> str | None:
    """Accept ``"prep_graph:<terminal>"`` or ``{prep_graph: <terminal>}`` strings."""
    if spec is None:
        return None
    if isinstance(spec, str) and spec.startswith("prep_graph:"):
        return spec[len("prep_graph:") :]
    if isinstance(spec, Mapping) and "prep_graph" in spec:
        return str(spec["prep_graph"])
    return None


@register("data_module", "prep_graph")
class PrepGraphDataModule:
    """DataModule that consumes terminal nodes from a PrepGraph run.

    Parameters
    ----------
    prep_graph : Mapping
        The ``prep_graph:`` block from the recipe (``{nodes, terminals}``).
    train : str
        Terminal node name to use as the training dataset.
    val : str | None
        Terminal node name to use as the validation dataset (optional).
    store_root : str | Path
        Cache directory for materialized PrepGraph outputs.
    tokenizer / collator / sampler / batch_size / ...
        Same as :class:`SimpleDataModule`.
    workers : int
        Parallelism for the runner (1 by default).
    """

    def __init__(
        self,
        *,
        prep_graph: Mapping[str, Any],
        train: str,
        val: str | None = None,
        store_root: str | Path = "./runs/prep",
        tokenizer: Mapping[str, Any] | Any | None = None,
        collator: Mapping[str, Any] | Any | None = None,
        sampler: Mapping[str, Any] | Any | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        workers: int = 1,
        console: Any | None = None,
        run_on_init: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.drop_last = bool(drop_last)
        self.train_terminal = str(train)
        self.val_terminal = str(val) if val else None

        # ----- tokenizer ----------------------------------------------------
        self.tokenizer = _maybe_resolve(tokenizer, "tokenizer")
        if self.tokenizer is None:
            from .tokenizers import ByteTokenizer

            self.tokenizer = ByteTokenizer()
        pad_id = getattr(self.tokenizer, "pad_id", PAD_ID)

        # ----- graph + runner ----------------------------------------------
        self.graph = PrepGraph.from_config(prep_graph)
        self.runner = PrepRunner(
            self.graph,
            store_root=Path(store_root),
            workers=workers,
            console=console,
        )
        self._results: dict[str, Any] = {}
        if run_on_init:
            self._results = self.runner.run()
        self._validate_terminals()

        # ----- datasets -----------------------------------------------------
        self.dataset = self._dataset_for(self.train_terminal)
        self.val_dataset = (
            self._dataset_for(self.val_terminal) if self.val_terminal else None
        )

        # ----- collator -----------------------------------------------------
        self.collator = _maybe_resolve(
            collator,
            "collator",
            default_kwargs={"pad_id": pad_id},
        )
        if self.collator is None:
            self.collator = CausalLMCollator(pad_id=pad_id)

        # ----- sampler ------------------------------------------------------
        self._train_sampler = _maybe_resolve_sampler(sampler, dataset=self.dataset)

    # ----- helpers ---------------------------------------------------------

    def _validate_terminals(self) -> None:
        if self.train_terminal not in self.graph.nodes:
            raise ValueError(
                f"PrepGraphDataModule: train terminal {self.train_terminal!r} "
                "not found in graph."
            )
        if self.val_terminal and self.val_terminal not in self.graph.nodes:
            raise ValueError(
                f"PrepGraphDataModule: val terminal {self.val_terminal!r} "
                "not found in graph."
            )

    def _dataset_for(self, terminal: str) -> Any:
        result = self._results.get(terminal)
        if result is None:
            raise RuntimeError(
                f"PrepGraphDataModule: terminal {terminal!r} has no result. "
                "Did the runner execute?"
            )
        # Prefer the on-disk view: the runner commits staging → final, leaving
        # any in-staging store handle's path stale. Rebuild from final_dir.
        if result.final_dir is not None and result.final_dir.exists():
            from ...prepgraph.nodes.materialize import _RowsDataset
            from ..cache._memmap import MemmapDataset, read_header

            if read_header(result.final_dir) is not None:
                return MemmapDataset(result.final_dir)
            return _RowsDataset(result.final_dir)
        if result.store is not None:
            return result.store
        if result.rows is not None:
            return list(result.rows)
        raise RuntimeError(
            f"PrepGraphDataModule: terminal {terminal!r} produced neither rows "
            "nor a store (did you forget a `materialize` node?)."
        )

    # ----- DataLoader surface ----------------------------------------------

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

    def state_dict(self) -> dict[str, Any]:
        sd: dict[str, Any] = {}
        if self._train_sampler is not None and hasattr(
            self._train_sampler, "state_dict"
        ):
            sd["sampler"] = self._train_sampler.state_dict()
        return sd

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        if (
            self._train_sampler is not None
            and "sampler" in sd
            and hasattr(self._train_sampler, "load_state_dict")
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


__all__ = ["PrepGraphDataModule", "_normalize_terminal"]
