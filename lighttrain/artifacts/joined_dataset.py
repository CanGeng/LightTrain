"""ArtifactJoinedDataset.

Wraps any base dataset (map-style or list) and, on ``__getitem__``, fetches
auxiliary tensors from one or more artifact stores keyed by ``sample.id``.
Joined tensors land under ``aux.<namespace>.<tensor_name>`` so they never
collide with the base sample's keys.

Missing-sample policy:

  * ``require`` (default) — raise :class:`KeyError`.
  * ``drop`` — return ``None`` from ``__getitem__``. The companion
    :func:`drop_none_collator` filters before the collator runs.
  * ``fill_zero`` — synthesize zero tensors using ``header.field_schema``.

Re-loading on a new artifact version is a deferred hook — the slots exist but
:class:`ArtifactJoinedDataset` only re-opens stores when the user explicitly
calls :meth:`reload`. Event-driven reload is wired by
:class:`DynamicArtifactCallback`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ..data.core._schema import derive_sample_id
from ..registry import register
from .store import (
    ArtifactHeader,
    ArtifactStoreProtocol,
    StaleArtifactError,
    open_artifact_store,
)


_MISSING_REQUIRE = "require"
_MISSING_DROP = "drop"
_MISSING_FILL = "fill_zero"


@register("dataset", "artifact_joined")
class ArtifactJoinedDataset:
    """Map-style dataset that lazy-joins artifact tensors per sample.

    Parameters
    ----------
    base : dataset | list | mapping spec
        Underlying dataset. Anything supporting ``__getitem__`` / ``__len__``
        works (or pass a ``{name | _target_: ...}`` mapping that resolves to
        such a dataset via the registry).
    join : list of mapping
        Each entry: ``{store: path, namespace: str, version?: str,
        missing?: str, expected_header?: dict, allow_stale?: bool}``.
        Stores are opened at construction; ``open_artifact_store`` validates
        the header.
    missing : str
        Default policy when a join entry does not specify its own.
    sample_id_key : str
        Override the key from which to read the sample id. Defaults to ``id``;
        falls back to :func:`derive_sample_id`.
    """

    def __init__(
        self,
        base: Any,
        *,
        join: list[Mapping[str, Any]] | None = None,
        missing: str = _MISSING_REQUIRE,
        sample_id_key: str = "id",
        allow_stale_artifact: bool = False,
        tokenizer: Any = None,  # injected by SimpleDataModule; forwarded to base
    ) -> None:
        self.base = _resolve_base(base, tokenizer=tokenizer)
        self.sample_id_key = str(sample_id_key)
        self.default_missing = str(missing)
        self.allow_stale_artifact = bool(allow_stale_artifact)
        self._join_specs: list[dict[str, Any]] = [dict(s) for s in (join or [])]
        self._stores: list[tuple[dict[str, Any], ArtifactStoreProtocol]] = []
        self._open_stores()

    def _open_stores(self) -> None:
        self._stores.clear()
        for spec in self._join_specs:
            cfg = dict(spec)
            root = cfg.pop("store", cfg.pop("path", None))
            if not root:
                raise ValueError("each join entry needs `store` (path to artifact root)")
            allow = bool(cfg.get("allow_stale_artifact", self.allow_stale_artifact))
            expected = cfg.get("expected_header")
            if isinstance(expected, Mapping):
                expected = ArtifactHeader.from_dict(expected)
            store = open_artifact_store(root, expected_header=expected, allow_stale=allow)
            cfg.setdefault("namespace", _default_namespace(Path(root).name))
            cfg.setdefault("missing", self.default_missing)
            self._stores.append((cfg, store))

    def reload(self) -> None:
        """Re-open all stores from disk. Used by event-driven artifact swap."""
        self._open_stores()

    def on_artifact_new_version(
        self, *, path: str | None = None, step: int | None = None, **_: Any
    ) -> None:
        """EventBus callback: reload stores when a new artifact version is available.

        Wired by :class:`DynamicArtifactCallback` via ``bus.dispatch``.
        Checks whether the new artifact path overlaps with any join spec before
        reloading; unknown paths are ignored.
        """
        self._open_stores()

    def __len__(self) -> int:
        if hasattr(self.base, "__len__"):
            return len(self.base)  # type: ignore[arg-type]
        raise TypeError("base dataset has no __len__")

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = self.base[idx]
        if not isinstance(sample, Mapping):
            raise TypeError(f"base dataset returned {type(sample).__name__}, want Mapping")
        merged: dict[str, Any] = dict(sample)
        sid = str(sample.get(self.sample_id_key) or derive_sample_id(sample))
        merged.setdefault(self.sample_id_key, sid)
        for cfg, store in self._stores:
            namespace = cfg["namespace"]
            missing_policy = cfg["missing"]
            if not store.contains(sid):
                if missing_policy == _MISSING_DROP:
                    return None
                if missing_policy == _MISSING_FILL:
                    for k, shape_str in store.header.field_schema.items():
                        shape = _parse_shape(shape_str)
                        merged[f"aux.{namespace}.{k}"] = torch.zeros(shape)
                    continue
                raise KeyError(
                    f"artifact_joined: sample {sid!r} not present in store at "
                    f"{store.root}. Set missing='drop' to skip or 'fill_zero' "
                    f"to substitute zeros."
                )
            tensors = store.get(sid)
            for k, v in tensors.items():
                merged[f"aux.{namespace}.{k}"] = v
        return merged

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self)):
            row = self[i]
            if row is not None:
                yield row


def _resolve_base(base: Any, tokenizer: Any = None) -> Any:
    if isinstance(base, Mapping):
        from ..config._resolver import resolve as _resolve
        spec = dict(base)
        if tokenizer is not None:
            spec.setdefault("tokenizer", tokenizer)
        return _resolve(spec, category="dataset")
    return base


def _default_namespace(name: str) -> str:
    return name.split("_")[0] or "aux"


def _parse_shape(shape_str: str) -> tuple[int, ...]:
    try:
        parsed = ast.literal_eval(shape_str)
        if isinstance(parsed, tuple):
            return parsed
        if isinstance(parsed, list):
            return tuple(int(x) for x in parsed)
        if isinstance(parsed, int):
            return (parsed,)
    except Exception:
        pass
    return ()


def drop_none_collator(collator: Any) -> Any:
    """Wrap an existing collator so it transparently filters out ``None``
    rows produced by ``missing='drop'``."""

    def _wrapped(samples: list[Any]) -> Any:
        kept = [s for s in samples if s is not None]
        if not kept:
            raise RuntimeError("drop_none_collator: entire batch was dropped")
        return collator(kept)

    return _wrapped


__all__ = [
    "ArtifactJoinedDataset",
    "drop_none_collator",
    "StaleArtifactError",
]
