"""ArtifactProducer + ModelForwardProducer.

``ModelForwardProducer`` runs a frozen model in ``eval`` / ``no_grad`` and
forwards each sample, capturing whatever the user declared via
:class:`ExtraOutputSpec`. Outputs land in an :class:`ArtifactStoreProtocol`
keyed by ``sample.id`` (or :func:`derive_sample_id`).

Other producers (off-the-shelf classifiers, external services, statistics
collectors) register against the ``artifact_producer`` category; this module
ships only the model-forward variant since that's the one R3 needs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import torch

from lighttrain.artifacts import ArtifactStoreProtocol
from lighttrain.data.core._schema import derive_sample_id
from lighttrain.models.extras import (
    ExtraOutputSpec,
    ExtrasHookManager,
    flatten_model_output_tensors,
)
from lighttrain.protocols import ModelOutput
from lighttrain.registry import register

from .store import SafetensorsShardStore, open_artifact_store

_log = logging.getLogger(__name__)


class ArtifactProducerProtocol(Protocol):
    def prepare(self, cfg: Mapping[str, Any] | None = None) -> None: ...
    def produce(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]: ...
    def finalize(self) -> Path: ...


def _as_batch(sample: Mapping[str, Any], device: torch.device | None = None) -> dict[str, Any]:
    """Turn a single sample into a batch-of-1 (everything moves to ``device``)."""
    batch: dict[str, Any] = {}
    for key in ("input_ids", "attention_mask", "labels"):
        if key not in sample:
            continue
        v = sample[key]
        if isinstance(v, torch.Tensor):
            t = v if v.dim() > 0 else v.view(1)
            batch[key] = t.unsqueeze(0) if t.dim() == 1 else t
        elif isinstance(v, list):
            batch[key] = torch.tensor([v], dtype=torch.long)
    if "modality_inputs" in sample:
        batch["modality_inputs"] = sample["modality_inputs"]
    if device is not None:
        batch = _to_device(batch, device)
    return batch


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(_to_device(v, device) for v in obj)
    return obj


def _resolve_store(
    store: Any,
    *,
    allow_stale: bool = False,
) -> ArtifactStoreProtocol:
    """Accept a path, a {root: ..., backend: ...} dict, or an already-built store."""
    if isinstance(store, Mapping):
        cfg = dict(store)
        root = cfg.pop("root", cfg.pop("path", None))
        backend = cfg.pop("name", cfg.pop("backend", "safetensors-shards"))
        if root is None:
            raise ValueError("artifact store config needs `root` or `path`.")
        from lighttrain.registry import get as _get
        cls = _get("artifact_store", backend)
        return cls(root, **cfg)
    if isinstance(store, (str, Path)):
        try:
            return open_artifact_store(store, allow_stale=allow_stale)
        except Exception:  # noqa: BLE001
            # Not yet finalized — produce mode opens fresh shard store.
            _log.warning(
                "artifacts: could not open existing store at %s; creating a fresh shard store",
                store,
                exc_info=True,
            )
            return SafetensorsShardStore(store)
    return store  # already an ArtifactStore instance


@register("artifact_producer", "model_forward")
class ModelForwardProducer:
    """Run a model in eval/no_grad and capture named tensors per sample.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate. Will be ``.eval()``-ed in :meth:`prepare`.
    store : ArtifactStore | dict | str | Path
        Destination. Dict / str / Path is auto-resolved via the registry.
    extras : list[ExtraOutputSpec]
        Hook-driven captures. Each spec's ``name`` becomes a
        key in ``ModelOutput.extras`` and on disk.
    collect_outputs : list[str] | None
        Names from ``ModelOutput.outputs`` to also persist (e.g. ``["logits"]``).
        ``None`` means **all** keys in ``model_output.outputs``.
    collect_hidden_states : bool
        Stack ``ModelOutput.hidden_states`` into a single ``hidden_states_layers``
        tensor and persist it.
    collect_attentions : bool
        Same for ``attentions`` → ``attentions_layers``.
    batch_keys : list[str]
        Sample keys to forward as model arguments.
    header_overrides : dict | None
        Manual header fields (model_id, data_version, etc.).
    forward_kwargs : dict | None
        Extra kwargs passed to ``model.forward(...)``. R3 uses this to set
        ``output_hidden_states=True`` on the HF / tiny LM adapters.
    """

    def __init__(
        self,
        *,
        model: Any,
        store: Any,
        extras: list[ExtraOutputSpec | Mapping[str, Any]] | None = None,
        collect_outputs: list[str] | None = None,
        collect_hidden_states: bool = False,
        collect_attentions: bool = False,
        batch_keys: list[str] | None = None,
        header_overrides: Mapping[str, Any] | None = None,
        forward_kwargs: Mapping[str, Any] | None = None,
        allow_stale_artifact: bool = False,
        artifact_name: str | None = None,
        artifact_version: str | None = None,
        producer_signature: str | None = None,
    ) -> None:
        self.model = model
        self.store: ArtifactStoreProtocol = _resolve_store(store, allow_stale=allow_stale_artifact)
        self.extras = [
            spec if isinstance(spec, ExtraOutputSpec) else ExtraOutputSpec(**dict(spec))
            for spec in (extras or [])
        ]
        self._hooks: ExtrasHookManager | None = None
        self.collect_outputs = collect_outputs
        self.collect_hidden_states = bool(collect_hidden_states)
        self.collect_attentions = bool(collect_attentions)
        self.batch_keys = list(batch_keys or ["input_ids", "attention_mask", "labels"])
        self.forward_kwargs = dict(forward_kwargs or {})
        if self.collect_hidden_states:
            self.forward_kwargs.setdefault("output_hidden_states", True)
        if self.collect_attentions:
            self.forward_kwargs.setdefault("output_attentions", True)
        self.producer_signature = producer_signature or type(self).__name__
        self.artifact_name = artifact_name
        self.artifact_version = artifact_version
        self.lineage_store: Any = None
        self._run_node_id: int | None = None
        self._sample_ids: list[str] = []
        self._device = torch.device("cpu")
        self._t_start = time.time()

        # Push header overrides onto the store now (so finalize writes them).
        if header_overrides:
            for k, v in header_overrides.items():
                if hasattr(self.store.header, k):
                    setattr(self.store.header, k, v)

    # ----- protocol --------------------------------------------------------

    def prepare(self, cfg: Mapping[str, Any] | None = None) -> None:
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            self._device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            self._device = torch.device("cpu")
        if self.extras and self._hooks is None:
            self._hooks = ExtrasHookManager(self.model, self.extras).attach()
        if cfg:
            if "lineage_store" in cfg:
                self.lineage_store = cfg["lineage_store"]
            if "artifact_name" in cfg:
                self.artifact_name = cfg["artifact_name"]
            if "artifact_version" in cfg:
                self.artifact_version = cfg["artifact_version"]
            # Explicit run-node id removes the
            # "first iter_nodes(kind='run') wins" guesswork.
            if "run_node_id" in cfg and cfg["run_node_id"] is not None:
                self._run_node_id = int(cfg["run_node_id"])

    def produce(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        sid = str(sample.get("id") or derive_sample_id(sample))
        if self.store.contains(sid):
            self._sample_ids.append(sid)
            return {}
        batch = _as_batch(sample, device=self._device)
        if self._hooks is not None:
            self._hooks.reset()
        with torch.no_grad():
            output = self.model(**{k: v for k, v in batch.items() if k in self.batch_keys
                                   or k == "modality_inputs"},
                                **self.forward_kwargs)
        if not isinstance(output, ModelOutput):
            output = _coerce_model_output(output)
        if self._hooks is not None:
            for name, payload in self._hooks.collect().items():
                if isinstance(payload, Mapping):
                    output.extras[name] = {k: v for k, v in payload.items()}
                else:
                    output.extras[name] = payload
        flat = flatten_model_output_tensors(output)
        tensors_out: dict[str, torch.Tensor] = {}
        for k, v in flat.items():
            if (self.collect_outputs is not None
                    and k in output.outputs
                    and k not in self.collect_outputs):
                continue
            t = v.detach().cpu()
            # Hidden-states / attention stacks come out as (L, B=1, ...) — the
            # outer L axis stays, the inner B=1 axis collapses.
            if k in ("hidden_states_layers", "attentions_layers") and t.dim() >= 3 and t.shape[1] == 1:
                t = t.squeeze(1)
            elif t.dim() and t.shape[0] == 1:
                t = t.squeeze(0)
            tensors_out[k] = t.contiguous()
        self.store.put(sid, tensors_out)
        self._sample_ids.append(sid)
        # update header schema map for parity with on-disk shapes
        for k, v in tensors_out.items():
            self.store.header.field_schema.setdefault(k, str(tuple(v.shape)))
            if not self.store.header.dtype:
                self.store.header.dtype = str(v.dtype)
        if not self.store.header.producer_signature:
            self.store.header.producer_signature = self.producer_signature
        return tensors_out

    def finalize(self) -> Path:
        manifest_path = self.store.finalize()
        if self._hooks is not None:
            self._hooks.detach()
            self._hooks = None

        # Lineage edge — failures are surfaced via ``warnings.warn`` instead of
        # silent ``except: pass`` so missing produced_by edges are visible in logs.
        ls = self.lineage_store
        if ls is not None:
            import warnings

            try:
                artifact_id = ls.upsert_node(
                    kind="artifact",
                    name=str(self.artifact_name or self.store.root.name),
                    version=str(self.artifact_version or "auto"),
                    schema_kind="artifact_header",
                    schema_version=self.store.header.schema_version,
                    payload_path=str(self.store.root),
                    payload={
                        "producer_signature": self.producer_signature,
                        "samples_count": len(self._sample_ids),
                        "header": self.store.header.to_dict(),
                    },
                )
                run_id = self._run_node_id
                if run_id is None:
                    # Fallback for embedded callers that didn't hand us a
                    # run_node_id. Prefer the most recent run row to keep
                    # behaviour stable when iter_nodes order is undefined.
                    candidates = [
                        rn for rn in ls.iter_nodes(kind="run") if rn.get("run_id")
                    ]
                    if candidates:
                        candidates.sort(
                            key=lambda r: (r.get("ts") or 0.0, r["id"]), reverse=True
                        )
                        run_id = int(candidates[0]["id"])
                        warnings.warn(
                            "ModelForwardProducer.finalize: no explicit run_node_id; "
                            f"falling back to most-recent run node id={run_id}",
                            stacklevel=2,
                        )
                if run_id is not None:
                    ls.add_edge(
                        int(run_id),
                        int(artifact_id),
                        "produced_by",
                        {"elapsed_s": time.time() - self._t_start},
                    )
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"ModelForwardProducer.finalize: lineage write failed: {e}",
                    stacklevel=2,
                )
        return manifest_path


def _coerce_model_output(out: Any) -> ModelOutput:
    if isinstance(out, ModelOutput):
        return out
    if isinstance(out, Mapping):
        return ModelOutput(outputs={k: v for k, v in out.items() if isinstance(v, torch.Tensor)})
    if hasattr(out, "logits"):
        return ModelOutput(outputs={"logits": out.logits},
                           loss=getattr(out, "loss", None),
                           hidden_states=tuple(getattr(out, "hidden_states", ())) or None,
                           attentions=tuple(getattr(out, "attentions", ())) or None)
    if isinstance(out, torch.Tensor):
        return ModelOutput(outputs={"output": out})
    return ModelOutput(outputs={})


def run_artifact_production(
    dataset: Iterable[Mapping[str, Any]],
    producer: ModelForwardProducer,
    *,
    progress: bool = False,
) -> Path:
    """CLI / convenience driver: iterate ``dataset`` end-to-end and finalize."""
    producer.prepare()
    for sample in dataset:
        producer.produce(sample)
    return producer.finalize()


__all__ = [
    "ArtifactProducerProtocol",
    "ModelForwardProducer",
    "run_artifact_production",
]
