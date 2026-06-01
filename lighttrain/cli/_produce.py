"""``lighttrain produce-artifact`` driver.

Pulled out of ``_app.py`` to keep the CLI module pickle-light and to give the
test suite a callable entry point. The flow:

  1. Build model + dataset via ``_runtime`` plumbing (no trainer / optimizer).
  2. Resolve ``cfg.artifacts.producer`` + ``cfg.artifacts.store`` into a
     :class:`ModelForwardProducer`.
  3. Iterate samples, call ``produce``, finalize, return manifest path.

The lineage store is opened against the run dir so the producer can write a
``produced_by`` edge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import OmegaConf

from ..config import RootConfig, load_config
from ..utils.run_dir import make_run_dir, slugify
from ..utils.seed import seed_everything
from ._runtime import _build_data, _build_model, _to_dict


def run_produce(
    config: Path,
    *,
    overrides: list[str] | None = None,
    estimate: bool = False,
    console: Any | None = None,
) -> Path:
    snapshot_yaml = Path(config).read_text(encoding="utf-8")
    # load_config populates the registry (register_components default True).
    cfg = load_config(config, overrides=overrides or [])
    seed_everything(int(cfg.seed))

    art_spec = _to_dict(getattr(cfg, "artifacts", None))
    if not art_spec:
        raise RuntimeError(f"recipe {config} is missing `artifacts:` section")
    producer_spec = dict(art_spec.get("producer") or {})
    store_spec = dict(art_spec.get("store") or {})
    if "name" not in producer_spec:
        producer_spec.setdefault("name", "model_forward")
    if not store_spec:
        raise RuntimeError("artifacts.store is required (root + backend name)")

    resolved_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump()))
    run_dir = make_run_dir(
        cfg.run_root,
        cfg.exp,
        slug=slugify(cfg.exp),
        snapshot_yaml=snapshot_yaml,
        resolved_yaml=resolved_yaml,
    )
    if console is not None:
        console.print(f"[green]produce run_dir[/] = {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)
    data_module = _build_data(cfg, run_dir=run_dir, console=console)
    dataset = getattr(data_module, "dataset", None)
    if dataset is None:
        raise RuntimeError("data module has no `dataset`; produce-artifact needs one")

    # Lineage store on the run dir.
    from ..lineage.store import LineageStore

    lineage_store = LineageStore(run_dir / "lineage.sqlite")
    run_node = lineage_store.upsert_node(
        kind="run", name=run_dir.name, version=str(run_dir.name),
        run_id=run_dir.name, schema_kind="run_meta", schema_version="0.4",
        payload_path=str(run_dir), payload={"command": "produce-artifact"},
    )

    # Build producer
    from ..config._resolver import resolve as _resolve

    spec = dict(producer_spec)
    spec.setdefault("model", model)
    spec.setdefault("store", store_spec)
    producer = _resolve(spec, category="artifact_producer")
    # Explicitly hand the run node id so finalize() writes ``produced_by``
    # against the correct run, not the first ``kind='run'`` row in the DB.
    producer.prepare({
        "lineage_store": lineage_store,
        "run_node_id": int(run_node),
        "artifact_name": store_spec.get("artifact_name") or store_spec.get("root", Path(".")).split("/")[-1],
    })

    if estimate:
        # Best-effort sample-count + size estimate.
        try:
            n = len(dataset)  # type: ignore[arg-type]
            if console is not None:
                console.print(f"[cyan]estimate[/] samples={n}, store={store_spec.get('name')}")
        except TypeError:
            if console is not None:
                console.print("[cyan]estimate[/] dataset has no __len__; iterating to count")
        return run_dir

    n_seen = 0
    for sample in _iter_dataset(dataset):
        producer.produce(sample)
        n_seen += 1
        if console is not None and n_seen % 100 == 0:
            console.print(f"[grey]produced[/] {n_seen}")
    manifest_path = producer.finalize()
    if console is not None:
        console.print(f"[green]produced {n_seen} samples → {manifest_path}[/]")
    return manifest_path


def _iter_dataset(dataset: Any) -> Iterable[Mapping[str, Any]]:
    if hasattr(dataset, "__iter__") and not hasattr(dataset, "__getitem__"):
        for sample in dataset:
            yield sample
        return
    n = len(dataset) if hasattr(dataset, "__len__") else None  # type: ignore[arg-type]
    if n is not None:
        for i in range(n):
            yield dataset[i]
    else:
        for sample in dataset:
            yield sample


__all__ = ["run_produce"]
