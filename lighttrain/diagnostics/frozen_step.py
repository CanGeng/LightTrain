"""Frozen Step Bundle.

A "frozen step" is a single-file zip that captures the *exact* state at
the entry of a training step so the user can replay that step
post-mortem:

```
frozen_steps/step_<n>_<reason>.zip
├── batch.pt
├── model_state.safetensors
├── optimizer_state.pt
├── rng.pt
├── config.resolved.yaml
├── code_snapshot_pointer.txt
├── lineage_pointer.json
└── step_metadata.json
```

There are two users:

1. :class:`FrozenStepCallback` (``callbacks/builtins/frozen_step.py``) —
   the scheduled producer; snapshots state at ``on_step_begin`` and
   commits a zip at ``on_step_end`` when ``step % every == 0``.

2. :class:`crash_bundle` and the ``replay`` CLI — read existing zips.

The Writer holds the *most recent* snapshot in memory so the
StandardUpdateRule's RETRY_STEP path can borrow it to restore model
parameters between retries.
"""

from __future__ import annotations

import copy
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_model as _save_model

from ..minimal import build_minimal_model, dump_spec, load_state
from ..utils.seed import restore_rng_state, rng_state


_REASONS = ("scheduled", "exception", "cli", "retry")


@dataclass
class FrozenStepBundle:
    """In-memory representation of an extracted frozen step zip."""

    step: int
    reason: str
    batch: dict[str, Any]
    model_spec: dict[str, Any]
    model_state_bytes: bytes  # safetensors blob
    optimizer_state: dict[str, Any] | None
    rng_state: dict[str, Any] | None
    config_resolved_yaml: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FrozenStepWriter:
    """Snapshot → commit → restore lifecycle for a single run.

    Usage from :class:`FrozenStepCallback`::

        writer.snapshot(step, ctx, batch, model, optimizer, config_yaml)
        # ... step runs ...
        writer.commit(reason="scheduled")
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        mode: str = "lab",
        lineage_store: Any | None = None,
        run_node_id: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.mode = mode
        self.lineage_store = lineage_store
        self.run_node_id = run_node_id
        self.run_id = run_id
        self._snapshot: dict[str, Any] | None = None
        self.frozen_dir = self.run_dir / "frozen_steps"
        self.frozen_dir.mkdir(parents=True, exist_ok=True)

    # ----- snapshot --------------------------------------------------------

    def snapshot(
        self,
        *,
        step: int,
        ctx: Any,
        batch: Mapping[str, Any],
        model: torch.nn.Module,
        optimizer: Any,
        config_resolved_yaml: str = "",
    ) -> None:
        """Cache the per-step state needed for both commit + retry restore.

        Model params and optimizer state are deep-copied to CPU so a
        subsequent ``optimizer.step()`` doesn't mutate the snapshot.
        """
        try:
            snap_model = {
                k: v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
        except Exception:  # noqa: BLE001
            snap_model = None
        try:
            opt_state = copy.deepcopy(
                optimizer.state_dict() if hasattr(optimizer, "state_dict") else None
            )
        except Exception:  # noqa: BLE001
            opt_state = None
        try:
            rng = rng_state()
        except Exception:  # noqa: BLE001
            rng = None
        safe_batch = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        model_spec = _infer_model_spec(model)
        self._snapshot = {
            "step": int(step),
            "epoch": int(getattr(ctx, "epoch", 0)),
            "batch": safe_batch,
            "model_state": snap_model,
            "model_obj": model,  # for safetensors write; not committed to disk
            "optimizer_state": opt_state,
            "rng_state": rng,
            "config_resolved_yaml": config_resolved_yaml or "",
            "model_spec": model_spec,
            "ts": time.time(),
        }

    def restore_snapshot(
        self,
        *,
        model: torch.nn.Module | None = None,
        optimizer: Any | None = None,
    ) -> None:
        """Restore model params / optimizer state / RNG from the cached
        snapshot. Used by RETRY_STEP path so retries see the pre-step state.
        """
        if self._snapshot is None:
            return
        snap = self._snapshot
        if model is not None and snap.get("model_state") is not None:
            try:
                model.load_state_dict(snap["model_state"], strict=False)
            except Exception:  # noqa: BLE001
                pass
        if (
            optimizer is not None
            and snap.get("optimizer_state") is not None
            and hasattr(optimizer, "load_state_dict")
        ):
            try:
                optimizer.load_state_dict(snap["optimizer_state"])
            except Exception:  # noqa: BLE001
                pass
        if snap.get("rng_state") is not None:
            try:
                restore_rng_state(snap["rng_state"])
            except Exception:  # noqa: BLE001
                pass

    # ----- commit ----------------------------------------------------------

    def commit(self, *, reason: str = "scheduled") -> Path | None:
        """Atomically write the most recent snapshot to disk."""
        if self._snapshot is None:
            return None
        if reason not in _REASONS:
            reason = "scheduled"
        snap = self._snapshot
        step = int(snap["step"])
        out_path = self.frozen_dir / f"step_{step}_{reason}.zip"
        tmp = out_path.with_suffix(".zip.tmp")
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
                # 1) batch.pt — torch.save into BytesIO then write into zip.
                buf = io.BytesIO()
                torch.save(snap["batch"], buf)
                zf.writestr("batch.pt", buf.getvalue())
                # 2) model state — use save_model on a fresh in-memory file.
                tmp_model = tmp.with_suffix(".st.tmp")
                try:
                    _save_model(snap["model_obj"], str(tmp_model))
                    zf.writestr(
                        "model_state.safetensors", tmp_model.read_bytes()
                    )
                finally:
                    try:
                        tmp_model.unlink()
                    except FileNotFoundError:
                        pass
                # 3) optimizer state.
                if snap.get("optimizer_state") is not None:
                    buf = io.BytesIO()
                    torch.save(snap["optimizer_state"], buf)
                    zf.writestr("optimizer_state.pt", buf.getvalue())
                # 4) RNG state.
                if snap.get("rng_state") is not None:
                    buf = io.BytesIO()
                    torch.save(snap["rng_state"], buf)
                    zf.writestr("rng.pt", buf.getvalue())
                # 5) config snapshot.
                zf.writestr(
                    "config.resolved.yaml",
                    snap.get("config_resolved_yaml") or "",
                )
                # 6) code_snapshot_pointer.txt — prefer the resolved
                #    ``<run_dir>/code.snapshot/`` directory when it exists.
                #    Falls back to the run dir so older bundles keep working.
                snap_dir = self.run_dir / "code.snapshot"
                pointer = snap_dir if snap_dir.exists() else self.run_dir
                zf.writestr(
                    "code_snapshot_pointer.txt",
                    str(pointer.resolve()) + "\n",
                )
                # 7) lineage_pointer.json.
                zf.writestr(
                    "lineage_pointer.json",
                    json.dumps(
                        {
                            "run_id": self.run_id,
                            "run_node_id": self.run_node_id,
                            "lineage_path": str(
                                self.run_dir / "lineage.sqlite"
                            ),
                        },
                        indent=2,
                    ),
                )
                # 8) step_metadata.json.
                zf.writestr(
                    "step_metadata.json",
                    json.dumps(
                        {
                            "step": step,
                            "epoch": snap["epoch"],
                            "reason": reason,
                            "ts": snap["ts"],
                            "model_spec": snap["model_spec"],
                        },
                        indent=2,
                    ),
                )
            os.replace(tmp, out_path)
        except Exception:  # noqa: BLE001
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            return None
        # Lineage write — non-fatal; a logging failure must never take
        # the trainer down here.
        if self.lineage_store is not None and self.run_id is not None:
            try:
                node_id = self.lineage_store.upsert_node(
                    kind="frozen_step",
                    name=str(self.run_id),
                    version=f"step_{step}_{reason}",
                    run_id=str(self.run_id),
                    step=step,
                    schema_kind="frozen_step",
                    schema_version="0.4",
                    payload_path=str(out_path),
                    payload={"reason": reason, "ts": snap["ts"]},
                )
                if self.run_node_id is not None and node_id:
                    self.lineage_store.add_edge(
                        int(self.run_node_id), int(node_id), "produced_by",
                        {"reason": reason, "step": step},
                    )
            except Exception:  # noqa: BLE001
                pass
        return out_path


# ---------------------------------------------------------------- read side


def read_frozen_step_bundle(path: str | Path) -> FrozenStepBundle:
    """Open a frozen step zip and return its contents."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("step_metadata.json").decode("utf-8"))
        cfg_yaml = zf.read("config.resolved.yaml").decode("utf-8")
        batch = torch.load(io.BytesIO(zf.read("batch.pt")), weights_only=True)
        model_state_bytes = zf.read("model_state.safetensors")
        opt_state = None
        if "optimizer_state.pt" in names:
            opt_state = torch.load(
                io.BytesIO(zf.read("optimizer_state.pt")), weights_only=False
            )
        rng = None
        if "rng.pt" in names:
            rng = torch.load(
                io.BytesIO(zf.read("rng.pt")), weights_only=False
            )
    return FrozenStepBundle(
        step=int(meta["step"]),
        reason=str(meta.get("reason", "scheduled")),
        batch=dict(batch),
        model_spec=dict(meta.get("model_spec", {})),
        model_state_bytes=model_state_bytes,
        optimizer_state=opt_state,
        rng_state=rng,
        config_resolved_yaml=cfg_yaml,
        metadata=meta,
    )


def replay_step_bundle(
    bundle: FrozenStepBundle | str | Path,
    *,
    loss_fn: Any | None = None,
    debugger: bool = False,
    inject: str | Path | None = None,
    do_backward: bool = True,
) -> dict[str, Any]:
    """Re-run forward (+ optionally backward) from a frozen step bundle.

    Returns a dict ``{loss, grad_norm, original_step}`` for the caller to
    compare against the original ``step_metadata.json``. ``debugger=True``
    drops into ``pdb`` before forward; ``inject=path`` exec's a snippet
    (intended for monkey-patches in lab mode).
    """
    if not isinstance(bundle, FrozenStepBundle):
        bundle = read_frozen_step_bundle(bundle)

    # 1) Rebuild model.
    if "name" in bundle.model_spec or "_target_" in bundle.model_spec:
        # Adapters must be importable for short-name resolution.
        try:
            import lighttrain.models.adapters  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        model = build_minimal_model(bundle.model_spec)
    else:
        raise RuntimeError("frozen step bundle has no model spec")
    # Load state from in-memory safetensors blob.
    tmp_st = Path(".__lt_frozen_step_state.tmp.safetensors")
    try:
        tmp_st.write_bytes(bundle.model_state_bytes)
        load_state(model, tmp_st, strict=False)
    finally:
        try:
            tmp_st.unlink()
        except FileNotFoundError:
            pass

    # 2) Restore RNG.
    if bundle.rng_state is not None:
        try:
            restore_rng_state(bundle.rng_state)
        except Exception:  # noqa: BLE001
            pass

    # 3) Debugger / inject hooks.
    if debugger:  # pragma: no cover — interactive
        import pdb

        pdb.set_trace()
    if inject is not None:
        path = Path(inject)
        code = path.read_text(encoding="utf-8")
        ns = {"model": model, "batch": bundle.batch}
        exec(code, ns, ns)  # noqa: S102 — explicit lab tool

    # 4) Forward + (optional) backward.
    model.train()
    out = model(**bundle.batch)
    logits = out.outputs.get("logits") if hasattr(out, "outputs") else None
    loss_value = None
    grad_norm = None
    if loss_fn is not None and logits is not None:
        from ..protocols import LossContext  # local — avoid cycles

        loss_dict = loss_fn(
            out,
            bundle.batch,
            LossContext(step=bundle.step, epoch=int(bundle.metadata.get("epoch", 0))),
        )
        loss = loss_dict.get("loss")
        if isinstance(loss, torch.Tensor):
            loss_value = float(loss.detach().item())
            if do_backward:
                loss.backward()
                total = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total += float(p.grad.detach().data.norm(2).item()) ** 2
                grad_norm = float(total**0.5)
    return {
        "step": bundle.step,
        "reason": bundle.reason,
        "loss": loss_value,
        "grad_norm": grad_norm,
        "logits_shape": (list(logits.shape) if isinstance(logits, torch.Tensor) else None),
    }


# ---------------------------------------------------------------- helpers


def _infer_model_spec(model: torch.nn.Module) -> dict[str, Any]:
    from ..registry import contains

    # PEFT wrappers (LoRA / IA3 / QLoRA) take priority — `dump_peft_spec`
    # records the base model spec alongside the adapter config so that
    # `build_minimal_model` can reconstruct the exact wrap. Falls back to the
    # short-name / _target_ paths if peft isn't installed or the model
    # isn't a recognised adapter.
    try:
        from ..models.peft import dump_peft_spec, is_peft_wrapped

        if is_peft_wrapped(model):
            return dump_peft_spec(model)
    except ImportError:
        pass

    cls = type(model)
    for name in (
        cls.__name__.lower(),
        cls.__name__.lower().replace("model", ""),
        cls.__name__.lower().replace("causallm", "_lm"),
    ):
        if contains("model", name):
            params = {}
            for k in (
                "vocab_size",
                "d_model",
                "n_layers",
                "n_heads",
                "max_seq_len",
                "dropout",
                "tie_weights",
            ):
                if hasattr(model, k):
                    v = getattr(model, k)
                    if isinstance(v, (int, float, bool, str)):
                        params[k] = v
            return dump_spec(name, params)
    return {
        "_target_": f"{cls.__module__}:{cls.__name__}",
        "params": {},
    }


__all__ = [
    "FrozenStepBundle",
    "FrozenStepWriter",
    "read_frozen_step_bundle",
    "replay_step_bundle",
]
