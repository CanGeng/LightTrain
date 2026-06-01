"""Pre-flight resource estimate.

Constructs the recipe's model + a synthetic 1-step batch, walks one forward
in ``no_grad`` to bound activation memory, and reports:

* ``trainable_params`` / ``all_params`` / ``trainable_ratio``
* Parameter + gradient + optimizer-state byte budgets
* Activation byte bound (token_count * d_model * n_layers * 2 — coarse)
* A tokens/s estimate (timer over the dummy forward)

When ``cfg.engine.name == "layer_offload"`` we additionally fill an
:class:`OffloadEstimate` block with per-layer recompute vs transfer
microbenchmarks so the user can pick ``resident_layers`` knowingly. NVMe
bandwidth probe and PCIe stat parsing are opt-in; the default report is
fast and pure-Python.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch


# ---------------------------------------------------------------- byte sizing


_BYTES_PER_DTYPE = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.bool: 1,
}


def _param_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * _BYTES_PER_DTYPE.get(p.dtype, 4)
    return total


def _grad_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel() * _BYTES_PER_DTYPE.get(p.dtype, 4)
    return total


def _optim_state_bytes(model: torch.nn.Module, optim_name: str) -> int:
    """Optimizer state size estimate.

    * AdamW / Adam: 2x trainable params (m + v)
    * Lion: 1x (momentum)
    * SGD without momentum: 0
    * cpu_offload: same as its base optimizer; we treat it as AdamW here.
    """
    n_trainable_params = sum(
        p.numel() * _BYTES_PER_DTYPE.get(p.dtype, 4)
        for p in model.parameters()
        if p.requires_grad
    )
    name = (optim_name or "adamw").lower()
    if name in ("adamw", "adam", "cpu_offload"):
        return 2 * n_trainable_params
    if name in ("lion",):
        return 1 * n_trainable_params
    if name in ("sgd",):
        return 0
    # Conservative: assume Adam-like
    return 2 * n_trainable_params


def _resolve_optim_state_bytes(
    optim_spec: Any, model: torch.nn.Module, optim_name: str
) -> int:
    """Optimizer-state footprint, preferring the wrapper's ``optim_state_bytes``
    hook when available (issue #4).

    Instantiates the optimizer wrapper from the recipe's ``optim`` spec (cheap —
    ``__init__`` only; no ``build()``) and calls ``optim_state_bytes(model)`` if
    present, falling back to the name-based ``2 × params`` estimate otherwise —
    so estimate never hard-fails on a custom optimizer.

    Failure-first distinction (so a silently-wrong number can't masquerade as a
    real one):

    * **Can't resolve / hook errors** → a *problem* (name typo, ``user_modules``
      not imported, buggy hook). Warn loudly, then fall back.
    * **Resolved but no hook** → legitimate (most optimizers don't need one).
      Fall back silently.
    """
    fallback = _optim_state_bytes(model, optim_name or "adamw")
    if not isinstance(optim_spec, Mapping):
        return fallback
    spec = dict(optim_spec)
    if "name" not in spec and "_target_" not in spec:
        return fallback

    name = optim_name or spec.get("name") or spec.get("_target_") or "<optimizer>"
    try:
        from ..config._resolver import resolve as _resolve

        wrapper = _resolve(spec, category="optimizer")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"estimate: optimizer {name!r} could not be resolved ({exc}); "
            f"optim_state_bytes falls back to the generic 2×params estimate "
            f"and may not reflect a memory-efficient optimizer. Check the name "
            f"and that the recipe's `user_modules` are importable.",
            UserWarning,
            stacklevel=2,
        )
        return fallback

    hook = getattr(wrapper, "optim_state_bytes", None)
    if hook is None:
        # Legitimate: this optimizer just uses the generic estimate.
        return fallback
    try:
        return int(hook(model))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"estimate: {name!r}.optim_state_bytes() raised ({exc}); "
            f"falling back to the generic 2×params estimate.",
            UserWarning,
            stacklevel=2,
        )
        return fallback


# ---------------------------------------------------------------- core API


@dataclass
class OffloadEstimate:
    layers: int
    resident_layers: int
    layer_param_bytes: int
    recompute_us_per_layer: float
    transfer_us_per_layer: float
    recommended_mode: str  # "recompute" | "offload" | "mixed"
    pcie_bandwidth_used: str


@dataclass
class EstimateReport:
    trainable_params: int
    all_params: int
    trainable_ratio: float
    param_bytes: int
    grad_bytes: int
    optim_state_bytes: int
    activation_bytes_per_step: int
    total_bytes_per_step: int
    tokens_per_sec_estimate: float
    model_name: str = ""
    optimizer_name: str = ""
    engine_name: str = ""
    notes: list[str] = field(default_factory=list)
    offload: OffloadEstimate | None = None


def _spec_name(spec: Any) -> str:
    if spec is None:
        return ""
    if hasattr(spec, "model_dump"):
        spec = spec.model_dump()
    if isinstance(spec, Mapping):
        return str(spec.get("name") or spec.get("_target_") or "")
    return ""


def _activation_estimate(
    cfg: Mapping[str, Any], model: torch.nn.Module
) -> tuple[int, int]:
    """Return ``(token_count, activation_bytes)``.

    Coarse formula: ``B * T * d_model * 4 bytes * n_layers * 2`` accounts for
    pre / post attention residuals + the MLP intermediate. Replace with
    actual ``torch.profiler`` numbers in future when we wire real probing.
    """
    data = cfg.get("data") if isinstance(cfg, Mapping) else None
    batch_size = 8
    seq_len = 128
    if isinstance(data, Mapping):
        if "batch_size" in data:
            batch_size = int(data["batch_size"])
        # try collator.max_len
        collator = data.get("collator")
        if isinstance(collator, Mapping) and "max_len" in collator:
            seq_len = int(collator["max_len"])
    d_model = int(getattr(model, "d_model", 0)) or 0
    n_layers = int(getattr(model, "n_layers", 0)) or 0
    # Fall back to summing nn.Linear out_features when adapter doesn't expose
    # d_model — coarse but never zero.
    if d_model == 0:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                d_model = max(d_model, int(m.out_features))
    if d_model == 0:
        d_model = 256
    if n_layers == 0:
        n_layers = sum(
            1 for _ in (m for m in model.modules() if isinstance(m, torch.nn.LayerNorm))
        )
    if n_layers == 0:
        n_layers = 4
    tokens = batch_size * seq_len
    act_bytes = tokens * d_model * 4 * n_layers * 2
    return tokens, act_bytes


def _tokens_per_sec(model: torch.nn.Module, tokens: int) -> float:
    """Run one no-grad forward on a dummy batch to time tokens/sec."""
    try:
        batch_size = max(1, tokens // 128)
        seq_len = min(128, max(1, tokens // batch_size))
        device = next(model.parameters()).device
        vocab_size = int(getattr(model, "vocab_size", 0)) or 32
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        model.eval()
        with torch.no_grad():
            # warmup
            _ = model(input_ids=ids)
            t0 = time.perf_counter()
            for _ in range(3):
                _ = model(input_ids=ids)
            dt = max(1e-6, time.perf_counter() - t0)
        return (3 * batch_size * seq_len) / dt
    except Exception:  # noqa: BLE001
        return 0.0


def _offload_estimate(
    cfg: Mapping[str, Any], model: torch.nn.Module
) -> OffloadEstimate | None:
    engine = cfg.get("engine") if isinstance(cfg, Mapping) else None
    if not isinstance(engine, Mapping):
        return None
    if engine.get("name") != "layer_offload":
        return None
    resident_layers = int(engine.get("resident_layers", 2))
    # Try to import the plugin to get accurate per-layer probing; fall back
    # to a coarse estimate if it isn't installed.
    try:
        from lighttrain.plugins.layer_offload._io import (
            probe_layer_bandwidth,
        )

        recompute_us, transfer_us, layer_param_bytes, layer_count = (
            probe_layer_bandwidth(model)
        )
    except Exception:  # noqa: BLE001
        n_layers = int(getattr(model, "n_layers", 0)) or 4
        d_model = int(getattr(model, "d_model", 0)) or 256
        layer_param_bytes = d_model * d_model * 12 * 4  # rough qkv+proj+mlp
        # Coarse: 50 us recompute, 200 us transfer at 8 GB/s for ~1 MB
        recompute_us = 50.0 + (layer_param_bytes / 1024 / 1024) * 50.0
        transfer_us = (layer_param_bytes / 1024 / 1024 / 1024) * 1e6 / 8.0
        layer_count = n_layers
    mode = (
        "recompute"
        if recompute_us < transfer_us
        else ("offload" if transfer_us * 2 < recompute_us else "mixed")
    )
    return OffloadEstimate(
        layers=int(layer_count),
        resident_layers=resident_layers,
        layer_param_bytes=int(layer_param_bytes),
        recompute_us_per_layer=float(recompute_us),
        transfer_us_per_layer=float(transfer_us),
        recommended_mode=mode,
        pcie_bandwidth_used=f"~{layer_param_bytes / max(1.0, transfer_us) / 1e3:.2f} MB/ms",
    )


def estimate(cfg: Mapping[str, Any]) -> EstimateReport:
    """Build a recipe's model and report resource estimates."""
    # Lazy imports keep lab decoupled at module load.
    from ..config._components import import_all_components
    from ..config._models import (
        normalize_model_set,
        optim_spec_for,
        primary_trainable,
    )
    from ..config._resolver import _as_plain_dict
    from ..config._resolver import resolve as _resolve

    cfg_dict = _as_plain_dict(cfg)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"estimate: cfg must be a mapping, got {type(cfg).__name__}")

    # estimate() is public API (lab/__init__) and can be called directly with a
    # raw dict that never went through load_config's chokepoint — populate the
    # built-in registry and the recipe's user_modules here too (both idempotent;
    # free when load_config already ran via estimate_cmd).
    import_all_components()
    user_mods = cfg_dict.get("user_modules") or []
    if user_mods:
        from ..config._user_modules import import_user_modules

        import_user_modules(list(user_mods))

    # Single normalisation (no second declaration parse): build the primary
    # model and read its name + optimizer off the SAME resolved entry — supports
    # `models:` sets, not just the lone `model:`+`model_profiles:` form.
    models_cfg, optimizers_cfg = normalize_model_set(cfg_dict)
    _primary_name, _primary_entry = primary_trainable(models_cfg)
    model = _resolve(_primary_entry["spec"], category="model")
    model_name = _spec_name(_primary_entry["spec"])
    optim_spec = optim_spec_for(_primary_entry, optimizers_cfg)
    optim_name = _spec_name(optim_spec)
    engine_name = _spec_name(cfg_dict.get("engine"))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    p_bytes = _param_bytes(model)
    g_bytes = _grad_bytes(model)
    o_bytes = _resolve_optim_state_bytes(optim_spec, model, optim_name)
    _tok, a_bytes = _activation_estimate(cfg_dict, model)
    tps = _tokens_per_sec(model, max(8, _tok))
    notes: list[str] = []
    if engine_name == "layer_offload":
        notes.append(
            "layer_offload engine: optimizer / weights live on host; see "
            "offload section for layer-level recompute vs transfer breakdown."
        )

    return EstimateReport(
        trainable_params=int(n_trainable),
        all_params=int(n_all),
        trainable_ratio=float(n_trainable / max(1, n_all)),
        param_bytes=int(p_bytes),
        grad_bytes=int(g_bytes),
        optim_state_bytes=int(o_bytes),
        activation_bytes_per_step=int(a_bytes),
        total_bytes_per_step=int(p_bytes + g_bytes + o_bytes + a_bytes),
        tokens_per_sec_estimate=float(tps),
        model_name=model_name,
        optimizer_name=optim_name,
        engine_name=engine_name,
        notes=notes,
        offload=_offload_estimate(cfg_dict, model),
    )


def report_to_dict(rpt: EstimateReport) -> dict[str, Any]:
    """JSON-safe dump."""
    return asdict(rpt)


__all__ = ["estimate", "report_to_dict", "EstimateReport", "OffloadEstimate"]
