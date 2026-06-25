"""OOM report.

When ``torch.cuda.OutOfMemoryError`` (or any exception whose ``str()``
matches the OOM signature) reaches the trainer's top-level except, we
read ``torch.cuda.memory_stats()`` + ``memory_summary()`` to identify the
peak component, then emit:

```
diagnostics/oom_<ts>/
  report.md      # human-readable analysis + top-3 patch suggestions
  patch.yaml     # smallest yaml diff that should help
  apply.sh       # one-liner to apply the patch
```

We never apply automatically — the user has
to run ``lighttrain train --apply-degrade patch.yaml`` themselves.

CPU-only environments short-circuit: ``write_oom_report`` returns a
report dir with an explanatory note so the path is still observable to
the index page.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

_log = logging.getLogger(__name__)

# ----- peak component identification --------------------------------------


def _classify_peak(stats: dict[str, int], summary: str) -> str:
    """Best-effort one-of-five classification of the OOM cause."""
    if not stats:
        return "unknown"
    alloc = int(stats.get("allocated_bytes.all.peak", 0))
    reserved = int(stats.get("reserved_bytes.all.peak", 0))
    active = int(stats.get("active_bytes.all.peak", 0))
    # Heuristics — coarse on purpose; the report is advisory.
    summary_l = summary.lower()
    if "kv" in summary_l or "decoder" in summary_l:
        return "kv_cache"
    if active > 0 and active * 2 < alloc:
        return "optimizer_state"
    if reserved > 0 and reserved > alloc * 1.5:
        return "fragmentation"
    if alloc > 10 * (1 << 30):  # > 10 GB
        return "activation"
    return "activation"


_PATCH_BY_COMPONENT: dict[str, dict[str, Any]] = {
    "activation": {
        "expected_savings_pct": 40,
        "note": "activation memory dominates",
        "patch": {
            "engine": {"mixed_precision": "bf16"},
            "trainer": {"grad_clip": 1.0, "accumulate": 2},
            "training_tricks": {"gradient_checkpointing": True},
            "data": {"collator": {"max_len": "<halve>"}},
        },
    },
    "optimizer_state": {
        "expected_savings_pct": 50,
        "note": "optimizer state dominates",
        "patch": {
            "optim": {"name": "lion"},  # cheaper state than AdamW
            "engine": {"mixed_precision": "bf16"},
        },
    },
    "kv_cache": {
        "expected_savings_pct": 30,
        "note": "KV cache dominates",
        "patch": {
            "engine": {"mixed_precision": "bf16"},
            "trainer": {"accumulate": 2},
        },
    },
    "fragmentation": {
        "expected_savings_pct": 15,
        "note": "memory fragmentation suspected",
        "patch": {
            "trainer": {"accumulate": 2},
        },
    },
    "unknown": {
        "expected_savings_pct": 10,
        "note": "no CUDA stats available; conservative fallback",
        "patch": {
            "trainer": {"accumulate": 2},
            "engine": {"mixed_precision": "bf16"},
        },
    },
}


def write_oom_report(
    run_dir: str | Path,
    *,
    exception: BaseException | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Generate an OOM report directory. Returns the path."""
    run_dir = Path(run_dir)
    out = run_dir / "diagnostics" / f"oom_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)

    cuda_ok = bool(torch.cuda.is_available())
    stats: dict[str, int] = {}
    summary: str = ""
    if cuda_ok:
        try:
            stats = dict(torch.cuda.memory_stats())
            summary = torch.cuda.memory_summary()
        except Exception:  # noqa: BLE001
            _log.warning(
                "oom_report: reading CUDA memory stats failed; peak component falls back to 'unknown'",
                exc_info=True,
            )

    component = _classify_peak(stats, summary) if cuda_ok else "unknown"
    suggestion = _PATCH_BY_COMPONENT[component]

    # report.md
    lines = [
        f"# OOM report — {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        f"- Exception: `{type(exception).__name__ if exception else 'OutOfMemoryError'}`",
        f"- Identified peak component: **{component}** — {suggestion['note']}",
        f"- Expected savings (rough): ~{suggestion['expected_savings_pct']}%",
        "",
    ]
    if cuda_ok:
        lines += [
            "## CUDA memory snapshot",
            "",
            f"- allocated peak: {_fmt_bytes(stats.get('allocated_bytes.all.peak', 0))}",
            f"- reserved peak:  {_fmt_bytes(stats.get('reserved_bytes.all.peak', 0))}",
            f"- active peak:    {_fmt_bytes(stats.get('active_bytes.all.peak', 0))}",
            "",
            "<details><summary>full memory_summary</summary>",
            "",
            "```",
            summary[:4000],
            "```",
            "",
            "</details>",
            "",
        ]
    else:
        lines += [
            "## CUDA snapshot",
            "",
            "_OOM detected on CPU-only environment — no `torch.cuda.memory_stats` available. "
            "The patch below is a conservative fallback._",
            "",
        ]

    lines += [
        "## Recommended top-3 patches",
        "",
        f"1. {suggestion['note']} → see `patch.yaml`",
        "2. Reduce per-step batch size (set `data.batch_size: 1`).",
        "3. Switch optimizer to 8-bit (e.g. `optim.name: lion`) if AdamW dominates.",
        "",
        "## How to apply",
        "",
        "```bash",
        f"lighttrain train -c {config_path or '<your-recipe.yaml>'} --apply-degrade patch.yaml",
        "```",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # patch.yaml — write under proper YAML.
    try:
        import yaml as _yaml

        (out / "patch.yaml").write_text(
            _yaml.safe_dump(suggestion["patch"], sort_keys=False),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        _log.warning(
            "oom_report: YAML dump of patch failed; writing patch.yaml as JSON fallback",
            exc_info=True,
        )
        (out / "patch.yaml").write_text(
            json.dumps(suggestion["patch"], indent=2),
            encoding="utf-8",
        )

    # apply.sh
    cfg = str(config_path or "<your-recipe.yaml>")
    (out / "apply.sh").write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\n"
        f'lighttrain train -c "{cfg}" --apply-degrade "{out / "patch.yaml"}"\n',
        encoding="utf-8",
    )
    return out


def _fmt_bytes(n: int) -> str:
    n = int(n)
    if n < 1024:
        return f"{n} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n_f = n / 1024
        if n_f < 1024 or unit == "TiB":
            return f"{n_f:.1f} {unit}"
        n = int(n_f)
    # unreachable: the ``unit == "TiB"`` arm above always returns on the last
    # iteration — fall back kept off to avoid dead code.
    raise AssertionError("unreachable")  # pragma: no cover


def is_oom_exception(exc: BaseException) -> bool:
    """Lightweight check that doesn't depend on torch>=2.0 OutOfMemoryError."""
    if torch.cuda.is_available():
        try:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:  # noqa: BLE001
            _log.warning(
                "oom_report: OutOfMemoryError isinstance check failed; falling back to message-string matching",
                exc_info=True,
            )
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


__all__ = ["is_oom_exception", "write_oom_report"]
