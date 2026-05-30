"""`resume-verify` — strong single-pass-vs-resume parity check.

Trains a recipe two ways and compares step-aligned losses:

1. **single pass** — one run of ``N+M`` steps;
2. **resume** — a run of ``N`` steps that checkpoints, then a second run that
   restores that checkpoint and continues to ``N+M``.

If the checkpoint capsule (model + optimizer + scheduler + RNG + data position)
is faithful, the two loss trajectories agree at every step. The post-``N`` steps
are the meaningful part: they prove resume reproduces what an uninterrupted run
would have computed. This is strictly stronger than "phase 2 ran without
crashing", which is all the mamba3 launcher's ``--resume-test`` checked.

Numerical note: under mixed precision (bf16) reduction order is
hardware-dependent, so the default tolerance is ``1e-2``, not bit-exact. For a
bit-exact check run the recipe in fp32 with a single dataloader worker. The
comparison also assumes a resumable sampler and ``num_workers=0`` so the data
order is reproducible across the resume boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TOL = 1e-2


@dataclass
class ResumeVerifyReport:
    phase1_steps: int
    phase2_steps: int
    tol: float
    single_pass_losses: list[float]
    resume_losses: list[float]
    per_step_delta: list[float] = field(default_factory=list)
    max_abs_delta: float = 0.0
    passed: bool = False
    note: str = ""


def _read_step_losses(run_dir: Path) -> dict[int, float]:
    """step → last logged loss for that step, from ``logs/metrics.jsonl``."""
    metrics = Path(run_dir) / "logs" / "metrics.jsonl"
    out: dict[int, float] = {}
    if not metrics.exists():
        return out
    for line in metrics.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in row and "step" in row:
            out[int(row["step"])] = float(row["loss"])
    return out


def _losses_in_order(by_step: dict[int, float]) -> list[float]:
    return [by_step[s] for s in sorted(by_step)]


def _fit_and_close(bundle: dict[str, Any]) -> None:
    try:
        bundle["trainer"].fit()
    finally:
        if bundle.get("logger") is not None:
            bundle["logger"].close()


def resume_verify(
    config: Any,
    phase1_steps: int,
    phase2_steps: int,
    *,
    tol: float = DEFAULT_TOL,
    overrides: list[str] | None = None,
) -> ResumeVerifyReport:
    """Run the single-pass and resume trajectories and diff their losses.

    ``config`` and ``overrides`` are passed straight to
    ``setup_run_from_config``; the caller's overrides (e.g. ``model=transformer``)
    apply to both paths so they stay comparable.
    """
    # Imported here to avoid a lab→cli import at module load.
    from ..cli._runtime import setup_run_from_config

    total = phase1_steps + phase2_steps
    base = list(overrides or [])

    # 1. Single pass: N+M steps in one go.
    single = setup_run_from_config(
        config,
        overrides=base + [f"trainer.max_steps={total}", "exp=rv_single_pass"],
    )
    _fit_and_close(single)
    single_losses = _losses_in_order(_read_step_losses(single["run_dir"]))

    # 2. Phase 1: N steps, checkpoint at step N.
    phase1 = setup_run_from_config(
        config,
        overrides=base
        + [f"trainer.max_steps={phase1_steps}", f"trainer.ckpt_every={phase1_steps}", "exp=rv_resume"],
    )
    _fit_and_close(phase1)
    ckpt = phase1["trainer"].ckpt_manager.latest()
    if ckpt is None:
        raise RuntimeError(
            f"resume-verify: no checkpoint written at step {phase1_steps}; "
            "set ckpt_every so a checkpoint lands on the phase-1 boundary."
        )

    # 3. Phase 2: resume into the phase-1 run dir, continue to N+M.
    phase2 = setup_run_from_config(
        config,
        overrides=base + [f"trainer.max_steps={total}"],
        existing_run_dir=phase1["run_dir"],
    )
    phase2["trainer"].load_checkpoint(ckpt)
    _fit_and_close(phase2)
    resume_losses = _losses_in_order(_read_step_losses(phase1["run_dir"]))

    # 4. Compare step-aligned.
    n = min(len(single_losses), len(resume_losses))
    deltas = [abs(single_losses[i] - resume_losses[i]) for i in range(n)]
    max_delta = max(deltas) if deltas else float("inf")
    length_ok = len(single_losses) == len(resume_losses) == total
    passed = length_ok and max_delta <= tol

    note = ""
    if not length_ok:
        note = (
            f"step-count mismatch: single={len(single_losses)}, "
            f"resume={len(resume_losses)}, expected={total}"
        )

    return ResumeVerifyReport(
        phase1_steps=phase1_steps,
        phase2_steps=phase2_steps,
        tol=tol,
        single_pass_losses=single_losses,
        resume_losses=resume_losses,
        per_step_delta=deltas,
        max_abs_delta=max_delta,
        passed=passed,
        note=note,
    )


def render_report(report: ResumeVerifyReport) -> str:
    """Human-readable summary, including the per-step deltas around the boundary."""
    lines = [
        f"resume-verify: phase1={report.phase1_steps} phase2={report.phase2_steps} "
        f"tol={report.tol:g}",
        f"  max |Δloss| = {report.max_abs_delta:.3e}  →  "
        f"{'PASS' if report.passed else 'FAIL'}",
    ]
    if report.note:
        lines.append(f"  note: {report.note}")
    boundary = report.phase1_steps
    for i, d in enumerate(report.per_step_delta):
        step = i + 1
        mark = "  ← resume boundary" if step == boundary + 1 else ""
        if step > boundary - 1 or d > report.tol:
            sp = report.single_pass_losses[i]
            rs = report.resume_losses[i]
            lines.append(f"  step {step:>4}: single={sp:.6f} resume={rs:.6f} Δ={d:.3e}{mark}")
    return "\n".join(lines)


__all__ = ["ResumeVerifyReport", "resume_verify", "render_report", "DEFAULT_TOL"]
