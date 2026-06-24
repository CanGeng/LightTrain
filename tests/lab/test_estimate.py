"""`lighttrain estimate` + lab.estimate API — DESIGN §20.10 (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.cli._runtime import _eager_import_components
from lighttrain.lab.estimate import EstimateReport, estimate
from tests._diagnostics import expect_exists

_eager_import_components()


def _tiny_cfg(**overrides):
    base = {
        "mode": "lab",
        "seed": 0,
        "exp": "estimate_smoke",
        "run_root": "runs",
        "model": "default",
        "model_profiles": {
            "default": {
                "name": "tiny_lm",
                "vocab_size": 64,
                "d_model": 16,
                "n_layers": 2,
                "n_heads": 4,
                "max_seq_len": 32,
            }
        },
        "data": {
            "name": "simple",
            "dataset": {"name": "line_file_text", "path": "tests/fixtures/tiny_corpus.txt", "max_len": 32},
            "tokenizer": {"name": "byte"},
            "collator": {"name": "causal_lm", "max_len": 32},
            "batch_size": 4,
        },
        "loss": {"name": "cross_entropy"},
        "optim": {"name": "adamw", "lr": 1e-3},
        "scheduler": {"name": "constant"},
        "engine": {"name": "standard", "mixed_precision": "no"},
        "trainer": {"name": "pretrain", "max_steps": 10},
    }
    base.update(overrides)
    return base


def test_estimate_returns_filled_report_for_tiny_lm():
    cfg = _tiny_cfg()
    rpt = estimate(cfg)
    assert isinstance(rpt, EstimateReport)
    assert rpt.trainable_params > 0
    assert rpt.all_params > 0
    assert rpt.trainable_ratio == pytest.approx(rpt.trainable_params / rpt.all_params)
    assert rpt.trainable_ratio == 1.0  # no LoRA — everything trainable
    assert rpt.param_bytes > 0
    assert rpt.grad_bytes > 0
    assert rpt.optim_state_bytes >= 2 * rpt.grad_bytes  # AdamW
    assert rpt.activation_bytes_per_step > 0
    assert rpt.total_bytes_per_step == (
        rpt.param_bytes
        + rpt.grad_bytes
        + rpt.optim_state_bytes
        + rpt.activation_bytes_per_step
    )
    assert rpt.engine_name == "standard"
    assert rpt.model_name == "tiny_lm"
    assert rpt.optimizer_name == "adamw"


def test_estimate_uses_optim_state_bytes_hook_lion_half_of_adamw():
    """Issue #4: estimate calls the wrapper's ``optim_state_bytes`` hook.

    Lion keeps one momentum buffer (1× params) vs AdamW's two (2× params), so
    for the same model Lion's reported optimizer-state bytes is half AdamW's.
    This exercises the hook path end-to-end (registry → wrapper → estimate).
    """
    adamw = estimate(_tiny_cfg(optim={"name": "adamw", "lr": 1e-3}))
    lion = estimate(_tiny_cfg(optim={"name": "lion", "lr": 1e-4}))
    assert lion.optim_state_bytes == pytest.approx(adamw.optim_state_bytes / 2)
    assert lion.optim_state_bytes < adamw.optim_state_bytes


def test_estimate_hook_overrides_name_default_for_memory_efficient_optimizer():
    """A registered optimizer exposing ``optim_state_bytes`` makes its real
    (smaller) footprint visible through ``estimate`` — the generalization of
    GaLore's saving to any memory-efficient optimizer.
    """
    from lighttrain.builtin_plugins.optim.wrappers import AdamWWrapper
    from lighttrain.registry import contains as _has
    from lighttrain.registry import register

    if not _has("optimizer", "_tiny_lowrank_test"):
        @register("optimizer", "_tiny_lowrank_test")
        class _TinyLowRank(AdamWWrapper):
            # Pretend only a quarter of Adam's full-rank state is needed.
            def optim_state_bytes(self, model):
                return super().optim_state_bytes(model) // 4

    baseline = estimate(_tiny_cfg(optim={"name": "adamw", "lr": 1e-3}))
    lowrank = estimate(_tiny_cfg(optim={"name": "_tiny_lowrank_test", "lr": 1e-3}))
    assert lowrank.optim_state_bytes == baseline.optim_state_bytes // 4
    assert lowrank.optim_state_bytes < baseline.optim_state_bytes


def test_estimate_warns_when_optimizer_cannot_be_resolved():
    """Failure-first: an unresolvable optimizer (name typo / user_modules not
    imported) must WARN before falling back to the generic 2×params estimate —
    so a silently-wrong number can't masquerade as a real one."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rpt = estimate(_tiny_cfg(optim={"name": "_no_such_optimizer_xyz", "lr": 1e-3}))
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("_no_such_optimizer_xyz" in m and "fall" in m.lower() for m in msgs), msgs
    # Fell back to the generic estimate (2× trainable bytes) rather than crashing.
    assert rpt.optim_state_bytes >= 2 * rpt.grad_bytes


def test_estimate_does_not_warn_for_resolvable_optimizer_without_hook():
    """A resolvable optimizer that simply doesn't implement optim_state_bytes is
    legitimate — no warning, silent fallback. (Here adamw DOES have the
    OptimizerWrapperBase default, so this also guards against spurious warnings.)"""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate(_tiny_cfg(optim={"name": "adamw", "lr": 1e-3}))
    optim_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "optim_state_bytes" in str(w.message)
    ]
    assert not optim_warnings, [str(w.message) for w in optim_warnings]


def test_estimate_with_lora_reports_low_trainable_ratio():
    pytest.importorskip("peft")
    cfg = _tiny_cfg(
        model="default",
        model_profiles={
            "default": {
                "name": "lora",
                "base": {
                    "name": "tiny_lm",
                    "vocab_size": 64,
                    "d_model": 16,
                    "n_layers": 2,
                    "n_heads": 4,
                    "max_seq_len": 32,
                },
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
            }
        },
    )
    rpt = estimate(cfg)
    assert rpt.trainable_ratio < 0.10
    assert rpt.model_name == "lora"


def test_estimate_layer_offload_attaches_offload_block():
    # builtin_plugins/layer_offload registers `layer_offload` engine; if the
    # plugin isn't importable yet (G stage hasn't landed), skip gracefully.
    from lighttrain.registry import contains as _has

    if not _has("engine", "layer_offload"):
        pytest.skip("layer_offload engine not registered")
    cfg = _tiny_cfg(
        engine={"name": "layer_offload", "resident_layers": 1, "prefetch": 1},
    )
    rpt = estimate(cfg)
    assert rpt.engine_name == "layer_offload"
    assert rpt.offload is not None
    assert rpt.offload.resident_layers == 1


def test_estimate_cli_smoke_json_output(tmp_path: Path):
    runner = CliRunner()
    cfg_text = json.dumps(_tiny_cfg())
    # Write a YAML-loadable cfg.
    import yaml

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_cfg()), encoding="utf-8")
    out_json = tmp_path / "estimate.json"
    result = runner.invoke(
        app, ["estimate", "-c", str(cfg_path), "--json", str(out_json)]
    )
    assert result.exit_code == 0, result.output
    assert "trainable_params" in result.output
    expect_exists(out_json, tmp_path, what="estimate.json")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "trainable_params" in data
    assert "engine_name" in data
    assert data["engine_name"] == "standard"
    _ = cfg_text  # silence unused
