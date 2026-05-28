"""`lighttrain estimate` + lab.estimate API — DESIGN §20.10 (M5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.cli._runtime import _eager_import_components
from lighttrain.lab.estimate import EstimateReport, estimate

_eager_import_components()


def _tiny_cfg(**overrides):
    base = {
        "mode": "lab",
        "seed": 0,
        "exp": "estimate_smoke",
        "run_root": "runs",
        "model": {
            "name": "tiny_lm",
            "vocab_size": 64,
            "d_model": 16,
            "n_layers": 2,
            "n_heads": 4,
            "max_seq_len": 32,
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


def test_estimate_with_lora_reports_low_trainable_ratio():
    pytest.importorskip("peft")
    cfg = _tiny_cfg(
        model={
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
        },
    )
    rpt = estimate(cfg)
    assert rpt.trainable_ratio < 0.10
    assert rpt.model_name == "lora"


def test_estimate_layer_offload_attaches_offload_block():
    # frontier_plugins/layer_offload registers `layer_offload` engine; if the
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
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "trainable_params" in data
    assert "engine_name" in data
    assert data["engine_name"] == "standard"
    _ = cfg_text  # silence unused
