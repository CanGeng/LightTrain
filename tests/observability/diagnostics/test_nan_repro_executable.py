"""F1 continued — ``python repro.py`` runs in a subprocess and exits."""

from __future__ import annotations

import subprocess
import sys

import torch

from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.observability.diagnostics.nan_repro import write_nan_repro
from tests._diagnostics import expect_exists


def test_repro_script_executes(tmp_path):
    model = TinyCausalLM(vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=8)
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("nan"))
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    diag = write_nan_repro(
        tmp_path,
        step=42,
        model=model,
        batch=batch,
        exception=RuntimeError("NaN/Inf detected in module 'tok_emb' at step 42"),
        module_name="tok_emb",
    )
    repro_py = diag / "repro.py"
    expect_exists(repro_py, diag, what="repro.py")
    # Run the script in a subprocess. It should *finish* (the anomaly detection
    # may print findings or raise; we only require exit cleanly *or* with a
    # RuntimeError mentioning anomaly/nan — both are acceptable per DESIGN §18.3).
    proc = subprocess.run(
        [sys.executable, str(repro_py)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(diag),
    )
    combined = (proc.stdout + proc.stderr).lower()
    # Either the script ran cleanly and printed something, or it raised a
    # nan/anomaly-related error (also a successful "reproduction").
    assert proc.returncode in (0, 1), proc.stderr
    assert (
        "loss" in combined
        or "nan" in combined
        or "anomaly" in combined
        or "non-finite" in combined
    ), f"unexpected output:\n{proc.stdout}\n---\n{proc.stderr}"
