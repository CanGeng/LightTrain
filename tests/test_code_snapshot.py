"""Code snapshot — DESIGN §21.3 (M5)."""

from __future__ import annotations

import zipfile

from lighttrain.utils.code_snapshot import capture_code_snapshot


def test_capture_creates_lighttrain_subtree(tmp_path):
    out = capture_code_snapshot(tmp_path)
    assert out == tmp_path / "code.snapshot"
    pkg_init = out / "lighttrain" / "__init__.py"
    assert pkg_init.exists(), f"{pkg_init} missing"
    # Spot-check a known M5 module is in the snapshot.
    surgery_init = out / "lighttrain" / "models" / "surgery" / "__init__.py"
    assert surgery_init.exists()
    # Pyc / __pycache__ excluded.
    for p in out.rglob("*"):
        assert "__pycache__" not in str(p)
        assert not str(p).endswith(".pyc")


def test_capture_copies_user_modules(tmp_path):
    user_mod = tmp_path / "my_ext.py"
    user_mod.write_text("# my custom extension\n", encoding="utf-8")
    out = capture_code_snapshot(tmp_path, user_modules=[str(user_mod)])
    copied = out / "user_modules" / "my_ext.py"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "# my custom extension\n"


def test_capture_is_idempotent(tmp_path):
    out1 = capture_code_snapshot(tmp_path)
    marker = out1 / "lighttrain" / "__init__.py"
    mtime1 = marker.stat().st_mtime_ns
    # Second call: should be a no-op (return existing snap_dir without
    # rewriting).
    out2 = capture_code_snapshot(tmp_path)
    assert out2 == out1
    assert marker.stat().st_mtime_ns == mtime1


def test_frozen_step_pointer_uses_code_snapshot_when_present(tmp_path):
    """When ``run_dir/code.snapshot/`` exists, the frozen_step zip's pointer
    file points there; otherwise it falls back to the run dir (M4 behavior).
    """
    import torch

    from lighttrain.diagnostics.frozen_step import FrozenStepWriter
    from lighttrain.models.adapters.tiny_lm import TinyCausalLM

    class _Ctx:
        epoch = 0

    model = TinyCausalLM(vocab_size=64, d_model=8, n_layers=1, n_heads=2,
                        max_seq_len=8)
    batch = {"input_ids": torch.randint(0, 64, (1, 4))}

    # Case A: no snapshot dir → pointer == run_dir (M4 fallback)
    writer = FrozenStepWriter(run_dir=tmp_path)
    writer.snapshot(step=1, ctx=_Ctx(), model=model, batch=batch, optimizer=None)
    p = writer.commit(reason="scheduled")
    with zipfile.ZipFile(p) as zf:
        ptr = zf.read("code_snapshot_pointer.txt").decode("utf-8").strip()
    assert ptr == str(tmp_path.resolve())

    # Case B: snapshot dir exists → pointer points there (M5 behavior)
    capture_code_snapshot(tmp_path)
    writer.snapshot(step=2, ctx=_Ctx(), model=model, batch=batch, optimizer=None)
    p2 = writer.commit(reason="scheduled")
    with zipfile.ZipFile(p2) as zf:
        ptr2 = zf.read("code_snapshot_pointer.txt").decode("utf-8").strip()
    assert ptr2 == str((tmp_path / "code.snapshot").resolve())
