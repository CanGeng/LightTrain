"""Code snapshot — DESIGN §21.3 (M5)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from lighttrain.utils.code_snapshot import capture_code_snapshot


def _manifest(snapshot_dir: Path) -> dict:
    return json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))


def _entry(manifest: dict, path: str) -> dict:
    for item in manifest["files"]:
        if item["path"] == path:
            return item
    raise AssertionError(f"{path} missing from manifest")


def test_capture_creates_cas_manifest_and_blobs(tmp_path, monkeypatch):
    monkeypatch.delenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", raising=False)
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", str(tmp_path / "store"))

    out = capture_code_snapshot(tmp_path)
    assert out == tmp_path / "code.snapshot"
    manifest = _manifest(out)
    assert manifest["mode"] == "cas"
    assert not (out / "lighttrain").exists()

    pkg_init = _entry(manifest, "lighttrain/__init__.py")
    blob = (
        Path(manifest["store_dir"])
        / "blobs"
        / pkg_init["sha256"][:2]
        / pkg_init["sha256"]
    )
    assert blob.exists()

    # Spot-check a known module is represented without copying the tree.
    _entry(manifest, "lighttrain/models/surgery/__init__.py")

    for item in manifest["files"]:
        assert "__pycache__" not in item["path"]
        assert not item["path"].endswith(".pyc")


def test_capture_records_user_modules_in_cas(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", str(tmp_path / "store"))
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("# package\n", encoding="utf-8")
    user_mod = tmp_path / "my_ext.py"
    user_mod.write_text("# my custom extension\n", encoding="utf-8")

    out = capture_code_snapshot(
        tmp_path, package_root=package_root, user_modules=[str(user_mod)]
    )
    manifest = _manifest(out)
    copied = _entry(manifest, "user_modules/my_ext.py")
    blob = (
        Path(manifest["store_dir"])
        / "blobs"
        / copied["sha256"][:2]
        / copied["sha256"]
    )
    assert blob.read_text(encoding="utf-8") == "# my custom extension\n"


def test_capture_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", str(tmp_path / "store"))
    out1 = capture_code_snapshot(tmp_path)
    marker = out1 / "manifest.json"
    mtime1 = marker.stat().st_mtime_ns
    # Second call: should be a no-op (return existing snap_dir without
    # rewriting).
    out2 = capture_code_snapshot(tmp_path)
    assert out2 == out1
    assert marker.stat().st_mtime_ns == mtime1


def test_capture_archive_mode_writes_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "archive")
    package_root = tmp_path / "pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("# package\n", encoding="utf-8")
    (package_root / "core.py").write_text("VALUE = 1\n", encoding="utf-8")

    out = capture_code_snapshot(tmp_path, package_root=package_root)
    manifest = _manifest(out)
    assert manifest["mode"] == "archive"
    assert manifest["archive"] == "code.zip"
    assert "store_dir" not in manifest
    with zipfile.ZipFile(out / "code.zip") as zf:
        assert zf.read("lighttrain/core.py").decode("utf-8") == "VALUE = 1\n"


def test_capture_off_mode_skips_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_MODE", "off")

    out = capture_code_snapshot(tmp_path)
    assert out == tmp_path
    assert not (tmp_path / "code.snapshot").exists()


def test_frozen_step_pointer_uses_code_snapshot_when_present(tmp_path, monkeypatch):
    """When ``run_dir/code.snapshot/`` exists, the frozen_step zip's pointer
    file points there; otherwise it falls back to the run dir (M4 behavior).
    """
    import torch

    from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
    from lighttrain.observability.diagnostics.frozen_step import FrozenStepWriter

    monkeypatch.setenv("LIGHTTRAIN_CODE_SNAPSHOT_STORE_DIR", str(tmp_path / "store"))

    class _Ctx:
        epoch = 0

    model = TinyCausalLM(
        vocab_size=64, d_model=8, n_layers=1, n_heads=2, max_seq_len=8
    )
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
