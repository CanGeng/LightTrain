"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lighttrain.protocols import ModelOutput
from lighttrain.registry import get_registry


@pytest.fixture(autouse=True)
def _isolate_os_environ():
    """Snapshot ``os.environ`` and restore it after every test.

    CLI tests run ``lighttrain`` in-process (via ``CliRunner``), and the CLI
    loads a repo-local ``.env`` into ``os.environ`` *permanently*
    (``load_dotenv_if_present``). Vars such as
    ``LIGHTTRAIN_CODE_SNAPSHOT_MODE`` would otherwise leak into later tests and
    cause order-dependent failures (e.g. the code-snapshot cas/archive flip).
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the adversarial suite."""
    config.addinivalue_line(
        "markers",
        "slow: tests that spawn multiple processes or otherwise take >1s "
        "wall-clock; skip with `-m 'not slow'`",
    )


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Scratch run dir shaped like a real lighttrain run (has ``checkpoints/``)."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    return run


@pytest.fixture
def lineage_store_factory(tmp_path: Path):
    """Factory yielding fresh ``LineageStore`` instances; closed at teardown.

    Use ``store = lineage_store_factory()`` to get a clean SQLite-backed store.
    """
    from lighttrain.lineage import LineageStore

    stores: list[LineageStore] = []

    def _make(name: str = "lineage.sqlite") -> LineageStore:
        s = LineageStore(tmp_path / name)
        stores.append(s)
        return s

    try:
        yield _make
    finally:
        for s in stores:
            try:
                s.close()
            except Exception:  # noqa: BLE001 — best-effort
                pass


@pytest.fixture
def clean_registry():
    """Snapshot the global registry, yield, then restore.

    Tests can register dummies freely; this fixture isolates them.
    """
    reg = get_registry()
    snap = reg.snapshot()
    try:
        yield reg
    finally:
        reg.restore(snap)


@pytest.fixture
def tmp_yaml(tmp_path: Path):
    """Factory: write a YAML file at tmp_path/<name> and return its path."""

    def _write(content: str, name: str = "cfg.yaml") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return _write


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """A scratch directory for multi-file YAML composition."""
    d = tmp_path / "configs"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Shared fixtures for engine/trainers/update_rules tests
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal causal-LM: embed → linear head. Deterministic given a seed."""

    def __init__(self, vocab: int = 8, dim: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **_):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.head(h)})


@pytest.fixture
def tiny_model() -> nn.Module:
    """Deterministic tiny causal-LM (vocab=8, dim=4). Seeded init each call."""
    torch.manual_seed(0)
    return _TinyLM(vocab=8, dim=4)


@pytest.fixture
def tiny_optimizer(tiny_model: nn.Module) -> torch.optim.Optimizer:
    """SGD lr=1e-2 over tiny_model's params. Deterministic."""
    return torch.optim.SGD(tiny_model.parameters(), lr=1e-2)


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    """A (B=2, T=4, V=8) batch with input_ids/attention_mask/labels."""
    torch.manual_seed(1)
    B, T, V = 2, 4, 8
    return {
        "input_ids": torch.randint(0, V, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "labels": torch.randint(0, V, (B, T)),
    }


class _FakeGradSync:
    """Records every call so tests can assert on temporal order + arg shapes.

    Mirrors the GradSyncStrategy surface used by Standard/RL update rules:
        - accumulate(model) ⇒ context manager (records enter/exit)
        - backward(loss, model)
        - clip_grad_norm(model, max_norm, parallel_ctx) → float
        - optimizer_step(optimizer, model)

    The recorded ``calls`` is a list of ``(name, kwargs_dict)`` tuples so
    tests can pin both order and key arguments.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.grad_norm_value: float = 0.0

    def accumulate(self, model):
        outer = self

        class _AccumCtx:
            def __enter__(self_inner):
                outer.calls.append(("accumulate_enter", {"model": id(model)}))
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                outer.calls.append(("accumulate_exit", {}))
                return False

        return _AccumCtx()

    def backward(self, loss, model):
        self.calls.append(("backward", {"loss_id": id(loss), "model_id": id(model)}))
        loss.backward()

    def clip_grad_norm(self, model, max_norm, parallel_ctx):
        self.calls.append(
            ("clip_grad_norm", {"max_norm": float(max_norm), "model_id": id(model)})
        )
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))
        self.grad_norm_value = gn
        return gn

    def optimizer_step(self, optimizer, model):
        self.calls.append(("optimizer_step", {"model_id": id(model)}))
        optimizer.step()


class _FakeAccelerator:
    """Minimal accelerator with autocast/backward/clip surface — records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def autocast(self):
        from contextlib import nullcontext

        self.calls.append(("autocast_enter", {}))
        return nullcontext()

    def backward(self, loss):
        self.calls.append(("backward", {"loss_id": id(loss)}))
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        self.calls.append(("clip_grad_norm_", {"max_norm": float(max_norm)}))
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))


@pytest.fixture
def fake_dist_env():
    """Factory: returns (grad_sync, parallel_ctx) MagicMock pair.

    The grad_sync records every call so tests can assert order. parallel_ctx
    is a SimpleNamespace mimicking the single-GPU defaults — tests that care
    about specific attrs should set them explicitly.
    """
    from types import SimpleNamespace

    def _factory():
        gs = _FakeGradSync()
        pctx = SimpleNamespace(is_main_process=True, world_size=1, rank=0)
        return gs, pctx

    return _factory


@pytest.fixture
def fake_accelerator() -> _FakeAccelerator:
    """A fresh recording accelerator stub."""
    return _FakeAccelerator()
