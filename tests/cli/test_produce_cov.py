"""Tests for ``lighttrain.cli._produce`` — run_produce and _iter_dataset.

Strategy
--------
* All tests for ``run_produce`` monkeypatch the heavy operations
  (load_config, _build_model, _build_data, make_run_dir, LineageStore, _resolve)
  so no GPU or real network is required.
* ``LineageStore`` and ``resolve`` are lazy-imported inside run_produce, so they
  are patched at their *source* module paths:
    - ``lighttrain.observability.lineage.store.LineageStore``
    - ``lighttrain.config._resolver.resolve``
* ``_iter_dataset`` is a pure Python helper tested directly with synthetic
  dataset objects.
* CLI surface tested via ``typer.testing.CliRunner`` against
  ``produce-artifact``; the command imports ``run_produce`` lazily inside the
  handler, so we patch ``lighttrain.cli._produce.run_produce``.

Skipped branches
----------------
* Lines that depend on ``torch.cuda.is_available()`` returning True (GPU branch
  of ``device = torch.device(...)``): not achievable in CI without a GPU.
* Branches inside ``producer.produce`` that require actual model inference:
  skipped by stub.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.cli._produce import _iter_dataset, run_produce
from lighttrain.config import RootConfig

# Canonical patch targets for the two lazy imports inside run_produce
_LINEAGE_STORE_TARGET = "lighttrain.observability.lineage.store.LineageStore"
_RESOLVE_TARGET = "lighttrain.config._resolver.resolve"


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_root_config(tmp_path: Path, extra: dict | None = None) -> RootConfig:
    """Build a minimal RootConfig; ``extra`` keys are merged in."""
    data: dict[str, Any] = {
        "mode": "lab",
        "seed": 7,
        "exp": "produce_test",
        "run_root": str(tmp_path / "runs"),
        "data": {"name": "simple", "dataset": {"name": "dummy"}},
    }
    if extra:
        data.update(extra)
    return RootConfig.model_validate(data)


def _write_minimal_recipe(tmp_path: Path, extra_yaml: str = "") -> Path:
    """Write a minimal YAML recipe to disk and return its path."""
    p = tmp_path / "recipe.yaml"
    p.write_text(
        textwrap.dedent(
            f"""
            mode: lab
            seed: 7
            exp: produce_test
            run_root: {tmp_path / "runs"}
            data:
              name: simple
              dataset:
                name: dummy
            {extra_yaml}
            """
        ).strip(),
        encoding="utf-8",
    )
    return p


def _cfg_with_artifacts(tmp_path: Path, store_root: str | None = None) -> RootConfig:
    """RootConfig that has both producer and store sections."""
    root = store_root or str(tmp_path / "store")
    return _make_root_config(
        tmp_path,
        extra={
            "artifacts": {
                "producer": {"name": "model_forward"},
                "store": {"name": "safetensors-shards", "root": root},
            }
        },
    )


class _FakeDataModule:
    """Minimal data module with a dataset attribute."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset


class _IndexableDataset:
    """Dataset with __len__ and __getitem__ (indexed path)."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> Any:
        return self._samples[i]


class _IterableOnlyDataset:
    """Dataset that has __iter__ but NOT __getitem__ — pure iterable path."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)


class _NoLenIterDataset:
    """Dataset with __iter__ + __getitem__ but no __len__ — no-len fallback."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, i: int) -> Any:
        return self._samples[i]


class _FakeProducer:
    """Stub artifact producer — records calls; finalize() returns manifest."""

    def __init__(self, manifest: Path) -> None:
        self._manifest = manifest
        self.prepared: dict = {}
        self.produced: list = []
        self.finalized = False

    def prepare(self, cfg: dict) -> None:
        self.prepared = dict(cfg)

    def produce(self, sample: Any) -> dict:
        self.produced.append(sample)
        return {}

    def finalize(self) -> Path:
        self.finalized = True
        self._manifest.touch()
        return self._manifest


def _setup_module_stubs(
    monkeypatch,
    *,
    cfg: RootConfig,
    fake_model: Any,
    fake_data_module: Any,
    run_dir: Path,
) -> None:
    """Apply monkeypatches for all module-level names in _produce that are
    imported at the top level (so they ARE attributes of the module)."""
    import lighttrain.cli._produce as _mod

    monkeypatch.setattr(_mod, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(_mod, "seed_everything", lambda s: None)
    monkeypatch.setattr(_mod, "_build_model", lambda c: fake_model)
    monkeypatch.setattr(_mod, "_build_data", lambda c, **kw: fake_data_module)
    monkeypatch.setattr(_mod, "make_run_dir", lambda *a, **kw: run_dir)


# ---------------------------------------------------------------------------
# _iter_dataset — exhaustive branch coverage
# ---------------------------------------------------------------------------


class TestIterDataset:
    def test_iterable_only_yields_all(self):
        """Pure-iterable path (has __iter__, no __getitem__): yields every sample."""
        samples = [{"id": i} for i in range(5)]
        ds = _IterableOnlyDataset(samples)
        result = list(_iter_dataset(ds))
        assert result == samples

    def test_indexable_with_len_yields_all(self):
        """Indexed path (has __len__ + __getitem__): iterates via range(len(ds))."""
        samples = [{"id": i} for i in range(3)]
        ds = _IndexableDataset(samples)
        result = list(_iter_dataset(ds))
        assert result == samples

    def test_indexable_with_len_yields_empty_when_empty(self):
        """Indexed path with zero-length dataset must yield nothing."""
        ds = _IndexableDataset([])
        assert list(_iter_dataset(ds)) == []

    def test_no_len_falls_back_to_iter(self):
        """When __len__ is absent but both __iter__ and __getitem__ exist,
        fallback iterates via ``for sample in dataset``."""
        samples = [{"x": 1}, {"x": 2}]
        ds = _NoLenIterDataset(samples)
        result = list(_iter_dataset(ds))
        assert result == samples

    def test_iterable_only_empty(self):
        """Pure-iterable empty dataset: no samples yielded, no errors."""
        ds = _IterableOnlyDataset([])
        assert list(_iter_dataset(ds)) == []

    def test_indexable_single_sample(self):
        """Edge: single-element indexed dataset."""
        ds = _IndexableDataset([{"val": 99}])
        assert list(_iter_dataset(ds)) == [{"val": 99}]


# ---------------------------------------------------------------------------
# run_produce — success path
# ---------------------------------------------------------------------------


def test_run_produce_success_full_iteration(tmp_path: Path, monkeypatch):
    """Happy path: run_produce iterates all samples, calls produce per sample,
    calls finalize(), and returns the manifest path."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    samples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
    fake_dataset = _IndexableDataset(samples)
    fake_data_module = _FakeDataModule(fake_dataset)

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    producer = _FakeProducer(manifest_path)

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=fake_data_module, run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer) as mock_resolve,
    ):
        result = run_produce(recipe)

    assert result == manifest_path
    assert producer.finalized
    assert len(producer.produced) == 2
    assert mock_resolve.call_args[1]["category"] == "artifact_producer"


def test_run_produce_estimate_mode_with_len(tmp_path: Path, monkeypatch):
    """estimate=True with a sizeable dataset: returns run_dir (not manifest),
    does NOT call finalize(), and prints an estimate line via console."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    fake_dataset = _IndexableDataset([{"x": i} for i in range(10)])
    fake_data_module = _FakeDataModule(fake_dataset)

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-est"
    run_dir.mkdir(parents=True)
    producer = _FakeProducer(run_dir / "manifest.json")

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1
    console = MagicMock()

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=fake_data_module, run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        result = run_produce(recipe, estimate=True, console=console)

    assert result == run_dir
    assert not producer.finalized
    console.print.assert_called()


def test_run_produce_estimate_mode_no_len(tmp_path: Path, monkeypatch):
    """estimate=True on a dataset with no __len__: covers the TypeError branch
    that prints 'iterating to count'."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    fake_dataset = _IterableOnlyDataset([{"x": 0}])
    fake_data_module = _FakeDataModule(fake_dataset)

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-nolen"
    run_dir.mkdir(parents=True)
    producer = _FakeProducer(run_dir / "manifest.json")

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1
    console = MagicMock()

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=fake_data_module, run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        result = run_produce(recipe, estimate=True, console=console)

    assert result == run_dir
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "count" in printed.lower() or "estimate" in printed.lower()


def test_run_produce_console_prints_run_dir_and_progress(tmp_path: Path, monkeypatch):
    """Console prints run_dir at start, progress at every 100 samples, and
    the final 'produced N samples' message."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    # 101 samples so progress fires once at n_seen == 100
    samples = [{"x": i} for i in range(101)]
    fake_dataset = _IndexableDataset(samples)
    fake_data_module = _FakeDataModule(fake_dataset)

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-console"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    producer = _FakeProducer(manifest)

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1
    console = MagicMock()

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=fake_data_module, run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        result = run_produce(recipe, console=console)

    assert result == manifest
    # At minimum: run_dir print + progress at 100 + final produced
    assert console.print.call_count >= 3
    printed = " ".join(str(c) for c in console.print.call_args_list)
    assert "produce" in printed.lower()


# ---------------------------------------------------------------------------
# run_produce — error paths
# ---------------------------------------------------------------------------


def test_run_produce_raises_when_artifacts_section_missing(tmp_path: Path, monkeypatch):
    """RuntimeError raised when recipe has no ``artifacts:`` section (lines 44-45)."""
    import lighttrain.cli._produce as _mod

    cfg = _make_root_config(tmp_path)  # no artifacts
    monkeypatch.setattr(_mod, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(_mod, "seed_everything", lambda s: None)

    recipe = _write_minimal_recipe(tmp_path)
    with pytest.raises(RuntimeError, match="artifacts"):
        run_produce(recipe)


def test_run_produce_raises_when_store_missing(tmp_path: Path, monkeypatch):
    """RuntimeError raised when ``artifacts.store`` is absent (lines 50-51)."""
    import lighttrain.cli._produce as _mod

    cfg = _make_root_config(
        tmp_path,
        extra={"artifacts": {"producer": {"name": "model_forward"}}},  # no store
    )
    monkeypatch.setattr(_mod, "load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr(_mod, "seed_everything", lambda s: None)

    recipe = _write_minimal_recipe(tmp_path)
    with pytest.raises(RuntimeError, match="store"):
        run_produce(recipe)


def test_run_produce_raises_when_data_module_has_no_dataset(tmp_path: Path, monkeypatch):
    """RuntimeError raised when data module has no ``dataset`` attribute (lines 67-69)."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    class _NoDatasetModule:
        pass

    run_dir = tmp_path / "runs" / "fake-run-nodataset"
    run_dir.mkdir(parents=True)

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=_NoDatasetModule(), run_dir=run_dir,
    )

    with pytest.raises(RuntimeError, match="dataset"):
        run_produce(recipe)


# ---------------------------------------------------------------------------
# run_produce — contract / pinning tests
# ---------------------------------------------------------------------------


def test_run_produce_producer_name_defaults_to_model_forward(tmp_path: Path, monkeypatch):
    """When ``artifacts.producer`` has no ``name``, defaults to ``model_forward``
    (line 49 setdefault)."""
    cfg = _make_root_config(
        tmp_path,
        extra={
            "artifacts": {
                "producer": {},  # no name
                "store": {"name": "safetensors-shards", "root": str(tmp_path / "store")},
            }
        },
    )
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-default"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    producer = _FakeProducer(manifest)

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=_FakeDataModule(_IndexableDataset([])), run_dir=run_dir,
    )

    captured_spec: dict = {}

    def _fake_resolve(spec, *, category):
        captured_spec.update(spec)
        return producer

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, side_effect=_fake_resolve),
    ):
        run_produce(recipe)

    assert captured_spec.get("name") == "model_forward"


def test_run_produce_with_overrides_passed_to_load_config(tmp_path: Path, monkeypatch):
    """Overrides list is forwarded verbatim to load_config (line 39)."""
    import lighttrain.cli._produce as _mod

    captured_overrides: list = []

    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-overrides"
    run_dir.mkdir(parents=True)
    manifest = run_dir / "manifest.json"
    producer = _FakeProducer(manifest)

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1

    def _fake_load(config, *, overrides=None, **kw):
        captured_overrides.extend(overrides or [])
        return cfg

    recipe = _write_minimal_recipe(tmp_path)
    monkeypatch.setattr(_mod, "load_config", _fake_load)
    monkeypatch.setattr(_mod, "seed_everything", lambda s: None)
    monkeypatch.setattr(_mod, "_build_model", lambda c: fake_model)
    monkeypatch.setattr(_mod, "_build_data",
                        lambda c, **kw: _FakeDataModule(_IndexableDataset([])))
    monkeypatch.setattr(_mod, "make_run_dir", lambda *a, **kw: run_dir)

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        run_produce(recipe, overrides=["++seed=99"])

    assert "++seed=99" in captured_overrides


def test_run_produce_lineage_upsert_called_with_run_kind(tmp_path: Path, monkeypatch):
    """lineage_store.upsert_node is called with kind='run' (lines 75-79)."""
    cfg = _cfg_with_artifacts(tmp_path)
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-lineage"
    run_dir.mkdir(parents=True)
    producer = _FakeProducer(run_dir / "manifest.json")

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 42

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=_FakeDataModule(_IndexableDataset([])), run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        run_produce(recipe)

    call_kwargs = fake_lineage.upsert_node.call_args
    assert call_kwargs is not None
    # 'run' appears as a positional or keyword argument
    all_values = list(call_kwargs.args) + list(call_kwargs.kwargs.values())
    assert "run" in all_values, f"'run' not in upsert_node call: {call_kwargs}"


def test_run_produce_producer_prepare_called_with_lineage_store(tmp_path: Path, monkeypatch):
    """producer.prepare() receives lineage_store and run_node_id (lines 90-94)."""
    cfg = _make_root_config(
        tmp_path,
        extra={
            "artifacts": {
                "producer": {"name": "model_forward"},
                "store": {
                    "name": "safetensors-shards",
                    "root": str(tmp_path / "store"),
                    "artifact_name": "my_artifact",
                },
            }
        },
    )
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-prepare"
    run_dir.mkdir(parents=True)
    producer = _FakeProducer(run_dir / "manifest.json")

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 7

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=_FakeDataModule(_IndexableDataset([])), run_dir=run_dir,
    )

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, return_value=producer),
    ):
        run_produce(recipe)

    assert "lineage_store" in producer.prepared
    assert producer.prepared["run_node_id"] == 7
    assert producer.prepared["artifact_name"] == "my_artifact"


def test_pin_current_behavior_producer_spec_gets_model_and_store(tmp_path: Path, monkeypatch):
    """Pin: the spec passed to _resolve always contains ``model`` and ``store``
    (lines 84-87 inject them). Callers rely on this contract."""
    cfg = _make_root_config(
        tmp_path,
        extra={
            "artifacts": {
                "producer": {"name": "model_forward", "extra_kwarg": True},
                "store": {"name": "safetensors-shards", "root": str(tmp_path / "store")},
            }
        },
    )
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    run_dir = tmp_path / "runs" / "produce_test" / "fake-run-spec"
    run_dir.mkdir(parents=True)
    producer = _FakeProducer(run_dir / "manifest.json")

    fake_lineage = MagicMock()
    fake_lineage.upsert_node.return_value = 1

    recipe = _write_minimal_recipe(tmp_path)
    _setup_module_stubs(
        monkeypatch, cfg=cfg, fake_model=fake_model,
        fake_data_module=_FakeDataModule(_IndexableDataset([])), run_dir=run_dir,
    )

    captured: dict = {}

    def _fake_resolve(spec, *, category):
        captured.update(spec)
        return producer

    with (
        patch(_LINEAGE_STORE_TARGET, return_value=fake_lineage),
        patch(_RESOLVE_TARGET, side_effect=_fake_resolve),
    ):
        run_produce(recipe)

    assert "model" in captured
    assert captured["model"] is fake_model
    assert "store" in captured


# ---------------------------------------------------------------------------
# CLI surface — produce-artifact command
# ---------------------------------------------------------------------------

# The command does `from lighttrain.cli._produce import run_produce` inside the
# handler each time it is called, so patching `lighttrain.cli._produce.run_produce`
# intercepts the lazy binding correctly.
_RUN_PRODUCE_TARGET = "lighttrain.cli._produce.run_produce"


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner per test."""
    return CliRunner()


def test_produce_artifact_missing_config_exits_one(runner, tmp_path):
    """produce-artifact with a missing config file exits non-zero (file-not-found)."""
    missing = tmp_path / "no_such_file.yaml"
    res = runner.invoke(app, ["produce-artifact", "-c", str(missing)])
    assert res.exit_code != 0


def test_produce_artifact_config_with_missing_artifacts_section_exits_one(runner, tmp_path):
    """produce-artifact exits 1 when recipe has no ``artifacts:`` section."""
    recipe = _write_minimal_recipe(tmp_path)
    res = runner.invoke(app, ["produce-artifact", "-c", str(recipe)])
    assert res.exit_code == 1
    assert "error" in res.stdout.lower()


def test_produce_artifact_success_path(runner, tmp_path):
    """produce-artifact exits 0 and prints manifest path on success."""
    manifest = tmp_path / "manifest.json"
    manifest.touch()

    recipe = _write_minimal_recipe(tmp_path)
    with patch(_RUN_PRODUCE_TARGET, return_value=manifest):
        res = runner.invoke(app, ["produce-artifact", "-c", str(recipe)])

    assert res.exit_code == 0, res.stdout
    assert "artifact finalized" in res.stdout.lower() or str(manifest) in res.stdout


def test_produce_artifact_runtime_error_exits_one(runner, tmp_path):
    """produce-artifact exits 1 when run_produce raises RuntimeError."""
    recipe = _write_minimal_recipe(tmp_path)
    with patch(_RUN_PRODUCE_TARGET, side_effect=RuntimeError("boom")):
        res = runner.invoke(app, ["produce-artifact", "-c", str(recipe)])

    assert res.exit_code == 1
    assert "error" in res.stdout.lower()


def test_produce_artifact_file_not_found_exits_one(runner, tmp_path):
    """produce-artifact exits 1 when run_produce raises FileNotFoundError."""
    recipe = _write_minimal_recipe(tmp_path)
    with patch(_RUN_PRODUCE_TARGET, side_effect=FileNotFoundError("no file")):
        res = runner.invoke(app, ["produce-artifact", "-c", str(recipe)])

    assert res.exit_code == 1
    assert "error" in res.stdout.lower()


def test_produce_artifact_estimate_flag_forwarded(runner, tmp_path):
    """--estimate flag is passed as estimate=True to run_produce."""
    manifest = tmp_path / "manifest.json"
    manifest.touch()
    captured: dict = {}

    def _fake(config, *, overrides=None, estimate=False, console=None):
        captured["estimate"] = estimate
        return manifest

    recipe = _write_minimal_recipe(tmp_path)
    with patch(_RUN_PRODUCE_TARGET, side_effect=_fake):
        res = runner.invoke(app, ["produce-artifact", "-c", str(recipe), "--estimate"])

    assert res.exit_code == 0, res.stdout
    assert captured.get("estimate") is True


def test_produce_artifact_overrides_forwarded(runner, tmp_path):
    """Positional overrides are forwarded to run_produce."""
    manifest = tmp_path / "manifest.json"
    manifest.touch()
    captured: dict = {}

    def _fake(config, *, overrides=None, estimate=False, console=None):
        captured["overrides"] = list(overrides or [])
        return manifest

    recipe = _write_minimal_recipe(tmp_path)
    with patch(_RUN_PRODUCE_TARGET, side_effect=_fake):
        res = runner.invoke(
            app, ["produce-artifact", "-c", str(recipe), "++seed=123"]
        )

    assert res.exit_code == 0, res.stdout
    assert "++seed=123" in captured.get("overrides", [])


# ---------------------------------------------------------------------------
# Invariant: __all__ exports run_produce
# ---------------------------------------------------------------------------


def test_invariant_all_exports_run_produce():
    """``__all__`` must contain ``run_produce`` — public API contract (line 133)."""
    from lighttrain.cli import _produce as _mod

    assert "run_produce" in _mod.__all__
