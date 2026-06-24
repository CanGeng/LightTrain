"""Structural guard: every recipe-eating command imports ``user_modules``.

Root cause A was "coverage depends on each command remembering to call import".
The fix folds the import into the single ``load_config`` chokepoint that all
recipe-eating commands flow through (plus two library bypass guards). These tests
pin that property so a future command can't silently regress ISSUE-1 again.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from lighttrain.config import load_config
from lighttrain.config._user_modules import _IMPORTED_USER_MODULES
from lighttrain.registry import contains, unregister

_CUSTOM_NODE_SRC = textwrap.dedent(
    """
    from lighttrain.registry import register
    from lighttrain.data.prepgraph.node import NodeResult, PrepNode

    @register("prep_node", "_chokepoint_pack")
    class ChokepointPackNode(PrepNode):
        kind = "_chokepoint_pack"
        schema_kind = "packed_rows"
        def run(self, ctx):
            rows = list(ctx.upstream[self.inputs[0]].rows or [])
            return NodeResult(fingerprint="", schema_kind=self.schema_kind,
                              rows=rows, extras={"row_count": len(rows)})
    """
)

_NODE_NAME = ("prep_node", "_chokepoint_pack")


@pytest.fixture
def custom_node_recipe(tmp_path: Path):
    """A recipe wiring a custom prep_node the *documented* way: ``user_modules:``
    + ``kind: <name>`` (no ``_target_``). Yields the recipe path."""
    modfile = tmp_path / "chokepoint_mod.py"
    modfile.write_text(_CUSTOM_NODE_SRC, encoding="utf-8")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nsecond document here\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        textwrap.dedent(
            f"""
            exp: chokepoint
            seed: 0
            run_root: {tmp_path / "runs"}
            user_modules: [{modfile}]
            prep_graph:
              nodes:
                - {{name: raw, kind: load, source: "lines:{corpus}"}}
                - {{name: tok, kind: tokenize, inputs: [raw], tokenizer: {{name: byte}}, text_field: text}}
                - {{name: packed, kind: _chokepoint_pack, inputs: [tok]}}
              terminals: [packed]
            """
        ),
        encoding="utf-8",
    )
    # Ensure a clean slate: the module may have been imported by a prior test.
    _IMPORTED_USER_MODULES.discard(str(modfile.resolve()))
    if contains(*_NODE_NAME):
        unregister(*_NODE_NAME)
    yield recipe
    if contains(*_NODE_NAME):
        unregister(*_NODE_NAME)
    _IMPORTED_USER_MODULES.discard(str(modfile.resolve()))


def test_load_config_chokepoint_imports_user_modules(custom_node_recipe):
    """The chokepoint itself registers the custom component on validate."""
    assert not contains(*_NODE_NAME)
    load_config(custom_node_recipe)
    assert contains(*_NODE_NAME), "load_config must import cfg.user_modules"


def test_load_config_import_flag_false_skips(custom_node_recipe):
    """The ``--print-config`` escape hatch must NOT trigger plugin imports."""
    assert not contains(*_NODE_NAME)
    load_config(custom_node_recipe, import_user_modules=False)
    assert not contains(*_NODE_NAME)


def test_build_prep_runner_imports_user_modules(custom_node_recipe):
    """The prep family (which regressed as ISSUE-1) resolves the custom node.

    ``build_prep_runner`` goes through ``load_config`` → the node must resolve
    without a NotRegisteredError, the historical failure.
    """
    from lighttrain.cli._runtime import build_prep_runner

    bundle = build_prep_runner(custom_node_recipe)
    assert "packed" in bundle["graph"].nodes
    assert contains(*_NODE_NAME)


@pytest.mark.parametrize(
    "argv, expect_zero",
    [
        (["prep", "-c"], True),
        (["prep", "-c", "--dry-run"], True),
        (["prep-status", "-c"], True),
        (["prep-status", "-c", "--extras"], True),
        (["prep-graph", "-c"], True),
        (["prep-clean", "-c", "--orphans"], True),
    ],
    ids=lambda a: "+".join(str(x) for x in a) if isinstance(a, list) else str(a),
)
def test_prep_family_cli_resolves_custom_node(custom_node_recipe, argv, expect_zero):
    """Forget-proof guard across the whole prep CLI family: each command must
    import ``user_modules`` and resolve the custom node registered the documented
    way (no ``_target_``). Regressing ISSUE-1 on any of them fails this test."""
    from typer.testing import CliRunner

    from lighttrain.cli._app import app

    full: list[str] = []
    for tok in argv:
        full.append(tok)
        if tok == "-c":
            full.append(str(custom_node_recipe))

    res = CliRunner().invoke(app, full)
    assert "NotRegisteredError" not in res.stdout
    assert "is not registered" not in res.stdout
    # The command imported the recipe's user_modules → the node is registered.
    assert contains(*_NODE_NAME), res.stdout
    if expect_zero:
        assert res.exit_code == 0, res.stdout
