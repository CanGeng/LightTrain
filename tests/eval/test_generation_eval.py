"""Edge-case tests for ``lighttrain.eval.generation_eval``.

Covers ``GenerationEvalTask.run`` + ``_write_lineage_edge`` end to end with
lightweight stubs (model / tokenizer / judge) and a REAL ``LineageStore``
(via the ``lineage_store_factory`` fixture) so the lineage path is exercised
against the actual SQLite-backed graph, not a mock.

* **run()**: ``model.eval()`` called; response = only the newly generated
  tokens; ``extras_per_prompt=None`` defaults to empty dicts; the judge item is
  a 2-tuple without extras / 3-tuple with extras; empty prompts → mean 0;
  ``score`` coerced to float; ``device`` move branch; ``max_new_tokens`` /
  ``do_sample`` forwarded.
* **_write_lineage_edge()**: int artifact id used directly; plain-name ref
  rewritten to ``artifact:<name>:latest``; full ``kind:name:version`` ref used
  verbatim; unresolved ref → silent skip; ``step`` → ``step_<n>`` vs ``latest``
  version; store/artifact_id None → skip; write failure swallowed (run still
  returns); defensive ``store is None`` early return.
"""

from __future__ import annotations

import torch

from lighttrain.eval.generation_eval import GenerationEvalResult, GenerationEvalTask

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Encodes every prompt to a fixed length-3 id tensor; decode = comma join."""

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(i) for i in ids)


class _StubModel:
    """Appends a fixed block of ``new_tokens`` to the prompt on generate()."""

    def __init__(self, new_tokens=(7, 8)):
        self.new_tokens = list(new_tokens)
        self.eval_called = False
        self.generate_calls: list[tuple[int, bool]] = []

    def eval(self):
        self.eval_called = True

    def generate(self, *, input_ids, max_new_tokens, do_sample):
        self.generate_calls.append((max_new_tokens, do_sample))
        new = torch.tensor([self.new_tokens], dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)


class _StubJudge:
    """Returns successive scores; records the items it was asked to score."""

    def __init__(self, scores=(1.0,)):
        self._scores = list(scores)
        self._i = 0
        self.items: list = []

    def score(self, items):
        self.items.extend(items)
        s = self._scores[min(self._i, len(self._scores) - 1)]
        self._i += 1
        return [s]


def _task(prompts, **kw):
    return GenerationEvalTask(_StubJudge(kw.pop("scores", (1.0,))), _StubTokenizer(), prompts, **kw)


# ---------------------------------------------------------------------------
# run(): core behavior
# ---------------------------------------------------------------------------

def test_invariant_run_returns_scored_results_and_mean():
    """``run`` returns task_name / mean_score / per-prompt results, and puts the
    model in eval mode."""
    model = _StubModel(new_tokens=[7, 8])
    judge = _StubJudge(scores=[0.5, 1.0])
    task = GenerationEvalTask(judge, _StubTokenizer(), ["p0", "p1"])
    out = task.run(model)

    assert model.eval_called is True
    assert out["task_name"] == "generation_eval"
    assert out["mean_score"] == 0.75  # mean(0.5, 1.0)
    assert [r.prompt for r in out["results"]] == ["p0", "p1"]
    assert all(isinstance(r, GenerationEvalResult) for r in out["results"])


def test_invariant_response_is_only_the_newly_generated_tokens():
    """Response decodes ``gen_ids[:, prompt_len:]`` — the prompt tokens are
    excluded."""
    task = _task(["hello"])
    out = task.run(_StubModel(new_tokens=[7, 8]))
    # prompt encodes to [1,2,3] (len 3); new tokens [7,8] → response "7,8".
    assert out["results"][0].response == "7,8"


def test_invariant_empty_prompts_gives_zero_mean_and_no_results():
    """No prompts → empty results, mean guarded to 0 (``max(1, len)``)."""
    out = _task([]).run(_StubModel())
    assert out["results"] == []
    assert out["mean_score"] == 0.0


def test_invariant_extras_none_defaults_and_judge_item_has_no_extras():
    """``extras_per_prompt=None`` → each result.extras == {}; the judge receives
    a 2-tuple (no extras appended)."""
    judge = _StubJudge(scores=[1.0])
    task = GenerationEvalTask(judge, _StubTokenizer(), ["p0"], extras_per_prompt=None)
    out = task.run(_StubModel())
    assert out["results"][0].extras == {}
    assert len(judge.items[0]) == 2  # (prompt, response), no extras


def test_invariant_extras_present_are_forwarded_to_judge_as_triple():
    """A non-empty extras dict → judge item is a 3-tuple and extras flow to the
    result."""
    judge = _StubJudge(scores=[1.0])
    task = GenerationEvalTask(
        judge, _StubTokenizer(), ["p0"], extras_per_prompt=[{"lang": "en"}]
    )
    out = task.run(_StubModel())
    assert len(judge.items[0]) == 3
    assert judge.items[0][2] == {"lang": "en"}
    assert out["results"][0].extras == {"lang": "en"}


def test_invariant_score_is_coerced_to_float():
    """An int judge score becomes a python float on the result."""
    judge = _StubJudge(scores=[1])  # int
    task = GenerationEvalTask(judge, _StubTokenizer(), ["p0"])
    out = task.run(_StubModel())
    assert out["results"][0].score == 1.0
    assert isinstance(out["results"][0].score, float)


def test_invariant_generate_receives_max_new_tokens_and_do_sample():
    """Constructor int/bool coercion + forwarding to ``model.generate``."""
    model = _StubModel()
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"], max_new_tokens=5, do_sample=True
    )
    task.run(model)
    assert model.generate_calls == [(5, True)]


def test_invariant_device_argument_moves_input_ids():
    """Passing ``device`` exercises the ``input_ids.to(device)`` branch."""
    out = _task(["p0"]).run(_StubModel(), device=torch.device("cpu"))
    assert out["results"][0].response == "7,8"  # runs cleanly through the move


def test_invariant_name_is_coerced_to_str():
    """A non-str ``name`` is coerced (used as ``task_name`` and node name)."""
    task = GenerationEvalTask(_StubJudge(), _StubTokenizer(), ["p0"], name=123)  # type: ignore[arg-type]
    assert task.name == "123"
    assert task.run(_StubModel())["task_name"] == "123"


# ---------------------------------------------------------------------------
# _write_lineage_edge(): real LineageStore
# ---------------------------------------------------------------------------

def test_invariant_lineage_skipped_when_store_is_none():
    """No lineage_store → no edge attempted; run returns normally."""
    out = _task(["p0"], artifact_id="anything").run(_StubModel())
    assert out["mean_score"] == 1.0  # completed without lineage


def test_invariant_lineage_skipped_when_artifact_id_is_none(lineage_store_factory):
    """lineage_store set but artifact_id None → skip (guard is AND of both)."""
    store = lineage_store_factory()
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"], lineage_store=store, artifact_id=None
    )
    task.run(_StubModel())
    assert list(store.iter_edges(kind="evaluated_by")) == []


def test_invariant_lineage_edge_written_for_int_artifact_id(lineage_store_factory):
    """An int artifact_id is used directly as the edge source; step → version
    ``step_<n>``."""
    store = lineage_store_factory()
    src = store.upsert_node(kind="artifact", name="m", version="v1")
    task = GenerationEvalTask(
        _StubJudge(scores=[1.0]),
        _StubTokenizer(),
        ["p0"],
        lineage_store=store,
        artifact_id=src,  # int
    )
    task.run(_StubModel(), step=5)
    edges = store.edges_from(src, kind="evaluated_by")
    assert len(edges) == 1
    assert store.find("run", "eval:generation_eval", "step_5") is not None


def test_invariant_lineage_plain_name_ref_rewritten_to_latest(lineage_store_factory):
    """A plain name artifact_id is resolved as ``artifact:<name>:latest``."""
    store = lineage_store_factory()
    src = store.upsert_node(kind="artifact", name="mymodel", version="v1")
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"],
        lineage_store=store, artifact_id="mymodel",  # no ":" → rewritten
    )
    task.run(_StubModel())
    assert len(store.edges_from(src, kind="evaluated_by")) == 1


def test_invariant_lineage_full_ref_used_verbatim(lineage_store_factory):
    """A ``kind:name:version`` ref (contains ':') is resolved as-is."""
    store = lineage_store_factory()
    src = store.upsert_node(kind="artifact", name="m2", version="v1")
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"],
        lineage_store=store, artifact_id="artifact:m2:v1",
    )
    task.run(_StubModel())
    assert len(store.edges_from(src, kind="evaluated_by")) == 1


def test_invariant_lineage_unresolved_ref_skips_silently(lineage_store_factory):
    """An artifact ref with no matching node → resolve_ref None → no edge,
    no error."""
    store = lineage_store_factory()
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"],
        lineage_store=store, artifact_id="artifact:ghost:latest",
    )
    task.run(_StubModel())
    assert list(store.iter_edges(kind="evaluated_by")) == []


def test_invariant_lineage_step_none_uses_latest_version(lineage_store_factory):
    """``step=None`` → the eval-result node version is ``latest``."""
    store = lineage_store_factory()
    src = store.upsert_node(kind="artifact", name="m3", version="v1")
    task = GenerationEvalTask(
        _StubJudge(), _StubTokenizer(), ["p0"],
        lineage_store=store, artifact_id=src,
    )
    task.run(_StubModel(), step=None)
    assert len(store.edges_from(src, kind="evaluated_by")) == 1
    assert store.find("run", "eval:generation_eval", "latest") is not None


def test_invariant_lineage_write_failure_is_swallowed():
    """A store that raises during the lineage write is caught — ``run`` still
    returns the eval results."""

    class _BoomStore:
        def upsert_node(self, **kw):
            raise RuntimeError("boom")

    task = GenerationEvalTask(
        _StubJudge(scores=[0.5]), _StubTokenizer(), ["p0"],
        lineage_store=_BoomStore(), artifact_id=1,  # type: ignore[arg-type]  # int → goes straight to upsert_node
    )
    out = task.run(_StubModel())
    assert out["mean_score"] == 0.5  # results returned despite lineage failure


def test_invariant_write_lineage_edge_defensive_return_when_store_none():
    """Directly calling ``_write_lineage_edge`` with no store returns early
    (defensive guard not reachable through ``run``)."""
    task = GenerationEvalTask(_StubJudge(), _StubTokenizer(), ["p0"], lineage_store=None)
    assert task._write_lineage_edge(0.5, 1) is None  # type: ignore[func-returns-value]
