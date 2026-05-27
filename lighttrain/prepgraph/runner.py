"""PrepRunner — orchestrates fingerprint planning + execution.

Walks ``PrepGraph.layers`` in order. For each node:

  1. Compute fingerprint from upstream fingerprints + own config + code_version.
  2. Look up the final dir; if MANIFEST_COMPLETE exists, skip (cache hit).
  3. Otherwise run the node into a staging dir and atomically commit.

Within a layer, nodes can run concurrently (thread pool by default; process
pool is available for CPU-bound nodes). The runner's public surface is small
and synchronous: ``plan()``, ``dry_run()``, ``run()``.

Each node's manifest carries ``lineage_pending: True`` and
``derived_from: [upstream_fps]`` so the :class:`LineageStore` can backfill
SQLite after the fact.
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from . import _io as _io
from ._banner import PlanEntry, print_plan
from ._fp import SCHEMA_VERSION, short
from .dag import PrepGraph
from .node import NodeResult, PrepNode, RunContext, materialize_manifest


@dataclass
class _ResolvedFingerprint:
    fp: str
    upstream_fps: list[str]
    final_dir: Path
    hit: bool
    reason: str
    schema_kind: str
    schema_version_known: str
    schema_version_recorded: str | None
    extras: dict[str, Any] = field(default_factory=dict)


class PrepRunner:
    """Orchestrate fingerprint planning + execution for a PrepGraph."""

    def __init__(
        self,
        graph: PrepGraph,
        *,
        store_root: str | Path,
        workers: int | None = None,
        console: Any | None = None,
        pool_kind: Literal["thread", "process"] = "thread",
    ) -> None:
        self.graph = graph
        self.store_root = Path(store_root)
        self.workers = max(1, int(workers)) if workers else 1
        self.console = console
        if pool_kind not in ("thread", "process"):
            raise ValueError(
                f"pool_kind must be 'thread' or 'process', got {pool_kind!r}"
            )
        self.pool_kind = pool_kind
        self._fp_cache: dict[str, _ResolvedFingerprint] = {}

    # ----- planning --------------------------------------------------------

    def plan(self) -> list[PlanEntry]:
        """Compute fingerprints + cache hits for every node, in topological order."""
        self._fp_cache.clear()
        plan: list[PlanEntry] = []
        for name in self.graph.topo_order():
            entry = self._resolve(name)
            self._fp_cache[name] = entry
            plan.append(
                PlanEntry(
                    name=name,
                    kind=self.graph.nodes[name].kind,
                    fingerprint=short(entry.fp),
                    full_fp=entry.fp,
                    hit=entry.hit,
                    reason=entry.reason,
                )
            )
        return plan

    def dry_run(self) -> list[PlanEntry]:
        return self.plan()

    def print_banner(self, plan: list[PlanEntry] | None = None) -> None:
        plan = plan or self.plan()
        print_plan(self.console, plan)

    # ----- execution -------------------------------------------------------

    def run(self) -> dict[str, NodeResult]:
        """Run the graph. Returns a name → NodeResult map for every node.

        Partial-cache reuse (post-review fix): when a downstream node misses
        but its upstream is a fingerprint hit, the upstream still has to
        materialize *something* the downstream can consume.

        * Non-materialize nodes don't write their rows to disk → they get
          demoted from "cache_hit" to "rerun_for_downstream" and execute
          again in memory. Their fingerprint stays valid, so the committed
          ``final_dir`` is just overwritten with the same MANIFEST_COMPLETE.
        * Materialize nodes did persist their data → they stay cache-hit
          but get **rehydrated** (rows + store loaded from ``final_dir``).
        """
        plan = self.plan()
        must_execute, must_rehydrate = self._compute_runtime_sets(plan)

        # Patch plan in place so the banner reflects post-review semantics.
        for entry in plan:
            if entry.hit and entry.name in must_execute:
                entry.hit = False
                entry.reason = "rerun_for_downstream"

        results: dict[str, NodeResult] = {}

        # Pre-load cached results (including rehydrated materialize stores)
        # so downstream nodes can mount them via ``ctx.upstream``.
        for entry in plan:
            if entry.name in must_execute:
                continue  # will be filled by _run_one below
            node = self.graph.nodes[entry.name]
            rfp = self._fp_cache[entry.name]
            if entry.name in must_rehydrate:
                results[entry.name] = self._rehydrate_cached(node, rfp)
            else:
                results[entry.name] = NodeResult(
                    fingerprint=rfp.fp,
                    final_dir=rfp.final_dir,
                    schema_kind=node.schema_kind,
                )

        for layer in self.graph.layers:
            todo = [n for n in layer if n in must_execute]
            if not todo:
                continue
            if self.workers <= 1 or len(todo) == 1:
                for name in todo:
                    results[name] = self._run_one(name, results)
            elif self.pool_kind == "process":
                # ProcessPool avoids GIL serialization for CPU-bound nodes.
                # Uses the module-level ``_run_node_in_subprocess`` so we don't
                # drag ``self.console`` (rich.Console — not picklable) into workers.
                with ProcessPoolExecutor(max_workers=self.workers) as ex:
                    futs = {}
                    for name in todo:
                        rfp = self._fp_cache[name]
                        upstream = {
                            u: results[u] for u in self.graph.nodes[name].inputs
                        }
                        futs[
                            ex.submit(
                                _run_node_in_subprocess,
                                self.graph.nodes[name],
                                rfp.fp,
                                rfp.upstream_fps,
                                rfp.final_dir,
                                upstream,
                                self.store_root,
                                self.workers,
                            )
                        ] = name
                    for fut in futs:
                        results[futs[fut]] = fut.result()
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as ex:
                    futs = {
                        ex.submit(self._run_one, name, results): name for name in todo
                    }
                    for fut in futs:
                        results[futs[fut]] = fut.result()
        return results

    # ----- partial-cache rehydration ---------------------------------------

    def _compute_runtime_sets(
        self, plan: list[PlanEntry]
    ) -> tuple[set[str], set[str]]:
        """Given the fingerprint-based plan, decide which nodes actually need
        to execute (in addition to misses) and which materialize-cache hits
        need their rows/store reloaded from disk.

        A cached non-materialize node must execute again if **any** descendant
        is going to execute: it doesn't persist rows, so the descendant would
        otherwise read ``upstream.rows = None`` (the original review-after-fix
        bug).
        """
        by_hit = {e.name: e.hit for e in plan}
        must_execute: set[str] = {e.name for e in plan if not e.hit}

        # Walk dependencies backwards from any executing node; demote
        # non-materialize hits along the way.
        work = list(must_execute)
        while work:
            cur = work.pop()
            node = self.graph.nodes[cur]
            for u in node.inputs:
                if u in must_execute:
                    continue
                up_node = self.graph.nodes[u]
                if by_hit.get(u, False) and up_node.kind != "materialize":
                    must_execute.add(u)
                    work.append(u)

        # Materialize cache hits whose direct downstream will execute → rehydrate.
        must_rehydrate: set[str] = set()
        for entry in plan:
            if entry.name in must_execute:
                continue
            node = self.graph.nodes[entry.name]
            if not by_hit.get(entry.name) or node.kind != "materialize":
                continue
            for other in plan:
                if other.name in must_execute and entry.name in self.graph.nodes[other.name].inputs:
                    must_rehydrate.add(entry.name)
                    break
        return must_execute, must_rehydrate

    def _rehydrate_cached(
        self, node: PrepNode, rfp: _ResolvedFingerprint
    ) -> NodeResult:
        """Reload rows + store for a cached materialize node so a non-hit
        downstream can consume them.

        Only ``materialize`` kind is rehydrated here; non-materialize kinds
        re-execute (see ``_compute_runtime_sets``). For materialize:

        * ``shards.json`` present → ``_RowsDataset`` + iterate rows
        * ``header.json`` present (memmap layout) → ``MemmapDataset``, no rows
        """
        from ..data.cache._shards import iter_rows, read_manifest

        rows: list[dict[str, Any]] | None = None
        store: Any = None

        if node.kind == "materialize":
            if (rfp.final_dir / "shards.json").exists() or read_manifest(rfp.final_dir):
                # Lazy import to avoid runner ↔ nodes circular import at top-level.
                from .nodes.materialize import _RowsDataset

                store = _RowsDataset(rfp.final_dir)
                rows = list(iter_rows(rfp.final_dir))
            elif (rfp.final_dir / "header.json").exists():
                from ..data.cache._memmap import MemmapDataset

                store = MemmapDataset(rfp.final_dir)
                # memmap layout: downstream consumes via store, not rows.

        return NodeResult(
            fingerprint=rfp.fp,
            final_dir=rfp.final_dir,
            schema_kind=node.schema_kind,
            rows=rows,
            store=store,
        )

    # ----- internals -------------------------------------------------------

    def _resolve(self, name: str) -> _ResolvedFingerprint:
        node = self.graph.nodes[name]
        upstream_fps = [self._fp_cache[u].fp for u in node.inputs]
        fp = node.fingerprint(upstream_fps)
        final = _io.final_dir(self.store_root, node.kind, name, fp)
        manifest = _io.read_manifest(final) if _io.is_complete(final) else None

        schema_known = SCHEMA_VERSION.get(node.schema_kind, "0.0")
        schema_recorded = manifest.get("schema_version") if manifest else None

        if manifest is None:
            hit = False
            reason = self._explain_miss(node, name, final)
        else:
            if schema_recorded != schema_known:
                hit = False
                reason = "schema_version_bumped"
            else:
                hit = True
                reason = "cache_hit"
        return _ResolvedFingerprint(
            fp=fp,
            upstream_fps=upstream_fps,
            final_dir=final,
            hit=hit,
            reason=reason,
            schema_kind=node.schema_kind,
            schema_version_known=schema_known,
            schema_version_recorded=schema_recorded,
        )

    def _explain_miss(self, node: PrepNode, name: str, final: Path) -> str:
        """Best-effort reason: scan sibling fingerprints for the same node name."""
        kind_dir = self.store_root / node.kind / name
        if not kind_dir.exists():
            return "first_run"
        # Look at sibling manifests to see what changed.
        for sibling in kind_dir.iterdir():
            sib_manifest = _io.read_manifest(sibling)
            if not sib_manifest:
                continue
            old_cv = sib_manifest.get("code_version")
            if old_cv and old_cv != node.code_version():
                return "code_version_changed"
            old_cfg = sib_manifest.get("config") or {}
            if old_cfg != dict(node.config):
                return "config_changed"
        return "upstream_changed"

    def _run_one(
        self,
        name: str,
        results: dict[str, NodeResult],
    ) -> NodeResult:
        node = self.graph.nodes[name]
        rfp = self._fp_cache[name]
        upstream = {u: results[u] for u in node.inputs}

        staging = _io.staging_dir(self.store_root, rfp.fp)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        _io.ensure_dir(staging)

        ctx = RunContext(
            store_root=staging,
            workers=self.workers,
            upstream=upstream,
            log=self.console,
        )

        t0 = time.time()
        result = node.run(ctx)
        elapsed = time.time() - t0

        manifest = materialize_manifest(
            node=node,
            fingerprint=rfp.fp,
            input_fps=rfp.upstream_fps,
            extra={
                "elapsed_s": elapsed,
                **result.extras,
            },
        )
        _io.write_manifest(staging, manifest)
        _io.commit(staging, rfp.final_dir)
        result.final_dir = rfp.final_dir
        result.fingerprint = rfp.fp
        # The store, if any, was constructed against the staging dir; rebuild
        # it against final_dir so callers see the post-commit on-disk layout.
        result.store = _rebind_store(result.store, rfp.final_dir)
        return result

    # ----- maintenance ----------------------------------------------------
    def cleanup_orphans(self, *, dry_run: bool = False) -> list[Path]:
        """Remove cache directories that no live node references."""
        live: set[Path] = set()
        for name in self.graph.topo_order():
            self._resolve(name)
        for entry in self._fp_cache.values():
            live.add(entry.final_dir.resolve())
        removed: list[Path] = []
        for kind_dir in self.store_root.iterdir() if self.store_root.exists() else []:
            if not kind_dir.is_dir() or kind_dir.name == "tmp":
                continue
            for name_dir in kind_dir.iterdir():
                if not name_dir.is_dir():
                    continue
                for fp_dir in name_dir.iterdir():
                    if not fp_dir.is_dir():
                        continue
                    if fp_dir.resolve() not in live:
                        removed.append(fp_dir)
                        if not dry_run:
                            shutil.rmtree(fp_dir, ignore_errors=True)
        if not dry_run:
            _io.cleanup_staging(self.store_root)
        return removed


__all__ = ["PrepRunner"]


def _run_node_in_subprocess(
    node: PrepNode,
    fp: str,
    upstream_fps: list[str],
    final_dir: Path,
    upstream: dict[str, NodeResult],
    store_root: Path,
    workers: int,
) -> NodeResult:
    """Subprocess entry-point for ProcessPool execution.

    Mirrors ``PrepRunner._run_one`` but without any ``self`` closure so it
    pickles cleanly. Nodes must themselves be pickle-friendly — any node that
    closes over a non-picklable resource (open file, ``rich.Console``, live
    HF model) must run in the thread pool path instead.
    """
    staging = _io.staging_dir(Path(store_root), fp)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    _io.ensure_dir(staging)

    ctx = RunContext(
        store_root=staging,
        workers=workers,
        upstream=upstream,
        log=None,  # No console across process boundaries.
    )
    t0 = time.time()
    result = node.run(ctx)
    elapsed = time.time() - t0

    manifest = materialize_manifest(
        node=node,
        fingerprint=fp,
        input_fps=upstream_fps,
        extra={"elapsed_s": elapsed, **result.extras},
    )
    _io.write_manifest(staging, manifest)
    _io.commit(staging, Path(final_dir))
    result.final_dir = Path(final_dir)
    result.fingerprint = fp
    result.store = _rebind_store(result.store, Path(final_dir))
    return result


def _rebind_store(store: Any, final_dir: Path) -> Any:
    """Re-point a materialize store at the post-commit final dir."""
    if store is None:
        return None
    out_dir_attr = getattr(store, "out_dir", None) or getattr(store, "root", None)
    if out_dir_attr is None:
        return store
    cls = type(store)
    try:
        return cls(final_dir)
    except Exception:
        return store
