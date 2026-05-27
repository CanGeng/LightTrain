"""PrepNode base class + result/estimate dataclasses.

Every concrete node lives in ``lighttrain/prepgraph/nodes/`` and registers via
``@register("prep_node", "<kind>")``. The base class implements the
``PrepNodeProtocol`` shape:

  * ``name``      — unique identifier within a graph
  * ``kind``      — registry kind (``load`` / ``tokenize`` / ``pack`` / ...)
  * ``inputs``    — list of upstream node names (for DAG construction)
  * ``config``    — node-specific config (used in fingerprint)
  * ``schema_kind`` — name in ``SCHEMA_VERSION`` (which row schema this emits)
  * ``code_version()`` — sha256 of the implementation source
  * ``fingerprint(input_fps)`` — composed via ``compose_fingerprint``
  * ``run(ctx)`` — does the work and returns a ``NodeResult``
  * ``estimate(ctx)`` — pre-flight cost estimate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ._fp import code_version_for, compose_fingerprint


@dataclass
class NodeEstimate:
    """Pre-flight estimate. All fields optional."""

    rows: int | None = None
    bytes: int | None = None
    eta_s: float | None = None
    note: str | None = None


@dataclass
class RunContext:
    """Resources passed into ``PrepNode.run``."""

    store_root: Path
    workers: int = 1
    upstream: dict[str, "NodeResult"] = field(default_factory=dict)
    dry_run: bool = False
    # Caller-supplied logger (rich console wrapper or print). May be None.
    log: Any | None = None


@dataclass
class NodeResult:
    """A node's output handle.

    ``rows`` is an in-memory iterable for streaming nodes (load/tokenize); on
    ``materialize`` it points at the on-disk dataset directory via ``store``.
    Both fields can coexist when a node both writes shards AND yields rows for
    downstream consumption.
    """

    fingerprint: str
    final_dir: Path | None = None
    schema_kind: str = "rows"
    rows: Any | None = None
    store: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class PrepNode:
    """Base class for PrepGraph nodes.

    Subclasses populate at least ``kind``, ``schema_kind``, and ``run``.
    """

    kind: str = ""
    schema_kind: str = "rows"

    def __init__(
        self,
        *,
        name: str,
        inputs: list[str] | None = None,
        config: Mapping[str, Any] | None = None,
        device_hint: str = "any",
    ) -> None:
        if not name:
            raise ValueError("PrepNode requires a non-empty name.")
        if not self.kind:
            raise ValueError(
                f"PrepNode subclass {type(self).__name__} must set class attribute `kind`."
            )
        self.name = str(name)
        self.inputs = list(inputs or [])
        self.config: dict[str, Any] = dict(config or {})
        self.device_hint = device_hint
        self._fp_cache: str | None = None

    # ----- identity --------------------------------------------------------

    def code_version(self) -> str:
        return code_version_for(type(self))

    def fingerprint(self, input_fps: Iterable[str] = ()) -> str:
        return compose_fingerprint(
            kind=self.kind,
            schema_kind=self.schema_kind,
            code_version=self.code_version(),
            config=self.config,
            input_fps=input_fps,
        )

    # ----- behaviour hooks -------------------------------------------------

    def run(self, ctx: RunContext) -> NodeResult:  # pragma: no cover — abstract
        raise NotImplementedError

    def estimate(self, ctx: RunContext) -> NodeEstimate:
        return NodeEstimate(note="no-estimator")

    # ----- introspection ---------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"<PrepNode kind={self.kind} name={self.name} inputs={self.inputs}>"


def materialize_manifest(
    *,
    node: PrepNode,
    fingerprint: str,
    input_fps: Iterable[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical manifest payload."""
    payload: dict[str, Any] = {
        "kind": node.kind,
        "name": node.name,
        "schema_kind": node.schema_kind,
        "schema_version": "0.1",
        "fingerprint": fingerprint,
        "code_version": node.code_version(),
        "config": dict(node.config),
        "lineage_pending": True,
        "derived_from": list(input_fps),
    }
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "NodeEstimate",
    "NodeResult",
    "PrepNode",
    "RunContext",
    "materialize_manifest",
]
