"""Lineage subcommand functions: tag / untag / invalidate / pin / gc / prune-orphans / graph.

Functions only — the ``lineage`` sub-typer is owned by the ``_app`` assembler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from lighttrain.cli._context import console


def _open_lineage(db: Path) -> Any:
    """Open a LineageStore at ``db`` — caller closes."""
    from lighttrain.lineage.store import LineageStore

    if not db.exists():
        console.print(f"[red]lineage db not found at {db}[/]")
        raise typer.Exit(code=1)
    return LineageStore(db)


def _resolve_node(store: Any, ref: str) -> int:
    nid = store.resolve_ref(ref)
    if nid is None:
        console.print(f"[red]no lineage node matches ref {ref!r}[/]")
        raise typer.Exit(code=1)
    return int(nid)


def lineage_tag_cmd(
    node: str = typer.Argument(...),
    tag: str = typer.Option(..., "--tag"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.tag(nid, tag)
        console.print(f"[green]tagged[/] #{nid} += {tag!r}")
    finally:
        store.close()


def lineage_untag_cmd(
    node: str = typer.Argument(...),
    tag: str = typer.Option(..., "--tag"),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.untag(nid, tag)
        console.print(f"[green]untagged[/] #{nid} -= {tag!r}")
    finally:
        store.close()


def lineage_invalidate_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.invalidate(nid)
        console.print(f"[yellow]invalidated[/] #{nid}")
    finally:
        store.close()


def lineage_pin_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
) -> None:
    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        store.pin(nid)
        console.print(f"[green]pinned[/] #{nid}")
    finally:
        store.close()


def lineage_gc_cmd(
    db: Path = typer.Option(..., "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    keep_last: int = typer.Option(3, "--keep-last"),
    kind: str = typer.Option("artifact", "--kind", help="artifact|checkpoint|config|run"),
) -> None:
    from lighttrain.lineage.retention import RetentionPolicy, gc_artifacts

    store = _open_lineage(db)
    try:
        report = gc_artifacts(
            store,
            policy=RetentionPolicy(keep_last=keep_last, keep_tagged=True, keep_pinned=True),
            kind=kind,
            dry_run=dry_run,
            delete_paths=not dry_run,
        )
        console.print(
            f"[green]gc[/] deprecated={len(report.deprecated)} deleted={len(report.deleted)} "
            f"paths_deleted={len(report.paths_deleted)}"
        )
    finally:
        store.close()


def lineage_prune_cmd(
    db: Path = typer.Option(..., "--db"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    from lighttrain.lineage.retention import prune_orphans

    store = _open_lineage(db)
    try:
        removed = prune_orphans(store, dry_run=dry_run)
        console.print(f"[green]pruned[/] {len(removed)} orphan node(s)")
    finally:
        store.close()


def lineage_graph_cmd(
    node: str = typer.Argument(...),
    db: Path = typer.Option(..., "--db"),
    depth: int = typer.Option(5, "--depth"),
    out: Path | None = typer.Option(None, "--out", help="Write to file; ext=.dot or .mermaid."),
    fmt: str = typer.Option("mermaid", "--fmt", help="mermaid | dot"),
) -> None:
    from lighttrain.lineage.dag import to_dot, to_mermaid

    store = _open_lineage(db)
    try:
        nid = _resolve_node(store, node)
        if fmt == "dot":
            text = to_dot(store, nid, depth=depth)
        else:
            text = to_mermaid(store, nid, depth=depth)
        if out is not None:
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]wrote[/] {out}")
        else:
            console.print(text)
    finally:
        store.close()
