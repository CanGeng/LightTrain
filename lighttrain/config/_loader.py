"""Config loader — OmegaConf YAML + ``defaults:`` composition + CLI overrides.

Pipeline:

    raw YAML  →  defaults composition  →  OmegaConf merge with overrides
              →  resolve interpolations  →  Pydantic validation  →  RootConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ._exceptions import ConfigError, ConfigSchemaError
from ._schema import RootConfig


def _read_yaml_node(path: Path) -> DictConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(cfg).__name__}: {path}")
    return cfg


def _compose_defaults(path: Path, _seen: set[Path] | None = None) -> DictConfig:
    """Resolve a ``defaults:`` list relative to ``path``'s parent and merge.

    A simple Hydra-style composer: a ``defaults: [base, ../shared/foo]`` list
    is resolved into sibling YAML files (with optional ``.yaml`` extension)
    and merged in order; the current file overrides them.
    """
    path = path.resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ConfigError(f"Circular defaults: {path}")
    seen.add(path)

    cfg = _read_yaml_node(path)
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg

    if not isinstance(defaults, (list, Iterable)) or isinstance(defaults, str):
        raise ConfigError(f"`defaults:` in {path} must be a list of relative refs.")

    base = OmegaConf.create({})
    for ref in defaults:
        if not isinstance(ref, str):
            raise ConfigError(f"`defaults:` entries must be strings, got {ref!r}")
        sub_path = (path.parent / ref).with_suffix(".yaml")
        if not sub_path.exists():
            sub_path = path.parent / ref
        if not sub_path.exists():
            raise ConfigError(f"defaults entry {ref!r} not found relative to {path}")
        sub = _compose_defaults(sub_path, seen)
        base = OmegaConf.merge(base, sub)

    seen.discard(path)
    return OmegaConf.merge(base, cfg)


def _apply_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply CLI-style overrides: ``a.b=c`` / ``++a.b=c`` (force-set) / ``~a.b`` (delete)."""
    for ov in overrides:
        if not isinstance(ov, str) or not ov:
            raise ConfigError(f"Invalid override entry: {ov!r}")
        if ov.startswith("~"):
            key = ov[1:].strip()
            if not key:
                raise ConfigError(f"Empty key in override {ov!r}")
            OmegaConf.update(cfg, key, None, merge=False)
            # OmegaConf has no public delete; mask via update + pop
            keys = key.split(".")
            node: Any = cfg
            for k in keys[:-1]:
                node = node[k]
            if keys[-1] in node:
                del node[keys[-1]]
            continue

        force = ov.startswith("++")
        body = ov[2:] if force else ov
        if "=" not in body:
            raise ConfigError(f"Override missing '=': {ov!r}")
        key, _, val = body.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"Empty key in override {ov!r}")
        # Parse value as YAML scalar so 1, 1.0, true, [1,2], {a:1} all work.
        parsed = OmegaConf.create(f"v: {val}").v if val != "" else ""
        OmegaConf.update(cfg, key, parsed, merge=False, force_add=force)
    return cfg


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | None = None,
    validate: bool = True,
) -> RootConfig | DictConfig:
    """Load a YAML config, apply overrides, and (by default) validate to RootConfig.

    Pass ``validate=False`` to get the raw resolved DictConfig (useful for tests
    and for ``--print-config``).
    """
    path = Path(path)
    cfg = _compose_defaults(path)
    cfg = _apply_overrides(cfg, list(overrides or []))
    OmegaConf.resolve(cfg)
    if not validate:
        return cfg

    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ConfigError(f"Resolved config is not a mapping: {type(plain).__name__}")

    try:
        return RootConfig.model_validate(plain)
    except ValidationError as e:
        raise ConfigSchemaError(str(e)) from e


def dump_resolved(cfg: RootConfig | DictConfig) -> str:
    """Serialize a (possibly validated) config back to YAML for ``--print-config``."""
    if isinstance(cfg, RootConfig):
        return OmegaConf.to_yaml(OmegaConf.create(cfg.model_dump()))
    return OmegaConf.to_yaml(cfg)


__all__ = ["dump_resolved", "load_config"]
