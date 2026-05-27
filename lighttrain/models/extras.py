"""ExtraOutputSpec — declarative forward-hook extraction.

A user describes "which submodule output do I want, under which name, with what
transform" in YAML; :class:`ExtrasHookManager` installs `register_forward_hook`
handlers, collects tensors as the model runs, and merges them into
``ModelOutput.extras``. Losses and artifact producers consume from
``extras`` by key without caring how it got there.

Source matching:
  * exact dotted name (``"model.lm_head"``)
  * glob-like ``{a,b,c}`` group (``"block.{8,16,24}.output"``)
  * Python regex (set ``source_kind="regex"``)
  * ``.input`` / ``.output`` suffix selects forward-input vs forward-output

Transforms (each spec applies at most one; chain by composing two specs):
  * ``{topk: K}``            — keep top-K logits along last dim, emit values + indices
  * ``{slice: [i, j]}``       — slice along last dim
  * ``{layer: i}``            — pick layer index from a tuple (used with hidden_states)
  * ``{mean_dim: d}``         — mean along dim

The protocol stays small on purpose; new transforms register here, not in core.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from ..protocols import ModelOutput

__all__ = [
    "ExtraOutputSpec",
    "ExtrasHookManager",
    "extract_extra_outputs",
    "flatten_model_output_tensors",
    "compile_pattern",
]


_BRACE_GROUP_RE = re.compile(r"\{([^{}]*)\}")


def _expand_braces(pattern: str) -> str:
    """Turn ``block.{8,16,24}.output`` into ``block\\.(8|16|24)\\.output`` regex."""

    def _sub(match: re.Match[str]) -> str:
        parts = [re.escape(p.strip()) for p in match.group(1).split(",") if p.strip()]
        if not parts:
            return ""
        return "(" + "|".join(parts) + ")"

    # Step 1: handle brace groups before fnmatch escaping eats them.
    body = _BRACE_GROUP_RE.sub(_sub, pattern)
    # Step 2: convert remaining glob metacharacters (* ? [seq]) to regex, but
    # preserve already-substituted brace groups and our literal dots.
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "*":
            out.append(".*")
        elif c == "?":
            out.append(".")
        elif c == ".":
            out.append(r"\.")
        else:
            out.append(c)
        i += 1
    return "".join(out)


def compile_pattern(source: str, kind: str = "auto") -> re.Pattern[str]:
    """Compile an ExtraOutputSpec ``source`` to a regex.

    ``kind="regex"`` uses the string verbatim; ``"glob"`` expands ``*``/``?``/
    ``{a,b}`` braces; ``"auto"`` picks ``glob`` if any of those metacharacters
    appear, else treats it as a literal module name.
    """
    if kind == "regex":
        return re.compile(source)
    if kind == "auto":
        kind = "glob" if any(ch in source for ch in "*?{") else "literal"
    if kind == "literal":
        return re.compile("^" + re.escape(source) + "$")
    # glob
    return re.compile("^" + _expand_braces(source) + "$")


@dataclass
class ExtraOutputSpec:
    """Declarative extraction spec.

    Parameters
    ----------
    name : str
        Output key — what shows up in ``ModelOutput.extras[name]`` and on disk.
    source : str
        Module-path pattern; ``.input`` / ``.output`` suffix selects the side.
        Default is ``output``. Wildcards: ``*``, ``?``, ``{a,b,c}``; pass
        ``source_kind="regex"`` for a raw Python regex.
    source_kind : str
        ``auto`` (default) / ``glob`` / ``regex`` / ``literal``.
    transform : Mapping
        Optional in-flight transform (``topk`` / ``slice`` / ``layer`` /
        ``mean_dim``). At most one per spec; chain by composing specs.
    detach : bool
        Detach captured tensor (recommended, default True).
    cpu : bool
        Move to CPU (default True — artifact stores expect CPU tensors).
    """

    name: str
    source: str
    source_kind: str = "auto"
    transform: Mapping[str, Any] | None = None
    detach: bool = True
    cpu: bool = True

    def __post_init__(self) -> None:
        self._side = "output"
        src = self.source
        if src.endswith(".input"):
            self._side = "input"
            src = src[: -len(".input")]
        elif src.endswith(".output"):
            self._side = "output"
            src = src[: -len(".output")]
        self._stripped_source = src
        self._regex = compile_pattern(src, self.source_kind)

    def side(self) -> str:
        return self._side

    def matches(self, module_name: str) -> bool:
        return bool(self._regex.match(module_name))

    def apply_transform(self, tensor: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        if not self.transform:
            return tensor
        t = dict(self.transform)
        if "topk" in t:
            k = int(t["topk"])
            vals, idx = torch.topk(tensor, k=k, dim=-1)
            return {"values": vals, "indices": idx.to(torch.int32)}
        if "slice" in t:
            i, j = t["slice"]
            return tensor[..., int(i) : int(j)]
        if "layer" in t:
            # When the captured tensor is itself a stack along dim 0 (e.g. hidden_states list).
            return tensor[int(t["layer"])]
        if "mean_dim" in t:
            return tensor.mean(dim=int(t["mean_dim"]))
        return tensor


class ExtrasHookManager:
    """Install + remove forward hooks for a set of ExtraOutputSpec.

    Usage::

        mgr = ExtrasHookManager(model, specs)
        mgr.attach()
        try:
            output = model(**batch)
            extras = mgr.collect()           # dict[name -> Tensor | (vals, idx)]
        finally:
            mgr.detach()
    """

    def __init__(self, model: torch.nn.Module, specs: Iterable[ExtraOutputSpec]) -> None:
        self.model = model
        self.specs = list(specs)
        self._handles: list[Any] = []
        self._cache: dict[str, Any] = {}
        self._matched: dict[str, list[ExtraOutputSpec]] = {}

    def attach(self) -> "ExtrasHookManager":
        if self._handles:
            return self  # already attached — idempotent
        for module_name, module in self.model.named_modules():
            specs_here = [s for s in self.specs if s.matches(module_name)]
            if not specs_here:
                continue
            self._matched.setdefault(module_name, []).extend(specs_here)
            for spec in specs_here:
                handle = module.register_forward_hook(
                    self._make_hook(spec=spec, module_name=module_name)
                )
                self._handles.append(handle)
        return self

    def _make_hook(self, *, spec: ExtraOutputSpec, module_name: str):
        def _hook(mod, inputs, output):
            t = inputs[0] if spec.side() == "input" else output
            if isinstance(t, tuple):
                t = t[0]
            if not isinstance(t, torch.Tensor):
                return
            if spec.detach:
                t = t.detach()
            if spec.cpu:
                t = t.cpu()
            result = spec.apply_transform(t)
            self._cache[spec.name] = result
        return _hook

    def collect(self) -> dict[str, Any]:
        """Return a snapshot dict of {spec.name: tensor | {values, indices}}."""
        return dict(self._cache)

    def reset(self) -> None:
        self._cache.clear()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._matched.clear()
        self._cache.clear()

    @property
    def matched_modules(self) -> dict[str, list[ExtraOutputSpec]]:
        return dict(self._matched)


def extract_extra_outputs(
    model_output: ModelOutput,
    specs: Iterable[ExtraOutputSpec] | None = None,
) -> dict[str, torch.Tensor]:
    """Pull tensors from ``model_output.extras``; specs without a hook source
    (i.e. their name is already populated by the hook manager) are flattened
    into a string-keyed dict. ``topk`` transforms expand to ``<name>.values`` +
    ``<name>.indices`` so the artifact store can put each tensor separately.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in model_output.extras.items():
        if isinstance(value, Mapping):
            for sub, t in value.items():
                if isinstance(t, torch.Tensor):
                    out[f"{key}.{sub}"] = t
        elif isinstance(value, torch.Tensor):
            out[str(key)] = value
    if specs is None:
        return out
    return out


def flatten_model_output_tensors(model_output: ModelOutput) -> dict[str, torch.Tensor]:
    """Flatten outputs + extras + hidden_states + attentions into one dict.

    Used by :class:`ModelForwardProducer` to dump everything the user asked for
    into a single artifact store ``put`` call.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in (model_output.outputs or {}).items():
        if isinstance(v, torch.Tensor):
            out[str(k)] = v
    for k, v in (model_output.extras or {}).items():
        if isinstance(v, torch.Tensor):
            out[str(k)] = v
        elif isinstance(v, Mapping):
            for sub, t in v.items():
                if isinstance(t, torch.Tensor):
                    out[f"{k}.{sub}"] = t
    if model_output.hidden_states is not None:
        hs = [t for t in model_output.hidden_states if isinstance(t, torch.Tensor)]
        if hs:
            stacked = torch.stack([t.detach().cpu() for t in hs], dim=0)
            out["hidden_states_layers"] = stacked
    if model_output.attentions is not None:
        ats = [t for t in model_output.attentions if isinstance(t, torch.Tensor)]
        if ats:
            stacked = torch.stack([t.detach().cpu() for t in ats], dim=0)
            out["attentions_layers"] = stacked
    return out
