"""Tokenize PrepNode — apply a Processor or raw tokenizer to upstream rows.

Two flavours, selected by ``processor`` config:

* ``processor: {name: chat_template, ...}`` — full Processor (e.g.
  ``ChatTemplateProcessor`` / ``HFTextProcessor``) called once per turn list,
  yielding ``{input_ids, labels, attention_mask, modality}``.
* otherwise: a tokenizer is built (registry/category=tokenizer or _target_)
  and applied to ``row[text_field]``; labels mirror input_ids.

Output rows always include ``input_ids`` + ``labels`` + ``attention_mask``
so downstream nodes are uniform.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping

from ...config._resolver import resolve as _resolve
from ...registry import register
from ..node import NodeResult, PrepNode, RunContext


def _build_processor(spec: Mapping[str, Any]) -> Any:
    return _resolve(spec, category="processor")


def _build_tokenizer(spec: Mapping[str, Any]) -> Any:
    return _resolve(spec, category="tokenizer")


def _label_with_mask(
    ids: list[int], labels: list[int], label_ignore: int
) -> list[int]:
    if len(labels) != len(ids):
        labels = list(ids)
    return labels


@register("prep_node", "tokenize")
class TokenizeNode(PrepNode):
    """Tokenize each upstream row.

    Config keys:

    * ``processor``: optional ComponentSpec for a Processor (preferred for
      chat / multimodal data). Mutually exclusive with ``tokenizer``.
    * ``tokenizer``: ComponentSpec for a plain tokenizer. Required if
      ``processor`` is omitted.
    * ``text_field``: row key to read text from when using a plain tokenizer
      (default ``"text"``).
    * ``turns_field``: row key for chat turns when using a Processor
      (default ``"messages"``).
    * ``label_ignore``: int (default ``-100``).
    * ``response_only_mask``: bool — forwarded to the Processor where
      applicable.
    """

    kind = "tokenize"
    schema_kind = "tokenized_rows"

    def __init__(
        self,
        *,
        name: str,
        inputs: list[str] | None = None,
        config: Mapping[str, Any] | None = None,
        device_hint: str = "any",
    ) -> None:
        super().__init__(name=name, inputs=inputs, config=config, device_hint=device_hint)
        self._proc: Any | None = None
        self._tok: Any | None = None

    def _ensure(self) -> tuple[Any | None, Any | None]:
        if self._proc is None and self._tok is None:
            proc_spec = self.config.get("processor")
            tok_spec = self.config.get("tokenizer")
            if proc_spec and tok_spec:
                raise ValueError(
                    f"TokenizeNode {self.name!r}: pass exactly one of `processor` "
                    "or `tokenizer`, not both."
                )
            if proc_spec:
                self._proc = _build_processor(proc_spec)
            elif tok_spec:
                self._tok = _build_tokenizer(tok_spec)
            else:
                raise ValueError(
                    f"TokenizeNode {self.name!r}: must declare `processor` or `tokenizer`."
                )
        return self._proc, self._tok

    def _iter_tokenized(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
        proc, tok = self._ensure()
        text_field = str(self.config.get("text_field", "text"))
        turns_field = str(self.config.get("turns_field", "messages"))
        label_ignore = int(self.config.get("label_ignore", -100))

        for row in rows:
            sample: dict[str, Any] = dict(row)
            if proc is not None:
                turns = row.get(turns_field) or row.get("turns") or row.get("conversations")
                if turns is None:
                    turns = [{"role": "user", "content": str(row.get(text_field, ""))}]
                kwargs: dict[str, Any] = {}
                if "response_only_mask" in self.config:
                    kwargs["response_only_mask"] = self.config["response_only_mask"]
                out = proc(turns, **kwargs)
                ids = list(out.get("input_ids", []))
                labels = list(out.get("labels", ids))
                attn = list(out.get("attention_mask", [1] * len(ids)))
                modality = out.get("modality", "text")
                sample.pop(turns_field, None)
                sample.update(
                    {
                        "input_ids": ids,
                        "labels": _label_with_mask(ids, labels, label_ignore),
                        "attention_mask": attn,
                        "modality": modality,
                    }
                )
                # Carry processor-emitted modality_inputs (image/audio/etc.)
                if "modality_inputs" in out:
                    sample["modality_inputs"] = out["modality_inputs"]
            else:
                text = str(row.get(text_field, ""))
                ids = list(tok.encode(text))  # type: ignore[union-attr]
                sample.update(
                    {
                        "input_ids": ids,
                        "labels": list(ids),
                        "attention_mask": [1] * len(ids),
                        "modality": "text",
                    }
                )
            yield sample

    def run(self, ctx: RunContext) -> NodeResult:
        if not self.inputs:
            raise ValueError(f"TokenizeNode {self.name!r}: requires upstream input.")
        upstream = ctx.upstream[self.inputs[0]]
        rows = upstream.rows
        if rows is None:
            raise RuntimeError(
                f"TokenizeNode {self.name!r}: upstream {self.inputs[0]!r} produced no rows in memory."
            )
        out_rows = list(self._iter_tokenized(rows))
        return NodeResult(
            fingerprint="",
            schema_kind=self.schema_kind,
            rows=out_rows,
            extras={"row_count": len(out_rows)},
        )


__all__ = ["TokenizeNode"]
