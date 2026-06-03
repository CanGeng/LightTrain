"""Text processor.

Two flavours:

* ``HFTextProcessor`` — wraps ``transformers.AutoTokenizer``; supports HF chat
  templates. Lazy-imports ``transformers``, so importing this module is free.

* ``ChatTemplateProcessor`` — hermetic byte-tokenizer-friendly chat templater
  that needs no model files. Uses Python f-strings as the template language
  (``{system}{user}{assistant}``), so it's enough for tests and the default
  ``recipes/sft_chat.yaml``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from lighttrain.registry import register

_DEFAULT_TEMPLATE = (
    "<|system|>\n{system}\n"
    "<|user|>\n{user}\n"
    "<|assistant|>\n{assistant}\n"
)


def _format_segment(template: str, role: str, content: str) -> str:
    # Pick the section starting at <|role|> up to the next <| token (or EOS).
    head = f"<|{role}|>"
    if head not in template:
        return f"{head}\n{content}\n"
    idx = template.index(head)
    rest = template[idx + len(head) :]
    end = rest.find("<|")
    body = rest[:end] if end >= 0 else rest
    return head + body.replace("{" + role + "}", content)


@register("processor", "chat_template")
class ChatTemplateProcessor:
    """Hermetic chat templater — no model file required.

    Each call expects a list of ``{"role": ..., "content": ...}`` turns; the
    processor returns a flat tokenized record::

        {
          "input_ids": [...],
          "labels": [...],          # -100 outside the assistant turns
          "attention_mask": [...],
          "modality": "text",
        }

    ``response_only_mask`` (default True) sets every non-assistant token's
    label to ``label_ignore`` (``-100`` by convention).
    """

    modality = "text"

    def __init__(
        self,
        *,
        tokenizer: Any,
        template: str = _DEFAULT_TEMPLATE,
        response_only_mask: bool = True,
        label_ignore: int = -100,
        max_len: int | None = None,
    ) -> None:
        # Allow callers (esp. PrepGraph node configs) to pass a tokenizer spec
        # rather than an instance — resolve it lazily through the registry.
        if isinstance(tokenizer, Mapping):
            from lighttrain.config._resolver import resolve as _resolve

            tokenizer = _resolve(dict(tokenizer), category="tokenizer")
        self.tokenizer = tokenizer
        self.template = template
        self.response_only_mask = bool(response_only_mask)
        self.label_ignore = int(label_ignore)
        self.max_len = int(max_len) if max_len else None

    def __call__(
        self,
        turns: Iterable[Mapping[str, str]],
        **_: Any,
    ) -> dict[str, Any]:
        ids: list[int] = []
        labels: list[int] = []
        for turn in turns:
            role = turn["role"]
            content = turn.get("content", "")
            text = _format_segment(self.template, role, content)
            seg = self.tokenizer.encode(text)
            ids.extend(seg)
            if self.response_only_mask and role != "assistant":
                labels.extend([self.label_ignore] * len(seg))
            else:
                labels.extend(seg)
        if self.max_len is not None:
            ids = ids[: self.max_len]
            labels = labels[: self.max_len]
        attn = [1] * len(ids)
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attn,
            "modality": "text",
        }


@register("processor", "hf_text")
class HFTextProcessor:
    """Wrap ``transformers.AutoTokenizer``; lazy-loads the tokenizer.

    Pass ``model_name_or_path`` and any kwargs forwarded to
    ``AutoTokenizer.from_pretrained``. Optional ``chat_template`` overrides
    the tokenizer's default if set.
    """

    modality = "text"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        chat_template: str | None = None,
        max_len: int | None = None,
        add_generation_prompt: bool = False,
        response_only_mask: bool = True,
        label_ignore: int = -100,
        from_pretrained_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.chat_template = chat_template
        self.max_len = int(max_len) if max_len else None
        self.add_generation_prompt = bool(add_generation_prompt)
        self.response_only_mask = bool(response_only_mask)
        self.label_ignore = int(label_ignore)
        self._fp_kwargs = dict(from_pretrained_kwargs or {})
        self._tokenizer: Any | None = None

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            tk = AutoTokenizer.from_pretrained(
                self.model_name_or_path, **self._fp_kwargs
            )
            if self.chat_template:
                tk.chat_template = self.chat_template
            self._tokenizer = tk
        return self._tokenizer

    def __call__(
        self,
        turns: Iterable[Mapping[str, str]],
        **_: Any,
    ) -> dict[str, Any]:
        tk = self._ensure_tokenizer()
        turns = list(turns)
        rendered = tk.apply_chat_template(
            turns,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )
        encoded = tk(rendered, return_offsets_mapping=False, add_special_tokens=False)
        ids = list(encoded["input_ids"])
        attn = list(encoded.get("attention_mask", [1] * len(ids)))
        if self.max_len is not None:
            ids = ids[: self.max_len]
            attn = attn[: self.max_len]
        labels = list(ids)
        if self.response_only_mask:
            # Mask every prefix turn — only the last assistant span gets to
            # keep its labels.
            prefix_turns = turns[:-1] if turns and turns[-1]["role"] == "assistant" else turns
            prefix = tk.apply_chat_template(
                prefix_turns,
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix_ids = list(tk(prefix, add_special_tokens=False)["input_ids"])
            cut = min(len(prefix_ids), len(labels))
            for i in range(cut):
                labels[i] = self.label_ignore
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attn,
            "modality": "text",
        }


__all__ = ["ChatTemplateProcessor", "HFTextProcessor"]
