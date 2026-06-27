"""MiniMind datasets + collators registered for lighttrain.

Adds @register decorators so recipes can reference these via user_modules:
  - ``minimind_pretrain``  → PretrainDataset
  - ``minimind_sft``       → SFTDataset
  - ``minimind_dpo``       → DPODataset
  - ``minimind_rlaif``     → RLAIFDataset
  - ``minimind_pad``       → MiniMindCollator (stacks (input_ids, labels) pairs)
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset

from lighttrain.registry import register

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Chat pre/post processing helpers (verbatim from original)
# ---------------------------------------------------------------------------

def _pre_processing_chat(conversations: list, add_system_ratio: float = 0.2) -> list:
    if any(conv.get("tools") for conv in conversations):
        return conversations
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations[0].get("role") != "system" and random.random() < add_system_ratio:
        return [{"role": "system", "content": random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def _post_processing_chat(prompt: str, empty_think_ratio: float = 0.2) -> str:
    if "<think>\n\n</think>\n\n" in prompt and random.random() > empty_think_ratio:
        prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    return prompt


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@register("dataset", "minimind_pretrain")
class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


@register("dataset", "minimind_sft")
class SFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 1024) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        features = Features({
            "conversations": [{"role": Value("string"), "content": Value("string"),
                               "reasoning_content": Value("string"), "tools": Value("string"),
                               "tool_calls": Value("string")}]
        })
        self.samples = load_dataset("json", data_files=jsonl_path, split="train", features=features)
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def _create_prompt(self, conversations: list) -> str:
        messages, tools = [], None
        for msg in conversations:
            msg = dict(msg)
            if msg.get("role") == "system" and msg.get("tools"):
                tools = json.loads(msg["tools"]) if isinstance(msg["tools"], str) else msg["tools"]
            if msg.get("tool_calls") and isinstance(msg["tool_calls"], str):
                msg["tool_calls"] = json.loads(msg["tool_calls"])
            messages.append(msg)
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=False, tools=tools)

    def _generate_labels(self, input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        prompt = _post_processing_chat(self._create_prompt(_pre_processing_chat(sample["conversations"])))
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self._generate_labels(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@register("dataset", "minimind_dpo")
class DPODataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 4096) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def _loss_mask(self, input_ids: list[int]) -> list[int]:
        mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start, end = i + len(self.bos_id), i + len(self.bos_id)
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        def _enc(conversations: list) -> tuple[list[int], list[int], list[int]]:
            prompt = _post_processing_chat(
                self.tokenizer.apply_chat_template(conversations, tokenize=False,
                                                   add_generation_prompt=False)
            )
            enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                                 padding="max_length")
            ids = enc["input_ids"]
            return ids[:-1], ids[1:], self._loss_mask(ids)[1:]

        cx, cy, cm = _enc(s["chosen"])
        rx, ry, rm = _enc(s["rejected"])
        return {
            "x_chosen": torch.tensor(cx, dtype=torch.long),
            "y_chosen": torch.tensor(cy, dtype=torch.long),
            "mask_chosen": torch.tensor(cm, dtype=torch.long),
            "x_rejected": torch.tensor(rx, dtype=torch.long),
            "y_rejected": torch.tensor(ry, dtype=torch.long),
            "mask_rejected": torch.tensor(rm, dtype=torch.long),
        }


@register("dataset", "minimind_rlaif")
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 1024,
                 thinking_ratio: float = 0.5) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.thinking_ratio = float(thinking_ratio)
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, str]:
        sample = self.samples[index]
        convs = _pre_processing_chat(sample["conversations"])
        use_thinking = random.random() < self.thinking_ratio
        prompt = self.tokenizer.apply_chat_template(
            convs[:-1], tokenize=False, open_thinking=use_thinking, add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": ""}


# ---------------------------------------------------------------------------
# Collator — stacks fixed-length (input_ids, labels) dicts; no padding needed
# ---------------------------------------------------------------------------

@register("collator", "minimind_pad")
class MiniMindCollator:
    """Stack pre-padded (input_ids, labels) samples from MiniMind datasets."""

    def __init__(self, pad_id: int | None = None) -> None:
        pass  # pad_id injected by data module; unused (data already padded)

    def __call__(self, samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            k: torch.stack([s[k] for s in samples])
            for k in samples[0]
        }


__all__ = ["PretrainDataset", "SFTDataset", "DPODataset", "RLAIFDataset", "MiniMindCollator"]
