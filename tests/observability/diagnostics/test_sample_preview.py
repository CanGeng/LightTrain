"""SamplePreviewCallback dumps the first N batches as decoded text."""

from __future__ import annotations

import torch

from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
from lighttrain.builtin_plugins.diagnostics.sample_preview import SamplePreviewCallback


class _DM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _Trainer:
    def __init__(self, run_dir, dm):
        self._run_dir = run_dir
        self.data_module = dm


def test_sample_preview_writes_first_n(tmp_path):
    tok = ByteTokenizer()
    cb = SamplePreviewCallback(max_batches=2)
    cb.on_train_start(trainer=_Trainer(tmp_path, _DM(tok)), ctx=None)
    for i in range(4):
        batch = {
            "input_ids": torch.tensor([[65, 66, 67, 68]], dtype=torch.long),
            "labels": torch.tensor([[65, 66, -100, 68]], dtype=torch.long),
        }
        cb.on_train_batch_start(step=i, batch=batch)
    out = tmp_path / "diagnostics" / "sample_preview"
    files = sorted(out.glob("*.txt"))
    assert len(files) == 2
    body = files[0].read_text(encoding="utf-8")
    assert "step=" in body
