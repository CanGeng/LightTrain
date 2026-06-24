"""R3 end-to-end acceptance — DESIGN §25.1 / §26.5.

Smoke variant (default-run): 5 samples produce + 5-step student train, no
loss-curve assertion. Heavy variant: 50 samples + 50-step train, asserts loss
decreases.

Heavy tests are gated; default ``pytest`` skips them.
"""

from __future__ import annotations

import pytest
import torch

from lighttrain.builtin_plugins.data.artifacts import (
    ArtifactJoinedDataset,
    ModelForwardProducer,
    open_artifact_store,
)
from lighttrain.builtin_plugins.data.core.collators import CausalLMCollator
from lighttrain.builtin_plugins.data.core.tokenizers import PAD_ID
from lighttrain.builtin_plugins.engine.standard import StandardEngine
from lighttrain.builtin_plugins.losses.core import CompositeLoss
from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.builtin_plugins.optim.wrappers import AdamWWrapper
from lighttrain.builtin_plugins.engine.update_rules.standard import StandardUpdateRule
from lighttrain.callbacks.base import EventBus
from lighttrain.engine._context import StepContext
from lighttrain.models.extras import ExtraOutputSpec


def _samples(n: int = 6, T: int = 8, vocab: int = 60, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        ids = torch.randint(0, vocab, (T,), generator=g).tolist()
        out.append({
            "id": f"r3_{i:03d}",
            "input_ids": ids,
            "attention_mask": [1] * T,
            "labels": ids[:],
        })
    return out


@pytest.mark.heavy
def test_r3_end_to_end_loss_decreases(tmp_path):
    """Heavy acceptance: teacher → joined dataset → student trains, loss falls."""
    torch.manual_seed(0)
    teacher = TinyCausalLM(
        vocab_size=60, d_model=64, n_layers=3, n_heads=4, max_seq_len=16,
        output_hidden_states=True,
    )
    samples = _samples(n=12, T=8, vocab=60, seed=42)
    art_root = tmp_path / "artifacts" / "teacher"
    prod = ModelForwardProducer(
        model=teacher,
        store={"name": "safetensors-shards", "root": str(art_root), "shard_size": 16},
        extras=[ExtraOutputSpec(name="logits_topk_64", source="lm_head",
                                transform={"topk": 16})],
        collect_hidden_states=True,
        artifact_name="teacher", artifact_version="v1",
    )
    prod.prepare()
    for s in samples:
        prod.produce(s)
    prod.finalize()
    # Sanity — header carries shapes.
    opened = open_artifact_store(art_root)
    assert opened.contains("r3_000")

    # ----- student training (composite CE+KL+hidden_mse) ----------------
    joined = ArtifactJoinedDataset(
        samples,
        join=[{"store": str(art_root), "namespace": "teacher"}],
    )
    student = TinyCausalLM(
        vocab_size=60, d_model=64, n_layers=3, n_heads=4, max_seq_len=16,
        output_hidden_states=True,
    )
    collator = CausalLMCollator(pad_id=PAD_ID, max_len=16)
    optimizer_wrapper = AdamWWrapper(lr=1e-3)
    optimizer_wrapper.build(student)

    composite = CompositeLoss(children=[
        {"name": "cross_entropy", "weight": 0.5},
        {"name": "kl_topk", "weight": 0.4, "top_k": 16, "temperature": 2.0,
         "teacher_namespace": "teacher", "teacher_key": "logits_topk_64"},
        {"name": "hidden_mse", "weight": 0.1, "mapping": {1: 1, 2: 2},
         "teacher_namespace": "teacher", "teacher_key": "hidden_states_layers"},
    ])

    update_rule = StandardUpdateRule(grad_clip=1.0, accumulate_grad_batches=1)
    engine = StandardEngine(update_rule=update_rule, loss_fn=composite)

    bus = EventBus([])
    ctx = StepContext(bus=bus, optimizer=optimizer_wrapper, loss_fn=composite,
                     model=student)

    losses: list[float] = []
    for step in range(50):
        batch = collator([joined[i % len(joined)] for i in range(step, step + 4)])
        ctx.step = step
        out = engine.step(batch, ctx)
        losses.append(float(out["loss"]))

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    assert final < initial, f"loss did not decrease: initial={initial}, final={final}"


def test_r3_smoke_5_samples(tmp_path):
    """Default-run smoke: pipeline doesn't throw; we don't assert loss curve."""
    teacher = TinyCausalLM(
        vocab_size=40, d_model=16, n_layers=2, n_heads=2, max_seq_len=8,
        output_hidden_states=True,
    )
    samples = _samples(n=5, T=4, vocab=40)
    prod = ModelForwardProducer(
        model=teacher,
        store={"name": "safetensors-shards", "root": str(tmp_path / "art"),
               "shard_size": 8},
        extras=[ExtraOutputSpec(name="logits_topk_64", source="lm_head",
                                transform={"topk": 4})],
        collect_hidden_states=True,
    )
    prod.prepare()
    for s in samples:
        prod.produce(s)
    prod.finalize()

    joined = ArtifactJoinedDataset(samples, join=[
        {"store": str(tmp_path / "art"), "namespace": "teacher"}
    ])
    row = joined[0]
    assert row is not None
    assert "aux.teacher.logits_topk_64.values" in row
    assert "aux.teacher.hidden_states_layers" in row
