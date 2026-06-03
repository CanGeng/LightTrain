"""StatefulResumableSampler resume tests."""

from __future__ import annotations

from lighttrain.builtin_plugins.data.samplers.stateful_resumable import StatefulResumableSampler
from lighttrain.builtin_plugins.data.samplers.length_grouped import LengthGroupedSampler
from lighttrain.builtin_plugins.data.samplers.curriculum import CurriculumSampler


class _FakeDataset:
    def __init__(self, lengths):
        self._lengths = list(lengths)

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, i):
        return {"input_ids": [0] * self._lengths[i]}


def test_stateful_resumable_round_trip():
    ds = _FakeDataset([4] * 100)
    s1 = StatefulResumableSampler(ds, chunk_size=10, seed=7, shuffle=True)
    it = iter(s1)
    consumed = [next(it) for _ in range(25)]
    sd = s1.state_dict()
    assert sd["chunk_idx"] == 2
    assert sd["consumed_in_chunk"] == 5

    s2 = StatefulResumableSampler(ds, chunk_size=10, seed=7, shuffle=True)
    s2.load_state_dict(sd)
    rest = list(s2)
    assert len(consumed) + len(rest) == 100
    # No duplicates between consumed and rest.
    assert set(consumed).isdisjoint(rest)
    # Together we cover every index.
    assert sorted(consumed + rest) == list(range(100))


def test_stateful_resumable_chunk_boundaries():
    ds = _FakeDataset([1] * 50)
    s = StatefulResumableSampler(ds, chunk_size=10)
    boundaries = list(s.chunk_boundaries())
    assert boundaries == [0, 10, 20, 30, 40]


def test_length_grouped_sampler_within_block():
    lengths = [10, 200, 30, 5, 100, 70, 8, 250]
    ds = _FakeDataset(lengths)
    s = LengthGroupedSampler(ds, batch_size=2, mega_batch_mult=2, descending=True)
    out = list(iter(s))
    assert sorted(out) == list(range(len(lengths)))
    # Each block of 4 should be roughly sorted by length descending.
    block = out[:4]
    block_lens = [lengths[i] for i in block]
    assert block_lens == sorted(block_lens, reverse=True)


def test_curriculum_sampler_widens_band():
    lengths = list(range(1, 101))
    ds = _FakeDataset(lengths)
    s = CurriculumSampler(ds, schedule="linear", p_start=0.2, p_end=1.0, steps=10)
    s.set_step(0)
    early = list(iter(s))
    s.set_step(10)
    late = list(iter(s))
    assert len(early) <= len(late)
