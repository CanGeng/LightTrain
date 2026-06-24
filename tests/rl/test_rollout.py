"""Edge-case tests for ``lighttrain.builtin_plugins.rl.rollout``.

Pins the rollout collection seam (``HFGenerateBackend`` + ``RolloutEngine``):

* **generate() kwarg assembly**: ``do_sample`` toggles temperature/top_p; the
  optional ``pad_token_id`` / ``eos_token_id`` / ``attention_mask`` keys are only
  emitted when supplied; ``num_return_sequences`` always passed.
* **generate() runs under no_grad** and forwards ``input_ids`` to ``model.generate``.
* **RolloutEngine.rollout** end to end: produces ``B*G`` episodes; labels are
  ``ignore_index`` over the prompt span and the response tokens after it; the
  full-sequence attention mask is all ones; log_probs are length-``T_full`` with a
  leading 0 and gathered log-softmax values elsewhere.
* **group_id assignment**: ``group_offset + i // G``.
* **eval/train toggle**: ``model.eval()`` during collection, restored to
  ``train()`` only when the model was training beforehand.
* **logits extraction**: both the ``out.outputs["logits"]`` (ModelOutput-like)
  and the ``out["logits"]`` (plain dict) branches.
* **reward_fn wiring**: called with the G-expanded prompts and the response slice;
  returned scores land on ``episode.reward`` as floats.
* **registry**: ``HFGenerateBackend`` is registered under ``("rl_backend",
  "hf_generate")``.

All stubs are deterministic (no sampling RNG is exercised — the stub
``model.generate`` just returns a fixed tensor).
"""

from __future__ import annotations

import math

import pytest
import torch

from lighttrain.builtin_plugins.rl.buffers import Episode
from lighttrain.builtin_plugins.rl.rollout import HFGenerateBackend, RolloutEngine

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingGenModel:
    """Stub policy whose ``generate`` records kwargs and returns a fixed tensor.

    ``no_grad_seen`` captures whether autograd was disabled at call time, so the
    test can confirm ``generate`` ran inside ``torch.no_grad()``.
    """

    def __init__(self, sequences: torch.Tensor) -> None:
        self._sequences = sequences
        self.last_kwargs: dict | None = None
        self.last_input_ids: torch.Tensor | None = None
        self.no_grad_seen: bool | None = None

    def generate(self, *, input_ids: torch.Tensor, **kwargs):
        self.last_input_ids = input_ids
        self.last_kwargs = dict(kwargs)
        self.no_grad_seen = not torch.is_grad_enabled()
        return self._sequences


class _ModelOutput:
    """Minimal ModelOutput-like wrapper exposing an ``.outputs`` mapping."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.outputs = {"logits": logits}


class _PolicyModel:
    """Stub policy used by ``RolloutEngine.rollout``.

    * ``generate`` returns a fixed full-sequence tensor (prompt + response).
    * ``__call__`` returns logits, wrapped per ``output_style``:
        - ``"modeloutput"`` → object with ``.outputs["logits"]``
        - ``"dict"`` → plain ``{"logits": ...}``
    * tracks ``training`` flag transitions via ``eval``/``train``.
    * logits are deterministic: a fixed (vocab) bias broadcast over time so the
      gathered log-probs are computable analytically.
    """

    def __init__(
        self,
        full_seqs: torch.Tensor,
        vocab_size: int,
        *,
        output_style: str = "modeloutput",
        start_training: bool = True,
    ) -> None:
        self._full_seqs = full_seqs
        self._vocab = vocab_size
        self._style = output_style
        self.training = start_training
        self.eval_calls = 0
        self.train_calls = 0
        # Fixed per-vocab bias → logits[t, v] = bias[v] for all t.
        self._bias = torch.arange(vocab_size, dtype=torch.float32) * 0.1

    # -- generation -------------------------------------------------------
    def generate(self, *, input_ids, **kwargs):
        return self._full_seqs

    # -- forward ----------------------------------------------------------
    def __call__(self, *, input_ids):
        # input_ids: (1, T_full) → logits (1, T_full, V)
        t_full = input_ids.size(1)
        logits = self._bias.unsqueeze(0).unsqueeze(0).expand(1, t_full, self._vocab).clone()
        if self._style == "modeloutput":
            return _ModelOutput(logits)
        return {"logits": logits}

    # -- mode toggles -----------------------------------------------------
    def eval(self):
        self.eval_calls += 1
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.train_calls += 1
        self.training = mode
        return self


def _expected_token_logprob(vocab_size: int, token_id: int) -> float:
    """log_softmax over the fixed bias arange*0.1, picked at ``token_id``."""
    bias = torch.arange(vocab_size, dtype=torch.float32) * 0.1
    lp = torch.log_softmax(bias, dim=-1)
    return float(lp[token_id])


# ===========================================================================
# HFGenerateBackend.__init__ / generate
# ===========================================================================


def test_invariant_init_coerces_field_types():
    """Constructor casts numeric/bool args to their declared types."""
    be = HFGenerateBackend(
        max_new_tokens="8",  # type: ignore[arg-type]
        do_sample=1,  # type: ignore[arg-type]
        temperature="0.5",  # type: ignore[arg-type]
        top_p="0.9",  # type: ignore[arg-type]
        num_return_sequences="3",  # type: ignore[arg-type]
    )
    assert be.max_new_tokens == 8 and isinstance(be.max_new_tokens, int)
    assert be.do_sample is True
    assert be.temperature == pytest.approx(0.5)
    assert be.top_p == pytest.approx(0.9)
    assert be.num_return_sequences == 3


def test_invariant_generate_kwargs_sampling_branch():
    """do_sample=True emits temperature + top_p (lines 73-75)."""
    seqs = torch.zeros(2, 5, dtype=torch.long)
    model = _RecordingGenModel(seqs)
    be = HFGenerateBackend(
        max_new_tokens=4, do_sample=True, temperature=0.7, top_p=0.9,
        num_return_sequences=2,
    )
    out = be.generate(model, torch.zeros(1, 3, dtype=torch.long))
    assert out is seqs
    kw = model.last_kwargs
    assert kw["max_new_tokens"] == 4
    assert kw["do_sample"] is True
    assert kw["num_return_sequences"] == 2
    assert kw["temperature"] == pytest.approx(0.7)
    assert kw["top_p"] == pytest.approx(0.9)


def test_invariant_generate_kwargs_greedy_omits_sampling():
    """do_sample=False does NOT emit temperature/top_p (branch on line 73)."""
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=4, do_sample=False, temperature=2.0, top_p=0.3)
    be.generate(model, torch.zeros(1, 3, dtype=torch.long))
    kw = model.last_kwargs
    assert kw["do_sample"] is False
    assert "temperature" not in kw
    assert "top_p" not in kw


def test_invariant_generate_omits_optional_ids_when_none():
    """pad_token_id / eos_token_id default None → keys absent (lines 76-79 skipped)."""
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=2)
    be.generate(model, torch.zeros(1, 3, dtype=torch.long))
    kw = model.last_kwargs
    assert "pad_token_id" not in kw
    assert "eos_token_id" not in kw
    assert "attention_mask" not in kw


def test_invariant_generate_emits_optional_ids_when_set():
    """pad_token_id / eos_token_id emitted when provided (lines 76-79)."""
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=2, pad_token_id=0, eos_token_id=2)
    be.generate(model, torch.zeros(1, 3, dtype=torch.long))
    kw = model.last_kwargs
    assert kw["pad_token_id"] == 0
    assert kw["eos_token_id"] == 2


def test_pin_current_behavior_pad_token_id_zero_is_emitted():
    """Pin: pad_token_id=0 is emitted because the guard is ``is not None``.

    A naive ``if self.pad_token_id`` would drop a legitimate id of 0; the source
    correctly uses ``is not None`` (line 76). Pin this so a regression to a
    truthiness check is caught.
    """
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=2, pad_token_id=0, eos_token_id=0)
    be.generate(model, torch.zeros(1, 3, dtype=torch.long))
    assert model.last_kwargs["pad_token_id"] == 0
    assert model.last_kwargs["eos_token_id"] == 0


def test_invariant_generate_passes_attention_mask_when_given():
    """attention_mask forwarded only when not None (lines 80-81)."""
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=2)
    mask = torch.ones(1, 3, dtype=torch.long)
    be.generate(model, torch.zeros(1, 3, dtype=torch.long), attention_mask=mask)
    assert model.last_kwargs["attention_mask"] is mask


def test_invariant_generate_runs_under_no_grad_and_forwards_input_ids():
    """generate wraps the call in torch.no_grad and forwards input_ids (lines 83-84)."""
    model = _RecordingGenModel(torch.zeros(1, 5, dtype=torch.long))
    be = HFGenerateBackend(max_new_tokens=2)
    prompt = torch.arange(3).unsqueeze(0)
    assert torch.is_grad_enabled()  # outer context has grad enabled
    be.generate(model, prompt)
    assert model.no_grad_seen is True
    torch.testing.assert_close(model.last_input_ids, prompt)
    assert torch.is_grad_enabled()  # restored afterward


# ===========================================================================
# RolloutEngine.rollout — shapes, labels, log-probs, groups, rewards
# ===========================================================================


class _Backend:
    """Minimal rollout backend: fixed G + delegate generate to the policy."""

    def __init__(self, num_return_sequences: int, full_seqs: torch.Tensor) -> None:
        self.num_return_sequences = num_return_sequences
        self._full_seqs = full_seqs

    def generate(self, model, prompt_ids, prompt_mask):
        return self._full_seqs


def _make_engine(full_seqs, G, ignore_index=-100):
    return RolloutEngine(_Backend(G, full_seqs), ignore_index=ignore_index)


def test_invariant_rollout_produces_b_times_g_episodes():
    """rollout returns exactly B*G episodes (lines 138-139, 153, 185)."""
    B, G, T_prompt, T_resp, vocab = 2, 3, 4, 5, 7
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(B * G * T_full).reshape(B * G, T_full) % vocab
    prompt_ids = torch.arange(B * T_prompt).reshape(B, T_prompt) % vocab
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, G)

    eps = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0] * (B * G))
    assert len(eps) == B * G
    assert all(isinstance(e, Episode) for e in eps)


def test_invariant_rollout_labels_mask_prompt_and_keep_response():
    """labels are ignore_index for prompt span and the response ids after it.

    Covers lines 154-159 (seq slice, full_like(ignore_index), labels[T:]=resp).
    """
    _, G, T_prompt, T_resp, vocab = 1, 1, 3, 4, 11
    T_full = T_prompt + T_resp
    full_seqs = (torch.arange(T_full) + 1).reshape(1, T_full) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, G, ignore_index=-100)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])[0]
    seq = full_seqs[0]
    expected = torch.full_like(seq, -100)
    expected[T_prompt:] = seq[T_prompt:]
    torch.testing.assert_close(ep.labels, expected)
    # prompt positions all ignore_index
    assert torch.all(ep.labels[:T_prompt] == -100)


def test_invariant_rollout_attention_mask_all_ones_full_length():
    """attn mask is ones over the full sequence (line 162)."""
    T_prompt, T_resp, vocab = 2, 3, 9
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(T_full).reshape(1, T_full) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, 1)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [1.0])[0]
    torch.testing.assert_close(ep.attention_mask, torch.ones(T_full, dtype=full_seqs.dtype))
    assert ep.input_ids.size(0) == T_full


def test_invariant_rollout_log_probs_modeloutput_branch():
    """log_probs use ``out.outputs['logits']`` and have a leading-0 + gathered tail.

    Covers lines 165-182 (forward, .outputs branch, log_softmax, gather, leading 0).
    """
    T_prompt, T_resp, vocab = 2, 2, 5
    T_full = T_prompt + T_resp
    full_seqs = torch.tensor([[1, 2, 3, 4]])  # all < vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, output_style="modeloutput")
    engine = _make_engine(full_seqs, 1)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])[0]
    log_probs = ep.log_probs
    assert log_probs.shape == (T_full,)
    # First position pinned to 0.
    assert log_probs[0].item() == pytest.approx(0.0)
    # Positions 1.. correspond to gathered log-softmax at seq[1:].
    seq = full_seqs[0]
    for t in range(1, T_full):
        expected = _expected_token_logprob(vocab, int(seq[t]))
        assert log_probs[t].item() == pytest.approx(expected, abs=1e-5)


def test_invariant_rollout_log_probs_dict_branch():
    """Plain-dict model output drives the ``out['logits']`` branch (lines 169-171)."""
    T_prompt, T_resp, vocab = 1, 3, 6
    T_full = T_prompt + T_resp
    full_seqs = torch.tensor([[0, 1, 2, 3]])
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, output_style="dict")
    engine = _make_engine(full_seqs, 1)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])[0]
    assert ep.log_probs.shape == (T_full,)
    assert ep.log_probs[0].item() == pytest.approx(0.0)
    seq = full_seqs[0]
    for t in range(1, T_full):
        expected = _expected_token_logprob(vocab, int(seq[t]))
        assert ep.log_probs[t].item() == pytest.approx(expected, abs=1e-5)


def test_invariant_rollout_group_id_assignment_with_offset():
    """group_id = group_offset + i // G (line 184)."""
    B, G, T_prompt, T_resp, vocab = 2, 2, 2, 2, 8
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(B * G * T_full).reshape(B * G, T_full) % vocab
    prompt_ids = torch.arange(B * T_prompt).reshape(B, T_prompt) % vocab
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, G)

    eps = engine.rollout(
        model, prompt_ids, None, lambda p, r: [0.0] * (B * G), group_offset=10
    )
    # i=0,1 → group 10 ; i=2,3 → group 11
    assert [e.group_id for e in eps] == [10, 10, 11, 11]


def test_invariant_rollout_rewards_assigned_from_reward_fn():
    """reward_fn output is written onto episodes as floats (lines 199-202)."""
    B, G, T_prompt, T_resp, vocab = 1, 3, 2, 2, 7
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(B * G * T_full).reshape(B * G, T_full) % vocab
    prompt_ids = torch.arange(B * T_prompt).reshape(B, T_prompt) % vocab
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, G)

    captured = {}

    def reward_fn(p_exp, resp):
        captured["p_exp"] = p_exp
        captured["resp"] = resp
        return [1, 2, 3]  # ints → cast to float

    eps = engine.rollout(model, prompt_ids, None, reward_fn)
    assert [e.reward for e in eps] == [1.0, 2.0, 3.0]
    assert all(isinstance(e.reward, float) for e in eps)


def test_invariant_rollout_reward_fn_receives_expanded_prompts_and_responses():
    """reward_fn gets G-expanded prompts (line 197) and response slice (line 198)."""
    B, G, T_prompt, T_resp, vocab = 2, 2, 3, 2, 9
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(B * G * T_full).reshape(B * G, T_full) % vocab
    prompt_ids = torch.arange(B * T_prompt).reshape(B, T_prompt) % vocab
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, G)

    captured = {}

    def reward_fn(p_exp, resp):
        captured["p_exp"] = p_exp
        captured["resp"] = resp
        return [0.0] * (B * G)

    engine.rollout(model, prompt_ids, None, reward_fn)
    # Expanded prompts: repeat_interleave by G → (B*G, T_prompt).
    torch.testing.assert_close(
        captured["p_exp"], prompt_ids.repeat_interleave(G, dim=0)
    )
    # Responses: full_seqs[:, T_prompt:] → (B*G, T_resp).
    torch.testing.assert_close(captured["resp"], full_seqs[:, T_prompt:])
    assert captured["resp"].shape == (B * G, T_resp)


# ===========================================================================
# eval/train toggle (lines 141-147)
# ===========================================================================


def test_invariant_rollout_sets_eval_then_restores_train_when_training():
    """A training model is put in eval during collection, then back to train."""
    T_prompt, T_resp, vocab = 2, 2, 7
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(T_full).reshape(1, T_full) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, start_training=True)
    engine = _make_engine(full_seqs, 1)

    engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])
    assert model.eval_calls == 1
    assert model.train_calls == 1  # restored because was_training
    assert model.training is True


def test_pin_current_behavior_rollout_leaves_model_in_eval_when_not_training():
    """Pin: if the model started in eval, rollout does NOT call train().

    The ``finally`` only restores train when ``was_training`` was True (line 146);
    a model that began in eval is left in eval. Pin this behavior so a change that
    unconditionally re-trains is flagged.
    """
    T_prompt, T_resp, vocab = 2, 2, 7
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(T_full).reshape(1, T_full) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, start_training=False)
    engine = _make_engine(full_seqs, 1)

    engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])
    assert model.eval_calls == 1
    assert model.train_calls == 0
    assert model.training is False


def test_invariant_rollout_restores_train_even_if_generate_raises():
    """The finally block restores train() if backend.generate raises mid-rollout."""
    T_prompt, vocab = 2, 7
    full_seqs = torch.arange(T_prompt + 2).reshape(1, -1) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, start_training=True)

    class _Boom:
        num_return_sequences = 1

        def generate(self, model, prompt_ids, prompt_mask):
            raise RuntimeError("gen exploded")

    engine = RolloutEngine(_Boom())
    with pytest.raises(RuntimeError, match="gen exploded"):
        engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])
    # was_training=True → train() restored despite the exception.
    assert model.train_calls == 1
    assert model.training is True


# ===========================================================================
# Episode tensors are detached to CPU
# ===========================================================================


def test_invariant_rollout_episode_tensors_on_cpu():
    """All episode tensors are moved to CPU (lines 187-192 .cpu() calls)."""
    T_prompt, T_resp, vocab = 2, 2, 7
    T_full = T_prompt + T_resp
    full_seqs = torch.arange(T_full).reshape(1, T_full) % vocab
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab)
    engine = _make_engine(full_seqs, 1)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])[0]
    for t in (ep.input_ids, ep.attention_mask, ep.labels, ep.log_probs):
        assert t.device.type == "cpu"


def test_invariant_log_probs_match_manual_full_softmax():
    """End-to-end numeric check of the full log_probs vector vs a manual compute."""
    T_prompt, T_resp, vocab = 1, 4, 6
    T_full = T_prompt + T_resp
    full_seqs = torch.tensor([[0, 2, 4, 5, 1]])
    prompt_ids = full_seqs[:, :T_prompt].clone()
    model = _PolicyModel(full_seqs, vocab, output_style="modeloutput")
    engine = _make_engine(full_seqs, 1)

    ep = engine.rollout(model, prompt_ids, None, lambda p, r: [0.0])[0]
    bias = torch.arange(vocab, dtype=torch.float32) * 0.1
    lp_table = torch.log_softmax(bias, dim=-1)
    seq = full_seqs[0]
    expected = torch.zeros(T_full)
    for t in range(1, T_full):
        expected[t] = lp_table[int(seq[t])]
    torch.testing.assert_close(ep.log_probs, expected, atol=1e-5, rtol=1e-5)
    assert not any(math.isnan(x) for x in ep.log_probs.tolist())


# ===========================================================================
# Registry
# ===========================================================================


def test_invariant_backend_registered_under_rl_backend_hf_generate():
    """HFGenerateBackend is registered as ('rl_backend', 'hf_generate')."""
    from lighttrain.registry import get as registry_get

    cls = registry_get("rl_backend", "hf_generate")
    assert cls is HFGenerateBackend
