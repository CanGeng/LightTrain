"""Edge-case tests for ``lighttrain.builtin_plugins.data.processors.audio``.

Drives the previously-uncovered branches toward 100 %:

* ``_mel_filterbank`` degenerate-band guard (``f_center == f_left`` widening,
  line 47) via a squeezed n_fft / n_mels configuration.
* ``_load_waveform`` source dispatch:
    - ``list`` / ``tuple`` -> float32 array (lines 79-80);
    - ``str`` / ``Path`` ``.wav`` via stdlib :mod:`wave` for 1/2/4-byte widths
      (lines 81-92);
    - non-``.wav`` path -> :mod:`soundfile` read (lines 93-94, 99, 104), with
      the sample-rate-mismatch ``ValueError`` (lines 100-103);
    - unsupported type -> ``TypeError`` (line 105).
* ``HFAudioProcessor.__init__`` field wiring (lines 173-176); lazy
  ``_ensure_extractor`` import + caching (lines 179-185); ``__call__``
  ``input_features`` squeeze, ``input_values`` fallthrough, no-squeeze when
  batch dim != 1 (lines 187-197); and the unexpected-keys ``RuntimeError``
  (lines 198-200).

The :mod:`soundfile` / :mod:`transformers` imports are stubbed through
``monkeypatch.setitem(sys.modules, ...)`` so no optional dependency or network
download is required (``soundfile`` is genuinely absent in this env; the real
``transformers`` is shadowed only for the duration of a test).
"""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from lighttrain.builtin_plugins.data.processors.audio import (
    HFAudioProcessor,
    MelSpectrogramProcessor,
    _load_waveform,
    _mel_filterbank,
)

# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samples: np.ndarray, *, sampwidth: int) -> None:
    """Write a mono PCM ``.wav`` of the given sample width (bytes)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(16_000)
        w.writeframes(samples.tobytes())


class _FakeSoundfile(types.ModuleType):
    """Stand-in for the absent :mod:`soundfile` package."""

    def __init__(self, *, arr: np.ndarray, sr: int) -> None:
        super().__init__("soundfile")
        self._arr = arr
        self._sr = sr
        self.read_calls: list[tuple[str, str]] = []

    def read(self, path, dtype="float32"):  # noqa: D401 — mirror sf.read signature
        self.read_calls.append((str(path), dtype))
        return self._arr, self._sr


class _FakeExtractor:
    """Mimic ``transformers.AutoFeatureExtractor`` instances."""

    last_kwargs: dict | None = None

    def __init__(self, out: dict) -> None:
        self._out = out
        self.seen_sampling_rate: int | None = None
        self.seen_return_tensors: str | None = None

    def __call__(self, wav, *, sampling_rate=None, return_tensors=None):
        self.seen_sampling_rate = sampling_rate
        self.seen_return_tensors = return_tensors
        return dict(self._out)


def _fake_transformers_module(out: dict) -> tuple[types.ModuleType, list]:
    """A fake ``transformers`` module exporting ``AutoFeatureExtractor``.

    Returns the module plus a list recording each ``from_pretrained`` call.
    """
    calls: list = []
    mod = types.ModuleType("transformers")

    class AutoFeatureExtractor:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls.append((name, kw))
            return _FakeExtractor(out)

    mod.AutoFeatureExtractor = AutoFeatureExtractor  # type: ignore[attr-defined]
    return mod, calls


# ---------------------------------------------------------------------------
# _mel_filterbank — degenerate-band guard (line 47)
# ---------------------------------------------------------------------------


def test_invariant_filterbank_handles_degenerate_bands():
    """A squeezed (small n_fft, many mels) config collapses adjacent bins so
    that ``f_center == f_left``; the guard widens ``f_right`` (line 47) and the
    filterbank still has the requested shape with finite weights."""
    fb = _mel_filterbank(
        sample_rate=16_000, n_fft=64, n_mels=40, f_min=0.0, f_max=8_000.0
    )
    assert fb.shape == (40, 64 // 2 + 1)
    assert fb.dtype == np.float32
    assert np.isfinite(fb).all()
    # The squeeze genuinely produced degenerate bands (the trigger for line 47).
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + 8_000.0 / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, 42)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bins = np.clip(np.floor((64 + 1) * hz_pts / 16_000).astype(int), 0, 32)
    assert any(bins[m] == bins[m - 1] for m in range(1, 41))


def test_invariant_filterbank_non_degenerate_weights_are_triangular():
    """A roomy config (large n_fft) keeps bands separated and yields the
    expected triangular peak of 1.0 per mel row."""
    fb = _mel_filterbank(
        sample_rate=16_000, n_fft=1024, n_mels=40, f_min=0.0, f_max=8_000.0
    )
    assert fb.shape == (40, 513)
    # Every mel filter peaks at (approximately) 1.0.
    assert fb.max(axis=1) == pytest.approx(np.ones(40), abs=1e-6)


# ---------------------------------------------------------------------------
# _load_waveform — source dispatch
# ---------------------------------------------------------------------------


def test_invariant_load_waveform_ndarray_passthrough_no_copy():
    """An ndarray is returned cast to float32 without an unnecessary copy."""
    src = np.arange(4, dtype=np.float32)
    out = _load_waveform(src, 16_000)
    assert out.dtype == np.float32
    assert out is src  # copy=False keeps the same buffer when dtype matches


def test_invariant_load_waveform_ndarray_casts_dtype():
    """A non-float32 ndarray is cast to float32 (copy permitted)."""
    out = _load_waveform(np.arange(3, dtype=np.int16), 16_000)
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("ctor", [list, tuple])
def test_invariant_load_waveform_list_and_tuple(ctor):
    """list / tuple sources (lines 79-80) become float32 arrays."""
    out = _load_waveform(ctor([1, 2, 3]), 16_000)
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("sampwidth", "dtype", "full_scale"),
    [
        (1, np.int8, 2 ** 7),
        (2, np.int16, 2 ** 15),
        (4, np.int32, 2 ** 31),
    ],
)
def test_invariant_load_waveform_wav_widths(
    tmp_path, sampwidth, dtype, full_scale
):
    """``.wav`` files of every supported width (lines 81-92) decode to
    ``[-1, 1]``-normalised float32; full-scale negative maps to exactly -1.0."""
    samples = np.array([0, -full_scale], dtype=dtype)
    p = tmp_path / "tone.wav"
    _write_wav(p, samples, sampwidth=sampwidth)
    out = _load_waveform(p, 16_000)
    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(-1.0)


def test_invariant_load_waveform_wav_accepts_str_path(tmp_path):
    """The ``.wav`` branch accepts a plain ``str`` path (str(src) on line 82)."""
    samples = np.array([16384, -16384], dtype=np.int16)
    p = tmp_path / "s.wav"
    _write_wav(p, samples, sampwidth=2)
    out = _load_waveform(str(p), 16_000)
    assert out == pytest.approx(np.array([0.5, -0.5], dtype=np.float32))


def test_invariant_load_waveform_soundfile_path(monkeypatch, tmp_path):
    """Non-``.wav`` paths route through :mod:`soundfile` (lines 93-94, 99, 104);
    soundfile is stubbed because it is not installed in this env."""
    arr = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    fake = _FakeSoundfile(arr=arr, sr=16_000)
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    out = _load_waveform(tmp_path / "clip.flac", 16_000)
    assert out.dtype == np.float32
    assert out == pytest.approx(arr)
    # sf.read was actually consulted with the float32 dtype request.
    assert fake.read_calls == [(str(tmp_path / "clip.flac"), "float32")]


def test_invariant_load_waveform_soundfile_sample_rate_mismatch(
    monkeypatch, tmp_path
):
    """A soundfile sample rate != the processor's raises ValueError
    (lines 100-103)."""
    fake = _FakeSoundfile(arr=np.zeros(2, dtype=np.float32), sr=22_050)
    monkeypatch.setitem(sys.modules, "soundfile", fake)
    with pytest.raises(ValueError, match="sample rate 22050 != processor 16000"):
        _load_waveform(str(tmp_path / "clip.ogg"), 16_000)


def test_invariant_load_waveform_rejects_unsupported_type():
    """An unsupported source type raises a typed TypeError (line 105)."""
    with pytest.raises(TypeError, match="unsupported audio source: int"):
        _load_waveform(123, 16_000)


# ---------------------------------------------------------------------------
# HFAudioProcessor.__init__ (lines 173-176)
# ---------------------------------------------------------------------------


def test_invariant_hf_init_records_fields_and_defers_extractor():
    """Constructor stores config, copies from_pretrained kwargs, and leaves the
    extractor uninitialised (lines 173-176)."""
    kw = {"cache_dir": "/tmp/x"}
    proc = HFAudioProcessor(
        model_name_or_path="openai/whisper-tiny",
        sample_rate=8_000,
        from_pretrained_kwargs=kw,
    )
    assert proc.model_name_or_path == "openai/whisper-tiny"
    assert proc.sample_rate == 8_000
    assert proc.modality == "audio"
    assert proc._extractor is None
    # The kwargs are *copied*, not aliased.
    assert proc._fp_kwargs == kw
    assert proc._fp_kwargs is not kw


def test_invariant_hf_init_none_kwargs_becomes_empty_dict():
    """Passing ``from_pretrained_kwargs=None`` yields an empty dict (line 175)."""
    proc = HFAudioProcessor(model_name_or_path="m")
    assert proc._fp_kwargs == {}
    assert proc.sample_rate == 16_000  # default


# ---------------------------------------------------------------------------
# HFAudioProcessor._ensure_extractor + __call__ (lines 179-198)
# ---------------------------------------------------------------------------


def test_invariant_hf_call_input_features_squeezed(monkeypatch):
    """``input_features`` with a leading batch dim of 1 is squeezed and the
    extractor is constructed via from_pretrained with the stored kwargs
    (lines 179-197)."""
    out = {"input_features": np.ones((1, 3, 4), dtype=np.float32)}
    mod, calls = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(
        model_name_or_path="whisper-tiny", from_pretrained_kwargs={"k": 1}
    )
    result = proc(np.zeros(100, dtype=np.float32))
    assert result["modality"] == "audio"
    assert result["audio_features"].shape == (3, 4)
    assert result["audio_features"].dtype == np.float32
    assert calls == [("whisper-tiny", {"k": 1})]


def test_invariant_hf_ensure_extractor_caches_singleton(monkeypatch):
    """``_ensure_extractor`` imports + builds once, then reuses the instance
    (the ``self._extractor is None`` guard on line 179)."""
    out = {"input_features": np.ones((1, 2, 2), dtype=np.float32)}
    mod, calls = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(model_name_or_path="m")
    first = proc._ensure_extractor()
    second = proc._ensure_extractor()
    assert first is second
    assert len(calls) == 1  # from_pretrained called exactly once


def test_invariant_hf_call_passes_sampling_rate_and_return_tensors(monkeypatch):
    """``__call__`` forwards ``sampling_rate`` / ``return_tensors='np'`` to the
    extractor (line 190)."""
    out = {"input_features": np.ones((1, 1, 1), dtype=np.float32)}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(model_name_or_path="m", sample_rate=8_000)
    proc(np.zeros(10, dtype=np.float32))
    ex = proc._extractor
    assert ex is not None
    assert ex.seen_sampling_rate == 8_000
    assert ex.seen_return_tensors == "np"


def test_invariant_hf_call_input_values_fallthrough(monkeypatch):
    """When ``input_features`` is absent the loop falls through to
    ``input_values`` (line 192 iteration)."""
    out = {"input_values": np.ones((1, 5), dtype=np.float32)}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(model_name_or_path="m")
    result = proc(np.zeros(10, dtype=np.float32))
    assert result["audio_features"].shape == (5,)


def test_invariant_hf_call_no_squeeze_when_batch_gt_one(monkeypatch):
    """A leading dim != 1 is left intact (the ``feats.shape[0] == 1`` branch on
    line 195 is False)."""
    out = {"input_features": np.ones((2, 3, 4), dtype=np.float32)}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(model_name_or_path="m")
    result = proc(np.zeros(10, dtype=np.float32))
    assert result["audio_features"].shape == (2, 3, 4)


def test_invariant_hf_call_unexpected_keys_raises(monkeypatch):
    """An extractor returning neither known key raises a RuntimeError naming the
    keys it saw (lines 198-200)."""
    out = {"weird_key": np.ones((1, 1), dtype=np.float32)}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFAudioProcessor(model_name_or_path="m")
    with pytest.raises(RuntimeError, match="unexpected keys.*weird_key"):
        proc(np.zeros(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# End-to-end sanity through MelSpectrogramProcessor with a wav source
# ---------------------------------------------------------------------------


def test_invariant_mel_processor_consumes_wav_file(tmp_path):
    """The processor reads a ``.wav`` source end-to-end and yields finite
    log-mel features of shape (n_mels, T)."""
    np.random.seed(0)
    samples = (np.random.randint(-2000, 2000, size=4000)).astype(np.int16)
    p = tmp_path / "noise.wav"
    _write_wav(p, samples, sampwidth=2)
    proc = MelSpectrogramProcessor(n_mels=16)
    out = proc(p)
    assert out["modality"] == "audio"
    assert out["audio_features"].shape[0] == 16
    assert np.isfinite(out["audio_features"]).all()
