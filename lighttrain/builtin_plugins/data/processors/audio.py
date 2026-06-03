"""Audio processors.

* ``MelSpectrogramProcessor`` — pure numpy STFT + closed-form mel filterbank.
  Hermetic; no librosa/torchaudio dependency. Output is log-mel (C=n_mels, T).

* ``HFAudioProcessor`` — wraps ``transformers.AutoFeatureExtractor`` (e.g.
  Whisper / Wav2Vec2). Lazy import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lighttrain.registry import register


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    mel_min = _hz_to_mel(np.array([f_min]))[0]
    mel_max = _hz_to_mel(np.array([f_max]))[0]
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        if f_center == f_left:
            f_right = max(f_right, f_center + 1)
        for k in range(f_left, f_center):
            denom = max(1, f_center - f_left)
            fb[m - 1, k] = (k - f_left) / denom
        for k in range(f_center, f_right):
            denom = max(1, f_right - f_center)
            fb[m - 1, k] = (f_right - k) / denom
    return fb


def _stft(
    waveform: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    pad = n_fft // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_frames, n_fft),
        strides=(padded.strides[0] * hop_length, padded.strides[0]),
    ) * window
    spec = np.fft.rfft(frames, axis=1)
    return spec.T  # (freq, frames)


def _load_waveform(src: Any, sample_rate: int) -> np.ndarray:
    if isinstance(src, np.ndarray):
        return src.astype(np.float32, copy=False)
    if isinstance(src, (list, tuple)):
        return np.asarray(src, dtype=np.float32)
    if isinstance(src, (str, Path)):
        path = str(src)
        if path.endswith(".wav"):
            import wave

            with wave.open(path, "rb") as w:
                n = w.getnframes()
                raw = w.readframes(n)
                sw = w.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
            arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            return arr / float(2 ** (8 * sw - 1))
        try:
            import soundfile as sf  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"reading {path!r} requires soundfile (install soundfile)"
            ) from exc
        arr, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(
                f"audio file sample rate {sr} != processor {sample_rate}"
            )
        return arr.astype(np.float32, copy=False)
    raise TypeError(f"unsupported audio source: {type(src).__name__}")


@register("processor", "mel_spectrogram")
class MelSpectrogramProcessor:
    """Pure numpy log-mel spectrogram processor.

    Output::

        {
          "audio_features": np.ndarray (n_mels, T) float32,
          "modality": "audio",
        }
    """

    modality = "audio"

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float | None = None,
        log_offset: float = 1e-10,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max) if f_max is not None else self.sample_rate / 2.0
        self.log_offset = float(log_offset)
        self._fb = _mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

    def __call__(self, audio: Any, **_: Any) -> dict[str, Any]:
        wav = _load_waveform(audio, self.sample_rate)
        spec = _stft(wav, n_fft=self.n_fft, hop_length=self.hop_length)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        mel = self._fb @ power  # (n_mels, T)
        log_mel = np.log(mel + self.log_offset).astype(np.float32)
        return {
            "audio_features": log_mel,
            "modality": "audio",
        }


@register("processor", "hf_audio")
class HFAudioProcessor:
    """Wrap ``transformers.AutoFeatureExtractor`` for audio (lazy import)."""

    modality = "audio"

    def __init__(
        self,
        *,
        model_name_or_path: str,
        sample_rate: int = 16_000,
        from_pretrained_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.sample_rate = int(sample_rate)
        self._fp_kwargs = dict(from_pretrained_kwargs or {})
        self._extractor: Any | None = None

    def _ensure_extractor(self) -> Any:
        if self._extractor is None:
            from transformers import AutoFeatureExtractor  # type: ignore

            self._extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name_or_path, **self._fp_kwargs
            )
        return self._extractor

    def __call__(self, audio: Any, **_: Any) -> dict[str, Any]:
        ex = self._ensure_extractor()
        wav = _load_waveform(audio, self.sample_rate)
        out = ex(wav, sampling_rate=self.sample_rate, return_tensors="np")
        # Most HF audio extractors return "input_features" or "input_values".
        for key in ("input_features", "input_values"):
            if key in out:
                feats = np.asarray(out[key], dtype=np.float32)
                if feats.shape[0] == 1:
                    feats = feats[0]
                return {"audio_features": feats, "modality": "audio"}
        raise RuntimeError(
            f"HF feature extractor returned unexpected keys: {list(out.keys())!r}"
        )


__all__ = ["MelSpectrogramProcessor", "HFAudioProcessor"]
