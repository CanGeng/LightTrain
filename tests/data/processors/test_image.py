"""Edge-case tests for ``lighttrain.builtin_plugins.data.processors.image``.

Drives the previously-uncovered branches toward 100 %:

* ``_open_image`` source dispatch:
    - ``PIL.Image.Image`` input → ``.convert("RGB")`` (line 31);
    - ``np.ndarray`` with float values (max <= 1.0) → scaled to uint8 (line 35);
    - ``np.ndarray`` with float values (max > 1.0) → cast to uint8 (line 35 else);
    - ``str`` / ``Path`` → ``Image.open`` (lines 37-38);
    - unsupported type → ``TypeError`` (line 39).
* ``HFImageProcessor.__init__`` field wiring (lines 114-116).
* ``HFImageProcessor._ensure_processor`` lazy load + caching (lines 119-120, 122, 125).
* ``HFImageProcessor.__call__`` full path including squeeze (batch==1, lines 134-135),
  no-squeeze (batch>1, line 133), and wrapping non-list inputs (line 130).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lighttrain.builtin_plugins.data.processors.image import (
    HFImageProcessor,
    SimpleImageProcessor,
    _open_image,
)

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


def _make_pil_rgb(w: int = 8, h: int = 8) -> Image.Image:
    """Return a plain RGB PIL image filled with mid-grey."""
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_pil_rgba(w: int = 4, h: int = 4) -> Image.Image:
    """Return an RGBA PIL image (needs convert to RGB)."""
    arr = np.full((h, w, 4), 200, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


class _FakeHFProcessor:
    """Stand-in for ``transformers.AutoImageProcessor`` instances."""

    def __init__(self, out: dict) -> None:
        self._out = out
        self.calls: list[dict] = []

    def __call__(self, images=None, return_tensors=None):
        self.calls.append({"images": images, "return_tensors": return_tensors})
        return dict(self._out)


def _fake_transformers_module(out: dict) -> tuple[types.ModuleType, list]:
    """Return a stub ``transformers`` module and a list tracking from_pretrained calls."""
    calls: list[tuple[str, dict]] = []
    mod = types.ModuleType("transformers")

    class AutoImageProcessor:
        @classmethod
        def from_pretrained(cls, name: str, **kw: object):
            calls.append((name, kw))
            return _FakeHFProcessor(out)

    mod.AutoImageProcessor = AutoImageProcessor  # type: ignore[attr-defined]
    return mod, calls


# ---------------------------------------------------------------------------
# _open_image — PIL.Image.Image input (line 31)
# ---------------------------------------------------------------------------


def test_invariant_open_image_pil_rgb_passthrough():
    """A PIL RGB image is returned as-is after ``convert('RGB')`` (idempotent)."""
    img = _make_pil_rgb()
    result = _open_image(img)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (8, 8)


def test_invariant_open_image_pil_rgba_converts_to_rgb():
    """A PIL RGBA image is converted to RGB (line 31: ``.convert('RGB')``)."""
    img = _make_pil_rgba()
    result = _open_image(img)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# _open_image — ndarray with float values (line 35)
# ---------------------------------------------------------------------------


def test_invariant_open_image_float_ndarray_unit_range():
    """A float ndarray with values in [0, 1] is scaled to uint8 (line 35, first branch).

    Values are multiplied by 255 and clipped before creating the PIL image.
    """
    arr = np.array([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.0]]], dtype=np.float32)  # (1,2,3)
    result = _open_image(arr)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    # The image should be 2-wide, 1-tall.
    assert result.size == (2, 1)
    raw = np.array(result)
    # Pixel (0,0): [0, 0.5*255, 255] → [0, ~127-128, 255]
    assert raw[0, 0, 0] == 0
    assert 125 <= raw[0, 0, 1] <= 129  # 0.5 * 255 = 127.5 → clip rounds
    assert raw[0, 0, 2] == 255


def test_invariant_open_image_float_ndarray_above_one_cast():
    """A float ndarray with max > 1.0 is cast to uint8 directly (line 35, else branch)."""
    arr = np.array([[[100.0, 200.0, 50.0]]], dtype=np.float32)  # max > 1
    result = _open_image(arr)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    raw = np.array(result)
    assert raw[0, 0, 0] == 100
    assert raw[0, 0, 1] == 200
    assert raw[0, 0, 2] == 50


def test_invariant_open_image_uint8_ndarray_no_scale():
    """A uint8 ndarray goes through the ndarray branch without rescaling."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 0] = 128
    result = _open_image(arr)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    raw = np.array(result)
    assert raw[0, 0, 0] == 128
    assert raw[0, 0, 1] == 0


# ---------------------------------------------------------------------------
# _open_image — str / Path (lines 37-38)
# ---------------------------------------------------------------------------


def test_invariant_open_image_path_object(tmp_path: Path):
    """A ``pathlib.Path`` to a PNG file is opened and converted to RGB (line 37-38)."""
    img = _make_pil_rgb(w=10, h=10)
    p = tmp_path / "test.png"
    img.save(str(p))
    result = _open_image(p)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (10, 10)


def test_invariant_open_image_str_path(tmp_path: Path):
    """A plain ``str`` path is opened and converted to RGB (line 37-38)."""
    img = _make_pil_rgb(w=6, h=6)
    p = tmp_path / "test.png"
    img.save(str(p))
    result = _open_image(str(p))
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (6, 6)


# ---------------------------------------------------------------------------
# _open_image — unsupported type → TypeError (line 39)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_input", [42, 3.14, {"a": 1}, [1, 2, 3]])
def test_invariant_open_image_unsupported_type_raises(bad_input: object):
    """Unsupported types raise a TypeError naming the source type (line 39)."""
    with pytest.raises(TypeError, match="unsupported image source type"):
        _open_image(bad_input)


# ---------------------------------------------------------------------------
# HFImageProcessor.__init__ (lines 114-116)
# ---------------------------------------------------------------------------


def test_invariant_hf_image_init_stores_fields_defers_processor():
    """Constructor records model_name_or_path, copies fp_kwargs, leaves _processor=None."""
    kw = {"cache_dir": "/tmp/cache"}
    proc = HFImageProcessor(
        model_name_or_path="openai/clip-vit-base-patch32",
        from_pretrained_kwargs=kw,
    )
    assert proc.model_name_or_path == "openai/clip-vit-base-patch32"
    assert proc._fp_kwargs == kw
    assert proc._fp_kwargs is not kw  # copied, not aliased
    assert proc._processor is None
    assert proc.modality == "image"


def test_invariant_hf_image_init_none_kwargs_yields_empty_dict():
    """Passing ``from_pretrained_kwargs=None`` defaults to an empty dict (line 115)."""
    proc = HFImageProcessor(model_name_or_path="model")
    assert proc._fp_kwargs == {}
    assert proc._processor is None


# ---------------------------------------------------------------------------
# HFImageProcessor._ensure_processor (lines 119-120, 122, 125)
# ---------------------------------------------------------------------------


def test_invariant_hf_ensure_processor_lazy_loads_and_caches(monkeypatch):
    """_ensure_processor calls from_pretrained once and caches the result (lines 119-125)."""
    out = {"pixel_values": np.ones((1, 3, 16, 16), dtype=np.float32)}
    mod, calls = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="clip-model", from_pretrained_kwargs={"a": 1})
    assert proc._processor is None

    first = proc._ensure_processor()
    second = proc._ensure_processor()

    assert first is second  # cached singleton
    assert len(calls) == 1  # from_pretrained called only once
    assert calls[0] == ("clip-model", {"a": 1})


def test_invariant_hf_ensure_processor_forwards_kwargs(monkeypatch):
    """The stored _fp_kwargs are forwarded to AutoImageProcessor.from_pretrained."""
    out = {"pixel_values": np.ones((1, 3, 8, 8), dtype=np.float32)}
    mod, calls = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    kw = {"revision": "main", "local_files_only": True}
    proc = HFImageProcessor(model_name_or_path="mymodel", from_pretrained_kwargs=kw)
    proc._ensure_processor()
    assert calls[0][1] == kw


# ---------------------------------------------------------------------------
# HFImageProcessor.__call__ — squeeze when batch dim == 1 (lines 134-135)
# ---------------------------------------------------------------------------


def test_invariant_hf_call_squeezes_single_image(monkeypatch):
    """When pixel_values has shape (1, C, H, W), the batch dim is squeezed (line 134-135)."""
    pv = np.ones((1, 3, 16, 16), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    img = _make_pil_rgb()
    result = proc(img)

    assert result["modality"] == "image"
    assert result["pixel_values"].shape == (3, 16, 16)  # squeezed
    assert result["pixel_values"].dtype == np.float32


def test_invariant_hf_call_no_squeeze_when_batch_gt_one(monkeypatch):
    """When pixel_values has shape (N, C, H, W) with N>1, shape is preserved (line 133)."""
    pv = np.ones((2, 3, 16, 16), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    imgs = [_make_pil_rgb(), _make_pil_rgb()]
    result = proc(imgs)

    assert result["modality"] == "image"
    assert result["pixel_values"].shape == (2, 3, 16, 16)  # not squeezed


def test_invariant_hf_call_wraps_non_list_image_in_list(monkeypatch):
    """A single non-list image is wrapped in a list before passing to the processor (line 130)."""
    pv = np.ones((1, 3, 8, 8), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    img = _make_pil_rgb(w=8, h=8)
    proc(img)
    # Result should have the processor called with a list of 1 PIL image.
    hf_proc: _FakeHFProcessor = proc._processor
    assert len(hf_proc.calls) == 1
    passed_images = hf_proc.calls[0]["images"]
    assert isinstance(passed_images, list)
    assert len(passed_images) == 1


def test_invariant_hf_call_list_input_not_double_wrapped(monkeypatch):
    """A list input is NOT re-wrapped — it's passed directly (line 129 branch is False)."""
    pv = np.ones((2, 3, 4, 4), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    imgs = [_make_pil_rgb(), _make_pil_rgb()]
    proc(imgs)
    hf_proc: _FakeHFProcessor = proc._processor
    passed_images = hf_proc.calls[0]["images"]
    assert isinstance(passed_images, list)
    assert len(passed_images) == 2


def test_invariant_hf_call_passes_return_tensors_np(monkeypatch):
    """The __call__ method always passes return_tensors='np' to the HF processor (line 132)."""
    pv = np.ones((1, 3, 4, 4), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    proc(_make_pil_rgb())
    hf_proc: _FakeHFProcessor = proc._processor
    assert hf_proc.calls[0]["return_tensors"] == "np"


def test_invariant_hf_call_opens_ndarray_images(monkeypatch):
    """ndarray inputs are opened via _open_image before being passed to HF processor."""
    pv = np.ones((1, 3, 4, 4), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    result = proc(arr)
    assert result["modality"] == "image"
    # The _open_image should have produced a PIL image passed to the HF processor.
    hf_proc: _FakeHFProcessor = proc._processor
    pil_images = hf_proc.calls[0]["images"]
    assert all(isinstance(im, Image.Image) for im in pil_images)


def test_invariant_hf_call_pixel_values_cast_to_float32(monkeypatch):
    """pixel_values from the HF processor are cast to float32 regardless of source dtype."""
    pv = np.ones((1, 3, 4, 4), dtype=np.float64)  # float64 from HF
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    result = proc(_make_pil_rgb())
    assert result["pixel_values"].dtype == np.float32


def test_invariant_hf_call_tuple_input_wrapped(monkeypatch):
    """A tuple input is also wrapped in a list (line 129-130 checks list/tuple)."""
    pv = np.ones((2, 3, 4, 4), dtype=np.float32)
    out = {"pixel_values": pv}
    mod, _ = _fake_transformers_module(out)
    monkeypatch.setitem(sys.modules, "transformers", mod)

    proc = HFImageProcessor(model_name_or_path="m")
    imgs = (_make_pil_rgb(), _make_pil_rgb())
    proc(imgs)
    hf_proc: _FakeHFProcessor = proc._processor
    passed_images = hf_proc.calls[0]["images"]
    # Tuple input: isinstance(images, (list, tuple)) is True → not wrapped
    assert len(passed_images) == 2


# ---------------------------------------------------------------------------
# SimpleImageProcessor — additional edge cases via _open_image
# ---------------------------------------------------------------------------


def test_invariant_simple_processor_pil_image_input():
    """SimpleImageProcessor accepts a PIL Image directly (exercises _open_image line 31)."""
    proc = SimpleImageProcessor(size=16)
    img = _make_pil_rgb(w=32, h=32)
    out = proc(img)
    assert out["modality"] == "image"
    assert out["pixel_values"].shape == (3, 16, 16)
    assert out["pixel_values"].dtype == np.float32


def test_invariant_simple_processor_path_input(tmp_path: Path):
    """SimpleImageProcessor accepts a Path source (exercises _open_image lines 37-38)."""
    img = _make_pil_rgb(w=20, h=20)
    p = tmp_path / "img.png"
    img.save(str(p))
    proc = SimpleImageProcessor(size=8)
    out = proc(p)
    assert out["modality"] == "image"
    assert out["pixel_values"].shape == (3, 8, 8)


def test_invariant_simple_processor_str_path_input(tmp_path: Path):
    """SimpleImageProcessor accepts a str path (exercises _open_image lines 37-38)."""
    img = _make_pil_rgb(w=12, h=12)
    p = tmp_path / "img.png"
    img.save(str(p))
    proc = SimpleImageProcessor(size=4)
    out = proc(str(p))
    assert out["modality"] == "image"
    assert out["pixel_values"].shape == (3, 4, 4)


def test_invariant_simple_processor_float_ndarray_unit_range_input():
    """SimpleImageProcessor accepts a float ndarray with values in [0,1] (line 35)."""
    np.random.seed(42)
    arr = np.random.rand(8, 8, 3).astype(np.float32)  # values in [0,1]
    proc = SimpleImageProcessor(size=4)
    out = proc(arr)
    assert out["modality"] == "image"
    assert out["pixel_values"].shape == (3, 4, 4)


def test_invariant_simple_processor_float_ndarray_above_one_input():
    """SimpleImageProcessor accepts float ndarray with max>1.0 (line 35 else branch)."""
    arr = np.full((8, 8, 3), 128.0, dtype=np.float32)  # max=128 > 1
    proc = SimpleImageProcessor(size=4)
    out = proc(arr)
    assert out["modality"] == "image"
    assert out["pixel_values"].shape == (3, 4, 4)


def test_invariant_simple_processor_batch_with_pil_images():
    """A list of PIL images is stacked into shape (N, C, H, W)."""
    proc = SimpleImageProcessor(size=8)
    imgs = [_make_pil_rgb(w=32, h=32) for _ in range(4)]
    out = proc(imgs)
    assert out["pixel_values"].shape == (4, 3, 8, 8)
    assert out["modality"] == "image"


def test_invariant_simple_processor_rejects_bad_type_in_batch():
    """A bad type inside a list raises TypeError from _open_image (line 39)."""
    proc = SimpleImageProcessor(size=8)
    with pytest.raises(TypeError, match="unsupported image source type"):
        proc([42])
