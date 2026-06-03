"""Video processors.

Frame-folder approach is hermetic: a "video" is a directory with
``frame_000.png``, ``frame_001.png``, ... files. Optional ``decord`` /
``av`` paths are tried lazily for real video files.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lighttrain.registry import register

from .image import SimpleImageProcessor


def _list_frame_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    return files


def _uniform_sample(items: Sequence[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]


@register("processor", "frame_folder")
class FrameFolderProcessor:
    """Read a directory of pre-extracted frames; sample uniformly to N frames.

    Output::

        {
          "video_frames": np.ndarray (T, C, H, W) float32,
          "modality": "video",
        }
    """

    modality = "video"

    def __init__(
        self,
        *,
        num_frames: int = 8,
        size: tuple[int, int] | int = 224,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.num_frames = int(num_frames)
        self._image_proc = SimpleImageProcessor(size=size, mean=mean, std=std)

    def __call__(self, video: Any, **_: Any) -> dict[str, Any]:
        if isinstance(video, (str, Path)):
            folder = Path(video)
            if not folder.is_dir():
                raise ValueError(
                    f"FrameFolderProcessor expects a directory of frames; got {video!r}"
                )
            files = _list_frame_files(folder)
            if not files:
                raise RuntimeError(f"no frame files in {folder!r}")
            sampled = _uniform_sample(files, self.num_frames)
        elif isinstance(video, (list, tuple)):
            sampled = _uniform_sample(list(video), self.num_frames)
        else:
            raise TypeError(
                f"unsupported video source: {type(video).__name__}; "
                "expected directory path or list of frames"
            )
        frames = [self._image_proc._process_one(f) for f in sampled]
        # Pad with zeros if fewer than num_frames available.
        while len(frames) < self.num_frames:
            frames.append(np.zeros_like(frames[0]))
        arr = np.stack(frames, axis=0).astype(np.float32, copy=False)
        return {
            "video_frames": arr,
            "modality": "video",
        }


@register("processor", "decord_video")
class DecordVideoProcessor:
    """Decode video files via ``decord`` (or ``av`` fallback). Lazy import."""

    modality = "video"

    def __init__(
        self,
        *,
        num_frames: int = 8,
        size: tuple[int, int] | int = 224,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.num_frames = int(num_frames)
        self._image_proc = SimpleImageProcessor(size=size, mean=mean, std=std)

    def _decode(self, path: str) -> list[np.ndarray]:
        try:
            import decord  # type: ignore

            vr = decord.VideoReader(path)
            n = len(vr)
            idx = np.linspace(0, max(0, n - 1), self.num_frames).astype(int)
            arr = vr.get_batch(idx).asnumpy()  # (T, H, W, C)
            return [arr[i] for i in range(arr.shape[0])]
        except ImportError:
            pass
        try:
            import av  # type: ignore

            container = av.open(path)
            stream = container.streams.video[0]
            total = stream.frames or 0
            target_idx = (
                set(np.linspace(0, max(0, total - 1), self.num_frames).astype(int))
                if total
                else None
            )
            frames: list[np.ndarray] = []
            for i, frame in enumerate(container.decode(video=0)):
                if target_idx is None or i in target_idx:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) >= self.num_frames:
                        break
            container.close()
            return frames
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DecordVideoProcessor requires `decord` or `av` to decode video files"
            ) from exc

    def __call__(self, video: Any, **_: Any) -> dict[str, Any]:
        if not isinstance(video, (str, os.PathLike)):
            raise TypeError(
                f"DecordVideoProcessor expects a file path; got {type(video).__name__}"
            )
        frames = self._decode(str(video))
        if not frames:
            raise RuntimeError(f"no frames decoded from {video!r}")
        processed = [self._image_proc._process_one(f) for f in frames]
        while len(processed) < self.num_frames:
            processed.append(np.zeros_like(processed[0]))
        arr = np.stack(processed[: self.num_frames], axis=0).astype(np.float32, copy=False)
        return {"video_frames": arr, "modality": "video"}


__all__ = ["FrameFolderProcessor", "DecordVideoProcessor"]
