"""Multimodal processors: text/image/audio/video.

Importing this module triggers registration of all processors via
``@register("processor", ...)``.
"""

from .audio import HFAudioProcessor, MelSpectrogramProcessor
from .image import HFImageProcessor, SimpleImageProcessor
from .text import ChatTemplateProcessor, HFTextProcessor
from .video import DecordVideoProcessor, FrameFolderProcessor

__all__ = [
    "ChatTemplateProcessor",
    "HFTextProcessor",
    "SimpleImageProcessor",
    "HFImageProcessor",
    "MelSpectrogramProcessor",
    "HFAudioProcessor",
    "FrameFolderProcessor",
    "DecordVideoProcessor",
]
