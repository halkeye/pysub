"""pysub - Generate subtitles from video files.

This package provides tools for transcribing and translating video content
into subtitle files (SRT or VTT format).
"""

__version__ = "0.1.0"

from pysub.config import (
    Config,
    SubtitleType,
    TranscriptionProvider,
    TranslationProvider,
)
from pysub.processor import process_single_video

__all__ = [
    "__version__",
    "Config",
    "SubtitleType",
    "TranscriptionProvider",
    "TranslationProvider",
    "process_single_video",
]
