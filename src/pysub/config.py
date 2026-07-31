"""Configuration classes and enums for pysub."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum


class TranscriptionProvider(Enum):
    """Which provider to use for transcription."""

    WHISPER = "whisper"
    QWEN3 = "qwen3"


class TranslationProvider(Enum):
    """Which provider to use for translation."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    WHISPER = "whisper"


class SubtitleType(Enum):
    """Which subtitle type."""

    VTT = "vtt"
    SRT = "srt"


@dataclass(frozen=True)
class Config:  # pylint: disable=too-many-instance-attributes
    """Configuration for subtitle generation.

    This dataclass holds all configuration options for transcription and translation.
    Use Config.from_args() to create an instance from parsed command-line arguments.
    """

    target_language: str
    source_language: str | None
    subtitle_type: SubtitleType
    srt_filename_template: str
    api_key: str | None
    translation: TranslationProvider
    transcription: TranscriptionProvider
    model: str
    server: str
    whisper_model: str
    tmp_dir: str | None = None
    min_subtitle_duration_seconds: float = 1.5

    @classmethod
    def from_args(cls, args: argparse.Namespace, tmp_dir: str | None = None) -> Config:
        """Create a Config instance from parsed argparse/configargparse arguments."""
        return cls(
            target_language=args.target_language,
            source_language=args.source_language,
            subtitle_type=args.subtitle_type,
            srt_filename_template=args.srt_filename,
            api_key=args.api_key,
            translation=args.translation,
            transcription=args.transcription,
            model=args.model,
            server=args.server,
            whisper_model=args.whisper_model,
            tmp_dir=tmp_dir,
            min_subtitle_duration_seconds=args.min_subtitle_duration,
        )
