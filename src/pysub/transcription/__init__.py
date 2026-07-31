"""Transcription providers for pysub."""

import logging
from dataclasses import dataclass
from typing import Iterator

from pysub.config import Config, TranscriptionProvider
from pysub.transcription.qwen import (
    QwenAlignerModel,
    QwenASRModel,
    clear_qwen_model,
    transcribe_qwen,
)
from pysub.transcription.types import TranscriptionInfo, TranscriptionSegment
from pysub.transcription.whisper import transcribe_whisper

logger = logging.getLogger(__name__)

__all__ = [
    "transcribe",
    "transcribe_whisper",
    "transcribe_qwen",
    "clear_qwen_model",
    "QwenASRModel",
    "QwenAlignerModel",
    "TranscriptionInfo",
    "TranscriptionSegment",
]


def transcribe(
    audio_path: str,
    config: Config,
    source_language: str | None = None,
) -> tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
    """Transcribe audio using the configured provider.

    Args:
        audio_path: Path to the audio file.
        config: Configuration specifying which provider to use.
        source_language: Optional source language.

    Returns:
        Tuple of (segment iterator, transcription info).

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    logger.debug("Transcribing with provider: %s", config.transcription.value)

    if config.transcription == TranscriptionProvider.WHISPER:
        return transcribe_whisper(audio_path, config, source_language)

    if config.transcription == TranscriptionProvider.QWEN3:
        transcribe_qwen(audio_path)
        # Qwen3 currently exits, so this won't be reached
        raise RuntimeError("Qwen3 transcription is work-in-progress")

    raise ValueError(f"Unsupported transcription provider: {config.transcription}")
