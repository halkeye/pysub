"""Transcription providers for pysub."""

import logging
from dataclasses import dataclass
from typing import Iterator

from pysub.config import Config, TranscriptionProvider
from pysub.transcription.qwen import transcribe_qwen
from pysub.transcription.types import TranscriptionInfo, TranscriptionSegment
from pysub.transcription.whisper import transcribe_whisper

logger = logging.getLogger(__name__)

__all__ = [
    "transcribe",
    "transcribe_whisper",
    "transcribe_qwen",
    "TranscriptionInfo",
    "TranscriptionSegment",
]


def transcribe(
    audio_path: str,
    config: Config,
    source_language: str | None = None,
    chunk_file: str | None = None,
) -> tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
    """Transcribe audio using the configured provider.

    Args:
        audio_path: Path to the audio file.
        config: Configuration specifying which provider to use.
        source_language: Optional source language.
        chunk_file: Path to chunk file (required for Qwen3).

    Returns:
        Tuple of (segment iterator, transcription info).

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    logger.debug("Transcribing with provider: %s", config.transcription.value)

    if config.transcription == TranscriptionProvider.WHISPER:
        return transcribe_whisper(audio_path, config, source_language)

    if config.transcription == TranscriptionProvider.QWEN3:
        if chunk_file is None:
            raise ValueError("chunk_file is required for Qwen3 transcription")
        transcribe_qwen(audio_path, chunk_file)
        # Qwen3 currently exits, so this won't be reached
        raise RuntimeError("Qwen3 transcription is work-in-progress")

    raise ValueError(f"Unsupported transcription provider: {config.transcription}")
