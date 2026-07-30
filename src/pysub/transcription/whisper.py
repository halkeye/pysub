"""Whisper transcription provider using faster-whisper."""

import logging
from dataclasses import dataclass
from typing import Iterator

import torch
from faster_whisper import WhisperModel

from pysub.config import Config, TranslationProvider
from pysub.language import get_language_code
from pysub.transcription import TranscriptionInfo, TranscriptionSegment

logger = logging.getLogger(__name__)


def transcribe_whisper(
    audio_path: str,
    config: Config,
    source_language: str | None = None,
) -> tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
    """Transcribe audio using Whisper model.

    Args:
        audio_path: Path to the audio file.
        config: Configuration containing whisper model settings.
        source_language: Optional source language code.

    Returns:
        Tuple of (segment iterator, transcription info).
    """
    logger.info("Loading Whisper model: %s", config.whisper_model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.debug("Using device: %s", device)

    whisper_model = WhisperModel(
        config.whisper_model,
        device=device,
        compute_type="float16",
    )

    task = (
        "translate"
        if config.translation == TranslationProvider.WHISPER
        else "transcribe"
    )

    logger.info("Transcribing audio with task: %s", task)

    segments, info = whisper_model.transcribe(
        audio_path,
        task=task,
        language=(get_language_code(source_language) if source_language else None),
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Wrap the segments iterator to convert to our dataclass
    def segment_wrapper():
        for seg in segments:
            yield TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )

    transcription_info = TranscriptionInfo(
        language=info.language,
        duration=info.duration,
    )

    return segment_wrapper(), transcription_info
