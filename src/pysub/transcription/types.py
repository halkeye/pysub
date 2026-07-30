"""Transcription providers for pysub."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionInfo:
    """Information about the transcription result."""

    language: str
    duration: float


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing information."""

    start: float
    end: float
    text: str
