"""Qwen3 transcription provider (Work in Progress)."""

import logging
import os
from typing import Iterator

import torch
from pydub import AudioSegment
from transformers import AutoModelForMultimodalLM, AutoProcessor

from pysub.transcription.types import TranscriptionInfo, TranscriptionSegment

logger = logging.getLogger(__name__)


def transcribe_qwen(
    audio_path: str,
) -> tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
    """Transcribe audio using Qwen3 ASR model.

    Args:
        audio_path: Path to the audio file.

    Raises:
        SystemExit: Currently exits after processing.
    """
    logger.debug("Starting Qwen3 transcription")

    torch.cuda.empty_cache()

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"

    logger.debug("Loading Qwen3 model: %s", model_id)

    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

    logger.debug(f"Model loaded on {model.device} with dtype {model.dtype}")

    audio = AudioSegment.from_file(os.path.realpath(audio_path))

    inputs = processor.apply_transcription_request(
        audio=os.path.realpath(audio_path),
    ).to(model.device, model.dtype)

    output_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]

    # Parsed output: dict with "language" and "transcription"
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    info = TranscriptionInfo(
        language=parsed["language"], duration=len(audio)
    )  # Duration is not provided by Qwen3
    segments = [
        TranscriptionSegment(start=0.0, end=len(audio), text=parsed["transcription"])
    ]  # Timing info is not provided by Qwen3

    return iter(segments), info
