"""Qwen3 transcription provider."""

import logging
import os
from typing import Any, ClassVar, Iterator, Optional

import torch
from pydub import AudioSegment
from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)
from transformers.models.qwen3_asr.processing_qwen3_asr import (
    FORCED_ALIGNER_LANGUAGES,
    resolve_language,
)

from pysub.transcription.types import TranscriptionInfo, TranscriptionSegment

logger = logging.getLogger(__name__)


class QwenASRModel:
    """Singleton class for managing the Qwen3 ASR model and processor."""

    ASR_MODEL_ID: ClassVar[str] = "Qwen/Qwen3-ASR-1.7B-hf"

    _instance: ClassVar["QwenASRModel | None"] = None
    _model: ClassVar[AutoModelForMultimodalLM | None] = None
    _processor: ClassVar[AutoProcessor | None] = None

    def __new__(cls) -> "QwenASRModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> AutoModelForMultimodalLM:
        """Get the ASR model, loading it on first access."""
        if QwenASRModel._model is None:
            logger.debug("Loading Qwen3 ASR model: %s", self.ASR_MODEL_ID)
            QwenASRModel._model = AutoModelForMultimodalLM.from_pretrained(
                self.ASR_MODEL_ID, device_map="auto"
            )
            logger.debug(
                "ASR model loaded on %s with dtype %s",
                QwenASRModel._model.device,
                QwenASRModel._model.dtype,
            )
        return QwenASRModel._model

    @property
    def processor(self) -> AutoProcessor:
        """Get the ASR processor, loading it on first access."""
        if QwenASRModel._processor is None:
            logger.debug("Loading Qwen3 ASR processor: %s", self.ASR_MODEL_ID)
            QwenASRModel._processor = AutoProcessor.from_pretrained(self.ASR_MODEL_ID)
        return QwenASRModel._processor

    @classmethod
    def clear(cls) -> None:
        """Clear the cached model and processor to free memory."""
        if cls._model is not None or cls._processor is not None:
            logger.debug("Clearing Qwen3 ASR model from memory")
            cls._model = None
            cls._processor = None
            torch.cuda.empty_cache()
            logger.debug("Qwen3 ASR model cleared")


class QwenAlignerModel:
    """Singleton class for managing the Qwen3 Forced Aligner model and processor."""

    ALIGNER_MODEL_ID: ClassVar[str] = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

    _instance: ClassVar["QwenAlignerModel | None"] = None
    _model: ClassVar[AutoModelForTokenClassification | None] = None
    _processor: ClassVar[AutoProcessor | None] = None

    def __new__(cls) -> "QwenAlignerModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def model(self) -> AutoModelForTokenClassification:
        """Get the aligner model, loading it on first access."""
        if QwenAlignerModel._model is None:
            logger.debug("Loading Qwen3 Aligner model: %s", self.ALIGNER_MODEL_ID)
            QwenAlignerModel._model = AutoModelForTokenClassification.from_pretrained(
                self.ALIGNER_MODEL_ID, dtype=torch.bfloat16, device_map="auto"
            )
            logger.debug(
                "Aligner model loaded on %s with dtype %s",
                QwenAlignerModel._model.device,
                QwenAlignerModel._model.dtype,
            )
        return QwenAlignerModel._model

    @property
    def processor(self) -> AutoProcessor:
        """Get the aligner processor, loading it on first access."""
        if QwenAlignerModel._processor is None:
            logger.debug("Loading Qwen3 Aligner processor: %s", self.ALIGNER_MODEL_ID)
            QwenAlignerModel._processor = AutoProcessor.from_pretrained(
                self.ALIGNER_MODEL_ID
            )
        return QwenAlignerModel._processor

    @classmethod
    def clear(cls) -> None:
        """Clear the cached model and processor to free memory."""
        if cls._model is not None or cls._processor is not None:
            logger.debug("Clearing Qwen3 Aligner model from memory")
            cls._model = None
            cls._processor = None
            torch.cuda.empty_cache()
            logger.debug("Qwen3 Aligner model cleared")


def clear_qwen_model() -> None:
    """Clear all cached Qwen3 models and processors to free memory.

    Call this function when you're done with transcription to release
    GPU/CPU memory used by the models.
    """
    QwenASRModel.clear()
    QwenAlignerModel.clear()


def _words_to_sentences(
    word_timestamps: list[dict[str, Any]],
    pause_threshold_sec: float = 0.7,
    max_segment_duration_sec: float = 10.0,
) -> list[TranscriptionSegment]:
    """Convert word-level timestamps to segments based on pauses.

    Segments are split when:
    - The gap between words exceeds `pause_threshold_sec`
    - The segment duration would exceed `max_segment_duration_sec`

    Args:
        word_timestamps: List of word dicts with 'text', 'start_time', 'end_time'.
        pause_threshold_sec: Silence gap threshold to trigger a new segment.
        max_segment_duration_sec: Maximum segment duration before forcing a split.

    Returns:
        List of TranscriptionSegment objects.
    """
    if not word_timestamps:
        return []

    segments: list[TranscriptionSegment] = []
    current_words: list[dict[str, Any]] = []
    segment_start_time: float | None = None

    for word in word_timestamps:
        word_start = word["start_time"]
        word_end = word["end_time"]

        # First word - start a new segment
        if not current_words:
            current_words.append(word)
            segment_start_time = word_start
            continue

        # Calculate gap from previous word
        prev_word_end = current_words[-1]["end_time"]
        gap = word_start - prev_word_end

        # Calculate potential segment duration if we add this word
        potential_duration = word_end - (segment_start_time or word_start)

        # Check if we should start a new segment
        should_split = (
            gap > pause_threshold_sec or potential_duration > max_segment_duration_sec
        )

        if should_split:
            # Save current segment
            segment_text = " ".join(w["text"] for w in current_words)
            segments.append(
                TranscriptionSegment(
                    start=segment_start_time or current_words[0]["start_time"],
                    end=current_words[-1]["end_time"],
                    text=segment_text,
                )
            )
            # Start new segment with current word
            current_words = [word]
            segment_start_time = word_start
        else:
            # Add word to current segment
            current_words.append(word)

    # Save final segment
    if current_words:
        segment_text = " ".join(w["text"] for w in current_words)
        segments.append(
            TranscriptionSegment(
                start=segment_start_time or current_words[0]["start_time"],
                end=current_words[-1]["end_time"],
                text=segment_text,
            )
        )

    return segments


def transcribe_qwen(
    audio_path: str,
    source_language: Optional[str] = None,
) -> tuple[Iterator[TranscriptionSegment], TranscriptionInfo]:
    """Transcribe audio using Qwen3 ASR model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Tuple of (segment iterator, transcription info).
    """
    logger.debug("Starting Qwen3 transcription")

    asr = QwenASRModel()
    aligner = QwenAlignerModel()

    audio = AudioSegment.from_file(os.path.realpath(audio_path))
    audio_duration_sec = len(audio) / 1000.0

    # Step 1: Transcribe
    logger.debug("Running ASR transcription")
    inputs = asr.processor.apply_transcription_request(
        audio=os.path.realpath(audio_path)
    ).to(asr.model.device, asr.model.dtype)
    output_ids = asr.model.generate(**inputs, max_new_tokens=512)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    # Parsed output: dict with "language" and "transcription"
    parsed = asr.processor.decode(generated_ids, return_format="parsed")[0]

    language = resolve_language(source_language or parsed["language"])
    assert language is not None
    transcript = parsed["transcription"]

    logger.debug("Detected language: %s", language)
    logger.debug(
        "Transcript: %s",
        transcript[:100] + "..." if len(transcript) > 100 else transcript,
    )

    if language not in FORCED_ALIGNER_LANGUAGES:
        logger.warning(
            "Forced aligner does not support %s (supported: %s); "
            "using whole-chunk timing instead of word-level alignment",
            language,
            sorted(FORCED_ALIGNER_LANGUAGES),
        )
        segments = [
            TranscriptionSegment(start=0.0, end=audio_duration_sec, text=transcript)
        ]
    else:
        # Step 2: Prepare alignment inputs
        logger.debug("Running forced alignment")
        aligner_inputs, word_lists = aligner.processor.prepare_forced_aligner_inputs(
            audio=os.path.realpath(audio_path),
            transcript=transcript,
            language=language,
        )
        aligner_inputs = aligner_inputs.to(aligner.model.device, aligner.model.dtype)

        # Step 3: Run forced aligner
        with torch.inference_mode():
            outputs = aligner.model(**aligner_inputs)

        # Step 4: Decode timestamps
        word_timestamps: list[dict[str, Any]] = (
            aligner.processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=aligner_inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=aligner.model.config.timestamp_token_id,
            )[0]
        )

        logger.debug("Got %d word timestamps", len(word_timestamps))

        # Step 5: Convert words to segments based on pauses
        logger.debug("Segmenting based on pauses")
        segments = _words_to_sentences(word_timestamps)

        logger.debug("Created %d sentence segments", len(segments))

    info = TranscriptionInfo(language=language, duration=audio_duration_sec)

    return iter(segments), info
