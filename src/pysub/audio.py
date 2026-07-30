"""Audio extraction and processing utilities."""

import logging
from dataclasses import dataclass
from typing import Any, List

import torch
from moviepy import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

from pysub.config import Config, TranscriptionProvider

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, audio_path: str = "temp_audio.mp3") -> str:
    """Extract audio from a video file.

    Args:
        video_path: Path to the input video file.
        audio_path: Path where the extracted audio will be saved.

    Returns:
        Path to the extracted audio file.
    """
    logger.debug("Extracting audio from %s to %s", video_path, audio_path)
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        logger.error("No audio stream found in the video file: %s", video_path)
        raise ValueError(f"No audio stream found in the video file: {video_path}")

    clip.audio.write_audiofile(audio_path)
    clip.close()
    logger.debug("Audio extraction complete")
    return audio_path


@dataclass(frozen=True)
class ChunkedAudio:
    """Audio Chunk"""

    start_time: float
    end_time: float
    filename: str


def chunk_audio(
    config: Config,
    audio_path: str,
    max_chunk_duration_ms: int = 1 * 60 * 1000,
    silence_gap_threshold_ms: int = 15 * 1000,
) -> List[ChunkedAudio]:
    """Chunk audio into segments based on speech timestamps.

    Chunks are created when either:
    - There's more than `silence_gap_threshold_ms` of silence between speech segments
    - The chunk would exceed `max_chunk_duration_ms` in length

    Args:
        config: Application configuration.
        audio_path: Path to the audio file to chunk.
        max_chunk_duration_ms: Maximum chunk duration in milliseconds (default: 5 minutes).
        silence_gap_threshold_ms: Silence gap threshold to split chunks (default: 30 seconds).

    Returns:
        List of ChunkedAudio objects with start times and file paths.
    """
    if config.transcription == TranscriptionProvider.WHISPER:
        audio = AudioSegment.from_file(audio_path)
        return [ChunkedAudio(0.0, len(audio) / 1000, audio_path)]

    logger.info("loading voice detection model")
    model, utils = torch.hub.load(  # type: ignore[misc]
        "snakers4/silero-vad", "silero_vad"
    )
    get_speech_timestamps, _, read_audio, _, _ = utils  # type: ignore[misc]
    logger.info("getting speech timestamps")
    speech_timestamps: list[dict[str, Any]] = get_speech_timestamps(
        read_audio(audio_path),
        model,
        return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    )

    if not speech_timestamps:
        logger.warning("No speech detected in audio file: %s", audio_path)
        return []

    logger.info("converting audio file")
    audio = AudioSegment.from_file(audio_path)

    logger.info("creating chunks")
    chunks = []

    def save_chunk(start: int, end: int) -> None:
        """Save a chunk of audio to a file and add to chunks list."""
        audio_chunk = audio[start:end]
        chunk_filename = (
            f"{config.tmp_dir}/chunk_{start / 1000:.2f}_{end / 1000:.2f}.mp3"
        )
        audio_chunk.export(chunk_filename, format="mp3")
        chunks.append(ChunkedAudio(start / 1000, end / 1000, chunk_filename))
        logger.debug(
            "Created chunk: %s (duration: %.2fs)", chunk_filename, (end - start) / 1000
        )

    chunk_start_ms: int | None = None
    last_end_ms: int | None = None

    for speech_timestamp in tqdm(
        speech_timestamps, desc="Speech Timestamps", position=1
    ):
        current_start_ms = int(speech_timestamp["start"] * 1000)
        current_end_ms = int(speech_timestamp["end"] * 1000)

        # First speech segment - initialize chunk
        if chunk_start_ms is None:
            chunk_start_ms = current_start_ms
            last_end_ms = current_end_ms
            continue

        # Calculate current chunk duration if we include this segment
        potential_chunk_duration = current_end_ms - chunk_start_ms
        silence_gap = current_start_ms - last_end_ms if last_end_ms else 0

        # Check if we should start a new chunk:
        # 1. Silence gap exceeds threshold
        # 2. Adding this segment would exceed max chunk duration
        should_split = (
            silence_gap > silence_gap_threshold_ms
            or potential_chunk_duration > max_chunk_duration_ms
        )

        if should_split and last_end_ms is not None:
            # Save the current chunk up to the last speech segment
            save_chunk(chunk_start_ms, last_end_ms)
            # Start a new chunk from the current speech segment
            chunk_start_ms = current_start_ms

        last_end_ms = current_end_ms

    # Save the final chunk
    if chunk_start_ms is not None and last_end_ms is not None:
        save_chunk(chunk_start_ms, last_end_ms)

    logger.info("Created %d chunks from audio file", len(chunks))
    return chunks
