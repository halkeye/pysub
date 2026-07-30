"""Video processing and subtitle generation orchestration."""

import logging
from datetime import timedelta
from typing import Optional

import srt
import webvtt
from tqdm import tqdm

from pysub.audio import chunk_audio, extract_audio
from pysub.config import (
    Config,
    SubtitleType,
    TranscriptionProvider,
    TranslationProvider,
)
from pysub.language import get_language_name
from pysub.subtitles import build_srt_filename, format_vtt_timestamp
from pysub.transcription import transcribe
from pysub.transcription.qwen import transcribe_qwen
from pysub.translation import translate_text

logger = logging.getLogger(__name__)


def process_single_video(
    video_path: str,
    config: Config,
) -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Process a single video file and generate subtitles.

    Args:
        video_path: Path to the video file.
        config: Configuration for transcription and translation.
    """
    # Extract config values for local use (some may be modified during processing)
    source_language = config.source_language
    target_language = config.target_language
    subtitle_type = config.subtitle_type

    subtitles: list[srt.Subtitle] = []

    vtt: Optional[webvtt.WebVTT] = None
    if subtitle_type == SubtitleType.VTT:
        vtt = webvtt.WebVTT()

    logger.info("Starting to read audio file")

    audio_path = "temp_audio.mp3"
    # FIXME
    # audio_path = extract_audio(video_path)
    chunks = chunk_audio(config, audio_path)
    if len(chunks) == 0:
        logger.warning(
            "No audio chunks were created. Check the audio file and VAD settings."
        )
        return
    logger.info("Audio file read and chunked into %d chunks", len(chunks))

    logger.info("Transcribing audio file")

    subtitle_path = None

    for chunk in tqdm(chunks, desc="Chunks", unit="chunk", position=1):
        if config.transcription == TranscriptionProvider.WHISPER:
            segments, info = transcribe(chunk.filename, config, source_language)

            if source_language is None:
                source_language = get_language_name(info.language)

        elif config.transcription == TranscriptionProvider.QWEN3:
            segments, info = transcribe_qwen(chunk.filename)
            if source_language is None:
                source_language = get_language_name(info.language)
        else:
            raise ValueError(
                f"Unsupported transcription provider: {config.transcription}"
            )

        if subtitle_path is None:
            subtitle_path = build_srt_filename(
                config.srt_filename_template,
                config.subtitle_type,
                video_path,
                target_language,
            )

            logger.info("Starting subtitle generation: %s", subtitle_path)

        for segment in segments:
            content = english = segment.text.strip()

            if config.translation == TranslationProvider.WHISPER:
                content = segment.text.strip()
            elif source_language.lower() != target_language.lower():
                content = translate_text(
                    english, source_language, target_language, config
                )

            start = timedelta(seconds=chunk.start_time + segment.start)
            end = timedelta(seconds=chunk.start_time + segment.end)

            if subtitle_type == SubtitleType.SRT:
                subtitles.append(
                    srt.Subtitle(
                        index=1,
                        start=start,
                        end=end,
                        content=content.replace("\n", "\\n"),
                    )
                )
            elif subtitle_type == SubtitleType.VTT:
                if vtt is None:
                    raise ValueError(
                        "VTT object is not initialized for VTT subtitle type"
                    )

                vtt.captions.append(
                    webvtt.Caption(
                        start=format_vtt_timestamp(start),
                        end=format_vtt_timestamp(end),
                        text=content.split("\n"),
                    )
                )

    if subtitle_type == SubtitleType.SRT:
        with open(subtitle_path, "w", encoding="utf-8") as subtitle_file:
            from pprint import pprint

            pprint(subtitles)
            subtitle_file.write(srt.compose(subtitles))
    if subtitle_type == SubtitleType.VTT:
        vtt.save(subtitle_path)

    logger.info("Subtitles saved to: %s", subtitle_path)
