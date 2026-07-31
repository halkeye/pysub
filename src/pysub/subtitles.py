"""Subtitle formatting and file generation utilities."""

import logging
import os
import re
from datetime import timedelta
from string import Template

from pysub.config import SubtitleType
from pysub.language import get_language_code

logger = logging.getLogger(__name__)

# Sentence-ending punctuation across supported target languages: Latin (.!?),
# fullwidth CJK (。！？), Devanagari danda/double danda (।॥), and the Arabic
# question mark (؟). Arabic reuses the Latin "." and "!".
_SENTENCE_END_CHARS = ".!?。！？।॥؟"

# Closing quotes/brackets that can trail a sentence-ending mark (e.g. the
# period in 'She said "Onii-chan."'). The real boundary is after these
# closers, not before them.
_CLOSER_CHARS = "\"')\\]}”’›»"

# Split right after a sentence-ending mark (and any closing quotes/brackets
# that follow it), consuming trailing whitespace (CJK has none). The first
# negative lookahead refuses to split before a closer, so the boundary lands
# after it instead of inside the quoted/parenthetical text. The second
# negative lookahead keeps clustered marks like "..." or "?!" together by
# refusing to split when another sentence-ending mark follows.
_SENTENCE_SPLIT_RE = re.compile(
    rf"(?:(?<=[{_SENTENCE_END_CHARS}])"
    rf"|(?<=[{_SENTENCE_END_CHARS}][{_CLOSER_CHARS}])"
    rf"|(?<=[{_SENTENCE_END_CHARS}][{_CLOSER_CHARS}][{_CLOSER_CHARS}]))"
    rf"(?![{_CLOSER_CHARS}])"
    rf"(?!\s*[{_SENTENCE_END_CHARS}])"
    r"\s*"
)


def format_vtt_timestamp(delta: timedelta) -> str:
    """Format a timedelta as HH:MM:SS.mmm for VTT format.

    webvtt.models.Timestamp.PATTERN requires a literal '.' followed by
    digits, but str(timedelta) omits the fraction entirely when there are
    no microseconds (e.g. "0:00:24" instead of "0:00:24.000000").

    Args:
        delta: A timedelta representing the timestamp.

    Returns:
        Formatted timestamp string in HH:MM:SS.mmm format.
    """
    total_ms = round(delta.total_seconds() * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    seconds, milliseconds = divmod(remainder_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def pad_subtitle_durations(
    entries: list[tuple[timedelta, timedelta, str]],
    min_duration: timedelta,
) -> list[tuple[timedelta, timedelta, str]]:
    """Extend subtitle end times so each one stays on screen for a minimum duration.

    Padding never pushes an end time past the next subtitle's start, so
    subtitles never end up overlapping as a side effect of padding.

    Args:
        entries: List of (start, end, content) tuples, in display order.
        min_duration: Minimum amount of time a subtitle should remain visible.

    Returns:
        A new list of (start, end, content) tuples with end times padded.
    """
    padded = list(entries)
    for i, (start, end, content) in enumerate(padded):
        if end - start >= min_duration:
            continue
        desired_end = start + min_duration
        if i + 1 < len(padded):
            desired_end = min(desired_end, padded[i + 1][0])
        if desired_end > end:
            padded[i] = (start, desired_end, content)
    return padded


def split_subtitle_by_punctuation(
    start: timedelta,
    end: timedelta,
    content: str,
) -> list[tuple[timedelta, timedelta, str]]:
    """Split subtitle content into sentences, distributing time proportionally.

    A transcription segment only carries one start/end pair for its whole
    text, so once it's split into multiple sentences there's no real timing
    signal left to place each one precisely. This allocates each sentence a
    slice of the original range sized by its share of the total character
    count, which approximates a constant speaking rate across the segment.

    Args:
        start: Original start time for the whole content.
        end: Original end time for the whole content.
        content: The subtitle text, potentially containing multiple sentences.

    Returns:
        List of (start, end, content) tuples, one per sentence. Returns a
        single-element list unchanged if no sentence boundary is found.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(content.strip()) if s]
    if len(sentences) <= 1 or end - start <= timedelta(0):
        return [(start, end, content)]

    total_chars = sum(len(sentence) for sentence in sentences)
    duration = end - start

    entries: list[tuple[timedelta, timedelta, str]] = []
    cursor = start
    for i, sentence in enumerate(sentences):
        if i == len(sentences) - 1:
            sentence_end = end
        else:
            sentence_end = cursor + duration * (len(sentence) / total_chars)
        entries.append((cursor, sentence_end, sentence))
        cursor = sentence_end
    return entries


def build_srt_filename(
    subtitle_template: str,
    subtitle_type: SubtitleType,
    video_path: str,
    target_language: str,
) -> str:
    """Build the subtitle filename from a template.

    Args:
        subtitle_template: Template string with placeholders.
        subtitle_type: The subtitle format (VTT or SRT).
        video_path: Path to the source video file.
        target_language: Target language name for the subtitles.

    Returns:
        Resolved subtitle filename.

    Template variables:
        $VIDEO_DIRECTORY: Directory containing the video file.
        $VIDEO_NAME: Video filename without extension.
        $VIDEO_EXTENSION: Video file extension.
        $LANGUAGE_NAME: Target language name.
        $LANGUAGE_CODE: ISO language code.
        $SUBTITLE_EXTENSION: Subtitle file extension (vtt or srt).
    """
    filename = Template(subtitle_template).safe_substitute(
        {
            "VIDEO_DIRECTORY": os.path.dirname(video_path) or ".",
            "VIDEO_NAME": os.path.splitext(os.path.basename(video_path))[0],
            "VIDEO_EXTENSION": os.path.splitext(os.path.basename(video_path))[1],
            "LANGUAGE_NAME": target_language,
            "LANGUAGE_CODE": get_language_code(target_language),
            "SUBTITLE_EXTENSION": subtitle_type.value,
        }
    )
    logger.debug("Built subtitle filename: %s", filename)
    return filename
