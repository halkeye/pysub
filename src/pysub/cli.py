"""Command-line interface for pysub."""

import logging
import tempfile

import configargparse
import platformdirs
from faster_whisper import available_models
from secret_type import secret
from tqdm.contrib.logging import logging_redirect_tqdm

from pysub.config import (
    Config,
    SubtitleType,
    TranscriptionProvider,
    TranslationProvider,
)
from pysub.processor import process_single_video

logger = logging.getLogger(__name__)


def parse_args() -> configargparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    p = configargparse.ArgParser(
        description="Generate subtitles from video(s).",
        config_file_parser_class=configargparse.TomlConfigParser(["pysub"]),
        default_config_files=[
            platformdirs.user_config_path("pysub") / "config.toml",
        ],
    )

    p.add_argument("--config", is_config_file=True, help="config file path")

    p.add_argument("input", help="Path to a video file or directory")
    p.add_argument(
        "--srt_filename",
        default="$VIDEO_DIRECTORY/$VIDEO_NAME.$LANGUAGE_CODE.$SUBTITLE_EXTENSION",
        help="SRT Filename (default will be video.lang.srt)",
    )
    p.add_argument(
        "--subtitle_type",
        help="Subtitle type",
        type=SubtitleType,
        default=SubtitleType.SRT,
        choices=[i.value for i in list(SubtitleType)],
    )
    p.add_argument(
        "--source_language",
        help="Source language for translation",
    )
    p.add_argument(
        "--target_language",
        help="Target language for translation",
        default="english",
    )
    p.add_argument(
        "--api_key",
        type=secret,
        help="API key for translation service",
    )
    p.add_argument(
        "--transcription",
        help="Transcription provider (openai or ollama)",
        type=TranscriptionProvider,
        choices=TranscriptionProvider,
        default=TranscriptionProvider.WHISPER,
    )
    p.add_argument(
        "--translation",
        help="Translation provider (Default: %(default)s)",
        type=TranslationProvider,
        choices=list(TranslationProvider),
        default=TranslationProvider.OLLAMA,
    )
    p.add_argument(
        "--model",
        help="Model for translation",
        default="translategemma:4b-it-q4_K_M",
    )
    p.add_argument(
        "--server",
        help="Server for Ollama translation",
        default="http://localhost:11434",
    )
    p.add_argument(
        "--whisper_model",
        choices=available_models(),
        help="Whisper model to use for transcription",
        default="large-v2",
    )
    p.add_argument(
        "--min_subtitle_duration",
        help="Minimum time (in seconds) a subtitle stays on screen; "
        "end times are padded to meet this without overlapping the next subtitle "
        "(default: %(default)s)",
        type=float,
        default=1.5,
    )
    p.add_argument(
        "-log",
        "--loglevel",
        help="Provide logging level. Example --loglevel debug",
        default="info",
    )

    return p.parse_args()


def setup_logging(level: str = "info") -> None:
    """Configure root logger and suppress noisy third-party loggers.

    Args:
        level: Logging level as a string (e.g., "debug", "info", "warning").
    """
    logging.basicConfig(
        level=level.upper(),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        handlers=[logging.FileHandler("pysub.log"), logging.StreamHandler()],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def main() -> None:
    """Main entry point for pysub CLI."""
    args = parse_args()

    setup_logging(args.loglevel)

    with tempfile.TemporaryDirectory() as temp_dir:
        config = Config.from_args(args, temp_dir)

        if args.input == "server":
            logger.info("Config: %s", config)
            raise ValueError("Server mode is not implemented")

        with logging_redirect_tqdm():
            logger.info("Config: %s", config)
            process_single_video(args.input, config)


if __name__ == "__main__":
    main()
