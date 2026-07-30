"""Translation providers for pysub."""

import logging

from pysub.config import Config, TranslationProvider
from pysub.translation.ollama import translate_with_ollama
from pysub.translation.openai import translate_with_openai

logger = logging.getLogger(__name__)

__all__ = [
    "translate_text",
    "translate_with_ollama",
    "translate_with_openai",
]


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    config: Config,
) -> str:
    """Translate text using the configured provider.

    Args:
        text: The text to translate.
        source_language: Source language name.
        target_language: Target language name.
        config: Configuration specifying which provider to use.

    Returns:
        Translated text.

    Raises:
        ValueError: If an unsupported provider is specified or required
            configuration is missing.
    """
    logger.debug(
        "Translating text using %s: %s -> %s",
        config.translation.value,
        source_language,
        target_language,
    )

    if config.translation == TranslationProvider.OPENAI:
        return translate_with_openai(text, source_language, target_language, config)

    if config.translation == TranslationProvider.OLLAMA:
        if config.model is None:
            raise ValueError(
                "Model must be specified for Ollama translation verification."
            )

        if config.server is None:
            raise ValueError(
                "Server must be specified for Ollama translation verification."
            )

        return translate_with_ollama(
            text,
            source_language,
            target_language,
            model=config.model,
            server=config.server,
        )

    raise ValueError(f"Unsupported translation provider: {config.translation}")
