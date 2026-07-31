"""Ollama translation provider."""

import logging
from difflib import SequenceMatcher
from string import Template

import requests

from pysub.language import get_language_code

logger = logging.getLogger(__name__)

_ECHO_SIMILARITY_THRESHOLD = 0.9


def _looks_like_echo(original: str, translated: str) -> bool:
    """Check whether translated text is suspiciously close to the untranslated original.

    Small local models sometimes fail to translate and just echo the source
    text back instead. This catches that failure mode via string similarity;
    it won't catch a model that translates incorrectly rather than not at all.
    """
    ratio = SequenceMatcher(None, original.casefold(), translated.casefold()).ratio()
    return ratio >= _ECHO_SIMILARITY_THRESHOLD

DEFAULT_PROMPT = (
    "You are a professional $SOURCE_LANG ($SOURCE_CODE) to $TARGET_LANG ($TARGET_CODE) translator.\n"
    "Your goal is to accurately convey the meaning and nuances of the original $SOURCE_LANG text while adhering to $TARGET_LANG grammar, vocabulary, and cultural sensitivities.\n"
    "\n"
    "Strict Output Rules:\n"
    "1. The translation MUST be written exclusively in **$TARGET_LANG script**.\n"
    "2. You MUST NOT include:\n"
    "   - Romanized or transliterated text\n"
    "   - $SOURCE_LANG words or phrases\n"
    "   - Commentary, notes, or metadata\n"
    "   - The original $SOURCE_LANG sentence\n"
    "3. The output MUST be clear, fluent, and natural to a native speaker of $TARGET_LANG.\n"
    "4. Your highest priority is accuracy and clarity for native speakers.\n"
    "5. After translating, **you MUST internally verify** that the output contains only $TARGET_LANG script, punctuation, or peoples names and contains NO $SOURCE_LANG, romanization, or foreign words.\n"
    "6. If verification fails, silently retry the translation until the output is clean and correct.\n\n"
    "\n"
    "Produce only the $TARGET_LANG translation, without any additional plesantries, responses, explanations or commentary. Please translate the following $SOURCE_LANG text into $TARGET_LANG:\n"
    "\n"
    "\n"
    "$TEXT"
)


def translate_with_ollama(
    text: str,
    source_language: str,
    target_language: str,
    model: str | None = None,
    server: str = "http://localhost:11434",
    prompt: str | None = None,
    max_attempts: int = 2,
) -> str:
    """Translate text using Ollama's local LLM API.

    Args:
        text: The text to translate.
        source_language: Source language name.
        target_language: Target language name.
        model: The Ollama model to use for translation.
        server: Ollama server URL.
        prompt: Custom prompt template (optional).
        max_attempts: Retries if the model echoes the source text back untranslated.

    Returns:
        Translated text.

    Raises:
        ValueError: If model is not specified.
        requests.HTTPError: If the API request fails.
    """
    if model is None:
        raise ValueError("Model must be specified for Ollama translation.")

    logger.debug(
        "Translating with Ollama (%s): %s -> %s",
        model,
        source_language,
        target_language,
    )

    if prompt is None:
        prompt = DEFAULT_PROMPT

    resolved_prompt = Template(prompt).safe_substitute(
        {
            "SOURCE_LANG": source_language,
            "SOURCE_CODE": get_language_code(source_language),
            "TARGET_LANG": target_language,
            "TARGET_CODE": get_language_code(target_language),
            "TEXT": text,
        }
    )

    result = text
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = resolved_prompt
        if attempt > 1:
            attempt_prompt = (
                f"Your previous response was not translated — it matched the "
                f"original {source_language} text almost exactly. You MUST output "
                f"{target_language} this time.\n\n"
            ) + resolved_prompt

        response = requests.post(
            f"{server}/api/generate",
            json={"model": model, "prompt": attempt_prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()

        result = response.json()["response"].strip().strip('"')

        if not _looks_like_echo(text, result):
            break

        logger.warning(
            "Ollama translation attempt %d/%d looks untranslated (echoed %s text); retrying",
            attempt,
            max_attempts,
            source_language,
        )
    else:
        logger.warning(
            "Ollama translation still looks untranslated after %d attempts", max_attempts
        )

    logger.debug("Ollama translation result: %s", result[:50] if result else "")
    return result
