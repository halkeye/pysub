"""Ollama translation provider."""

import logging
from string import Template

import requests

from pysub.language import get_language_code

logger = logging.getLogger(__name__)

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
) -> str:
    """Translate text using Ollama's local LLM API.

    Args:
        text: The text to translate.
        source_language: Source language name.
        target_language: Target language name.
        model: The Ollama model to use for translation.
        server: Ollama server URL.
        prompt: Custom prompt template (optional).

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

    body = {"model": model, "prompt": resolved_prompt, "stream": False}

    response = requests.post(
        f"{server}/api/generate",
        json=body,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()["response"].strip().strip('"')
    logger.debug("Ollama translation result: %s", result[:50] if result else "")
    return result
