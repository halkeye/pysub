"""OpenAI translation provider."""

import logging

from openai import OpenAI

from pysub.config import Config

logger = logging.getLogger(__name__)


def translate_with_openai(
    text: str,
    source_language: str,
    target_language: str,
    config: Config,
) -> str:
    """Translate text using OpenAI's API.

    Args:
        text: The text to translate.
        source_language: Source language name.
        target_language: Target language name.
        config: Configuration containing API key.

    Returns:
        Translated text.
    """
    logger.debug("Translating with OpenAI: %s -> %s", source_language, target_language)

    client = OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional translation engine. "
                    f"Translate the following {source_language} sentence into **{target_language}** only. Do not use any other language except the one provide. "
                    f"You must translate from {source_language} to **{target_language}**. Do not confuse one language with another. Check and verify your work after you are done"
                    f"Respond with ONLY the *{target_language} sentence, no extra commentary."
                ),
            },
            {"role": "user", "content": f"{text}"},
        ],
    )

    if (
        len(response.choices) > 0
        and response.choices[0].message is not None
        and response.choices[0].message.content is not None
    ):
        result = response.choices[0].message.content.strip()
        logger.debug("OpenAI translation result: %s", result[:50] if result else "")
        return result

    return ""
