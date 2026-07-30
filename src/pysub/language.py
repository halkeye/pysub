"""Language code and name utilities using pycountry."""

import logging

import pycountry

logger = logging.getLogger(__name__)


def get_language_code(language_name: str) -> str:
    """Get the ISO language code for a given language name.

    Args:
        language_name: Human-readable language name (e.g., "English", "Japanese").

    Returns:
        2-letter ISO code if available, otherwise 3-letter code.
        Returns "Language not found" if the language cannot be looked up.
    """
    try:
        lang = pycountry.languages.lookup(language_name)
        return lang.alpha_2 if hasattr(lang, "alpha_2") else lang.alpha_3
    except LookupError:
        logger.warning("Language code not found for: %s", language_name)
        return "Language not found"


def get_language_name(code: str) -> str:
    """Get the language name for a given ISO code.

    Args:
        code: ISO language code (2-letter or 3-letter).

    Returns:
        Lowercase language name, or "Language not found" if not found.
    """
    try:
        lang = (
            pycountry.languages.get(alpha_2=code)
            or pycountry.languages.get(alpha_3=code)
            or pycountry.languages.get(name=code)
        )
        if lang is None:
            raise LookupError(f"Language code '{code}' not found.")
        return lang.name.lower()
    except LookupError:
        logger.warning("Language name not found for code: %s", code)
        return "Language not found"
