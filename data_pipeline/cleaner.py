"""data_pipeline/cleaner.py

Utility to clean raw text while preserving markdown structure.
"""

import re
from typing import List


def standardize_newlines(text: str) -> str:
    """Convert all line endings to ``\n``.

    Handles Windows (\r\n), old Mac (\r) and Unix (\n).
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_trailing_whitespace(lines: List[str]) -> List[str]:
    """Remove trailing spaces/tabs from each line.
    """
    return [line.rstrip() for line in lines]


def clean_text(raw: str) -> str:
    """Full cleaning pipeline.

    * Normalise newline characters.
    * Remove trailing whitespace.
    * Preserve markdown headers, links, and fenced code blocks.
    """
    # Normalise line endings first
    normalized = standardize_newlines(raw)
    # Split into lines for fine‑grained whitespace stripping
    lines = normalized.split("\n")
    cleaned_lines = strip_trailing_whitespace(lines)
    return "\n".join(cleaned_lines)

# Export for external use
__all__ = ["clean_text"]

