"""data_pipeline/cleaner.py

Utility to clean raw text while preserving markdown structure.
"""

import re
from html.parser import HTMLParser
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


class _MarkdownHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "article", "blockquote", "div", "footer", "header", "main", "nav",
        "ol", "p", "section", "table", "tbody", "td", "th", "thead", "tr", "ul",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.pre_depth = 0
        self.skipped: List[str] = []
        self.links: List[tuple[int, str]] = []

    def _append(self, text: str) -> None:
        if text:
            self.parts.append(text)

    def _line_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self._append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skipped:
            if tag in self._SKIP_TAGS:
                self.skipped.append(tag)
            return
        if tag in self._SKIP_TAGS:
            self.skipped.append(tag)
        elif tag in {f"h{level}" for level in range(1, 7)}:
            self._line_break()
            self._append("#" * int(tag[1]) + " ")
        elif tag == "pre":
            self._line_break()
            self._append("```\n")
            self.pre_depth += 1
        elif tag == "code" and not self.pre_depth:
            self._append("`")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            self.links.append((len(self.parts), href))
        elif tag == "br":
            self._line_break()
        elif tag == "li":
            self._line_break()
            self._append("- ")
        elif tag in self._BLOCK_TAGS:
            self._line_break()
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self._append(alt)

    def handle_endtag(self, tag: str) -> None:
        if self.skipped:
            if tag == self.skipped[-1]:
                self.skipped.pop()
            return
        if tag == "a" and self.links:
            start, href = self.links.pop()
            label = "".join(self.parts[start:]).strip()
            if href and label:
                self.parts[start:] = [f"[{label}]({href})"]
            elif href:
                self.parts[start:] = [href]
        elif tag == "pre" and self.pre_depth:
            self._line_break()
            self._append("```\n")
            self.pre_depth -= 1
        elif tag == "code" and not self.pre_depth:
            self._append("`")
        elif tag in self._BLOCK_TAGS or tag in {f"h{level}" for level in range(1, 7)}:
            self._line_break()

    def handle_data(self, data: str) -> None:
        if self.skipped:
            return
        self._append(data if self.pre_depth else re.sub(r"\s+", " ", data))

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_markdown(raw: str) -> str:
    """Extract readable text while retaining headings, links, and code blocks."""
    parser = _MarkdownHTMLParser()
    parser.feed(raw)
    parser.close()
    return parser.markdown()

# Export for external use
__all__ = ["clean_text", "html_to_markdown"]

