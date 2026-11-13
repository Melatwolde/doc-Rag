"""Dependency-free HTML-to-Markdown conversion for documentation pages."""

from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin


class _DocumentationParser(HTMLParser):
    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self.in_pre = False
        self.link_url: str | None = None
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._newline(2)
            self.parts.append("#" * int(tag[1]) + " ")
        elif tag == "pre":
            self._newline(2)
            self.parts.append("```\n")
            self.in_pre = True
        elif tag == "a" and attributes.get("href"):
            self.link_url = urljoin(self.base_url or "", attributes["href"] or "")
            self.parts.append("[")
        elif tag in {"p", "div", "section", "article", "li", "br", "tr"}:
            self._newline(1)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._newline(2)
        elif tag == "pre":
            self._newline(1)
            self.parts.append("```\n")
            self.in_pre = False
        elif tag == "a" and self.link_url is not None:
            self.parts.append(f"]({self.link_url})")
            self.link_url = None
        elif tag in {"p", "div", "section", "article", "li", "tr"}:
            self._newline(1)

    def handle_data(self, data: str) -> None:
        if self.skip_depth or not data:
            return
        if self.in_pre:
            self.parts.append(data)
        elif self.link_url is not None:
            # Do not insert whitespace inside Markdown link labels.
            self.parts.append(" ".join(data.split()))
        else:
            self.parts.append(" ".join(data.split()) + " ")

    def _newline(self, count: int) -> None:
        existing = "".join(self.parts)
        trailing = len(existing) - len(existing.rstrip("\n"))
        if trailing < count:
            self.parts.append("\n" * (count - trailing))


def html_to_markdown(html: str, base_url: str | None = None) -> str:
    """Preserve page headings, links, and preformatted code as Markdown."""
    parser = _DocumentationParser(base_url)
    parser.feed(unescape(html))
    parser.close()
    return "\n".join(line.rstrip() for line in "".join(parser.parts).splitlines() if line.strip()) + "\n"
