"""data_pipeline/chunker.py

Structure‑aware chunker for markdown documents.
"""

import re
import uuid
from typing import Any, Dict, List, Tuple

from .models import Document, Chunk
from .cleaner import clean_text

# Regex patterns for markdown headers (H1‑H3) and fenced code blocks
HEADER_REGEX = re.compile(r"^(#{1,3})\s+(.*)")
CODE_FENCE_REGEX = re.compile(r"^```")


def _token_count(text: str) -> int:
    """Very rough token estimator – split on whitespace.
    """
    return len(text.split())


def _split_section_into_chunks(
    doc_id: str,
    section_title: str,
    content: str,
    token_limit: int,
    overlap_ratio: float,
    source_url: str | None = None,
    document_metadata: Dict[str, Any] | None = None,
    line_start: int = 1,
) -> List[Chunk]:
    """Split a single markdown section into chunks.

    The algorithm walks through the lines, ensuring that fenced code blocks are kept
    intact. When the accumulated token count exceeds ``token_limit`` a new Chunk is
    emitted. Overlap between consecutive chunks is expressed as a ratio of the
    ``token_limit`` (e.g. 0.2 → 20 % overlap).
    """
    lines = content.split("\n")
    chunks: List[Chunk] = []
    buffer: List[str] = []
    in_code_block = False
    current_token = 0
    # Tokens that will be carried over to the next chunk for overlap
    overlap_tokens: List[str] = []

    chunk_start_line = line_start

    def flush_buffer(end_line: int):
        nonlocal buffer, current_token, overlap_tokens, chunk_start_line
        if not buffer:
            return
        chunk_text = "\n".join(buffer).strip()
        if not chunk_text:
            buffer = []
            current_token = 0
            return
        chunk_id = str(uuid.uuid4())
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=chunk_text,
            source_url=source_url,
            section_title=section_title or None,
            char_count=len(chunk_text),
            token_count=_token_count(chunk_text),
            metadata={
                **(document_metadata or {}),
                "heading_path": section_title or None,
                "line_start": chunk_start_line,
                "line_end": end_line,
            },
        )
        chunks.append(chunk)
        # Prepare overlap buffer
        overlap_token_target = int(token_limit * overlap_ratio)
        if overlap_token_target > 0:
            # take last ``overlap_token_target`` tokens from the current buffer
            overlap_text = " ".join(chunk_text.split()[-overlap_token_target:])
            overlap_tokens = overlap_text.split()
        else:
            overlap_tokens = []
        # Reset for next chunk (start with any overlap tokens)
        buffer = overlap_tokens.copy()
        current_token = len(overlap_tokens)
        chunk_start_line = end_line + 1

    for line_offset, line in enumerate(lines):
        stripped = line.strip()
        is_fence = bool(CODE_FENCE_REGEX.match(stripped))
        # Token count for the line (rough)
        line_tokens = _token_count(line)
        # If adding this line would exceed the limit, flush first (unless we are
        # inside a code block – including its opening and closing fence – we keep
        # the whole block together).
        if not in_code_block and not is_fence and current_token + line_tokens > token_limit and buffer:
            flush_buffer(line_start + line_offset - 1)
        buffer.append(line)
        current_token += line_tokens
        if is_fence:
            in_code_block = not in_code_block
    # Flush any remaining text
    flush_buffer(line_start + len(lines) - 1)
    return chunks


def chunk_document(
    doc: Document,
    token_limit: int = 512,
    overlap_ratio: float = 0.1,
) -> List[Chunk]:
    """Chunk a :class:`~data_pipeline.models.Document` into logical pieces.

    The steps are:
    1. Clean the raw text with :func:`data_pipeline.cleaner.clean_text`.
    2. Walk the document line‑by‑line, detecting markdown headers (H1‑H3) to
       create *sections*.
    3. Feed each section to :func:`_split_section_into_chunks` which respects the
       ``token_limit`` and does not cut fenced code blocks.
    """
    cleaned = clean_text(doc.content)
    lines = cleaned.split("\n")
    sections: List[Tuple[str, List[str], int]] = []  # (heading path, lines, first line)
    heading_stack: List[Tuple[int, str]] = []
    current_title: str = ""
    current_lines: List[str] = []
    section_start_line = 1
    in_code_block = False

    for line_number, line in enumerate(lines, start=1):
        # Track code fences so that a header inside a code block is ignored.
        if CODE_FENCE_REGEX.match(line.strip()):
            in_code_block = not in_code_block
        if not in_code_block:
            header_match = HEADER_REGEX.match(line)
            if header_match:
                # Save previous section
                if current_lines:
                    sections.append((current_title, current_lines, section_start_line))
                # Start new section
                level = len(header_match.group(1))
                heading = header_match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))
                current_title = " > ".join(item[1] for item in heading_stack)
                current_lines = []
                section_start_line = line_number + 1
                continue
        # Accumulate line for the current section
        current_lines.append(line)
    # Append final section
    if current_lines:
        sections.append((current_title, current_lines, section_start_line))

    # Chunk each section
    all_chunks: List[Chunk] = []
    section_urls = doc.metadata.get("section_urls", {})
    for title, sec_lines, first_line in sections:
        sec_text = "\n".join(sec_lines)
        source_url = section_urls.get(title) or section_urls.get(title.split(" > ")[-1]) or doc.source_url
        sec_chunks = _split_section_into_chunks(
            doc_id=doc.doc_id,
            section_title=title,
            content=sec_text,
            token_limit=token_limit,
            overlap_ratio=overlap_ratio,
            source_url=source_url,
            document_metadata={
                "source_file": doc.metadata.get("source_file", doc.file_name),
                "publisher": doc.metadata.get("publisher"),
            },
            line_start=first_line,
        )
        all_chunks.extend(sec_chunks)
    return all_chunks

