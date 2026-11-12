"""data_pipeline/chunker.py

Structure‑aware chunker for markdown documents.
"""

import re
import uuid
from typing import List, Tuple

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

    def flush_buffer():
        nonlocal buffer, current_token, overlap_tokens
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
            source_url=None,
            section_title=section_title or None,
            char_count=len(chunk_text),
            token_count=_token_count(chunk_text),
            metadata={},
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

    for line in lines:
        stripped = line.strip()
        # Detect start/end of fenced code blocks – they are never split
        if CODE_FENCE_REGEX.match(stripped):
            in_code_block = not in_code_block
        # Token count for the line (rough)
        line_tokens = _token_count(line)
        # If adding this line would exceed the limit, flush first (unless we are
        # inside a code block – we keep the whole block together).
        if not in_code_block and current_token + line_tokens > token_limit and buffer:
            flush_buffer()
        buffer.append(line)
        current_token += line_tokens
    # Flush any remaining text
    flush_buffer()
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
    sections: List[Tuple[str, List[str]]] = []  # (title, lines)
    current_title: str = ""
    current_lines: List[str] = []
    in_code_block = False

    for line in lines:
        # Track code fences so that a header inside a code block is ignored.
        if CODE_FENCE_REGEX.match(line.strip()):
            in_code_block = not in_code_block
        if not in_code_block:
            header_match = HEADER_REGEX.match(line)
            if header_match:
                # Save previous section
                if current_lines:
                    sections.append((current_title, current_lines))
                # Start new section
                current_title = header_match.group(2).strip()
                current_lines = []
                continue
        # Accumulate line for the current section
        current_lines.append(line)
    # Append final section
    if current_lines:
        sections.append((current_title, current_lines))

    # Chunk each section
    all_chunks: List[Chunk] = []
    for title, sec_lines in sections:
        sec_text = "\n".join(sec_lines)
        sec_chunks = _split_section_into_chunks(
            doc_id=doc.doc_id,
            section_title=title,
            content=sec_text,
            token_limit=token_limit,
            overlap_ratio=overlap_ratio,
        )
        all_chunks.extend(sec_chunks)
    return all_chunks

