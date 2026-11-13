from __future__ import annotations

from data_pipeline.chunker import chunk_document
from data_pipeline.models import Document


def test_chunks_preserve_explicit_source_and_section_provenance():
    document = Document(
        doc_id="doc-1",
        file_name="guide.md",
        source_url="https://official.example/guide",
        content="# Guide\n\n## Install\n\nInstall it.\n\n```python\nprint('ready')\n```\n",
        metadata={
            "source_file": "guide.md",
            "publisher": "Official Example Docs",
            "section_urls": {"Guide > Install": "https://official.example/guide#install"},
        },
    )

    chunks = chunk_document(document, token_limit=3, overlap_ratio=0)

    assert chunks
    assert all(chunk.source_url == "https://official.example/guide#install" for chunk in chunks)
    assert all(chunk.metadata["heading_path"] == "Guide > Install" for chunk in chunks)
    assert all(chunk.metadata["source_file"] == "guide.md" for chunk in chunks)
    assert all(chunk.metadata["line_start"] <= chunk.metadata["line_end"] for chunk in chunks)
    assert "```python" in chunks[-1].content and "```" in chunks[-1].content


def test_numbered_heading_resolves_manifest_section_url():
    document = Document(
        doc_id="doc-2", content="# 4. Control Flow\n\n## 4.1. if Statements\n\nEvidence.",
        source_url="https://docs.example.com/controlflow",
        metadata={"section_urls": {"if Statements": "https://docs.example.com/controlflow#if-statements"}},
    )
    chunk = chunk_document(document, token_limit=50, overlap_ratio=0)[0]
    assert chunk.source_url == "https://docs.example.com/controlflow#if-statements"
