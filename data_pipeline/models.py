'''data_pipeline/models.py

Pydantic models for the Hybrid RAG pipeline.
'''

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a raw source document.

    Attributes
    ----------
    doc_id: str
        Unique identifier for the document (e.g., a UUID or filename).
    content: str
        Full text content of the document after cleaning.
    source_url: Optional[str]
        Original URL if the document was fetched from the web.
    file_name: Optional[str]
        Original file name on disk.
    metadata: Dict[str, Any]
        Arbitrary additional metadata (e.g., timestamps, author).
    """

    doc_id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Full text content of the document")
    source_url: Optional[str] = Field(None, description="Original source URL if applicable")
    file_name: Optional[str] = Field(None, description="Original file name on disk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class Chunk(BaseModel):
    """Represents a chunk of a document ready for embedding / retrieval.

    Attributes
    ----------
    chunk_id: str
        Unique identifier for the chunk.
    doc_id: str
        Identifier of the parent document.
    content: str
        Text content of the chunk.
    source_url: Optional[str]
        Propagated source URL.
    section_title: Optional[str]
        Title of the markdown section the chunk belongs to (if any).
    char_count: int
        Number of characters in the chunk.
    token_count: int
        Approximate token count (computed via whitespace split).
    metadata: Dict[str, Any]
        Additional metadata (e.g., page number, hierarchy).
    """

    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    doc_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Chunk text content")
    source_url: Optional[str] = Field(None, description="Source URL inherited from the document")
    section_title: Optional[str] = Field(None, description="Markdown heading that introduced the chunk")
    char_count: int = Field(..., description="Number of characters in the chunk")
    token_count: int = Field(..., description="Approximate token count for the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

