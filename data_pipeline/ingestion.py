"""data_pipeline/ingestion.py

Utilities for loading raw documents from the ``data/raw`` directory.

Supported formats:
- Markdown ``.md``
- Plain text ``.txt``
- HTML ``.html``

Each file is turned into a :class:`data_pipeline.models.Document` instance.
"""

import pathlib
import uuid
from typing import List, Optional

from .models import Document

RAW_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def _read_file(path: pathlib.Path) -> str:
    """Read a file as UTF‑8 text, returning an empty string on failure.

    The function deliberately catches ``UnicodeDecodeError`` because some raw
    assets might be binary or encoded differently. In such cases we simply skip
    the file – the caller can decide whether to raise or ignore.
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _build_document(path: pathlib.Path) -> Optional[Document]:
    """Create a :class:`Document` from ``path``.

    Parameters
    ----------
    path:
        Absolute path to a supported file.

    Returns
    -------
    Document | None
        ``None`` if the file could not be read or has an unsupported suffix.
    """
    if not path.is_file():
        return None

    if path.suffix.lower() not in {".md", ".txt", ".html"}:
        return None

    content = _read_file(path)
    if not content:
        return None

    doc_id = str(uuid.uuid4())
    metadata = {
        "file_path": str(path),
        "size_bytes": path.stat().st_size,
    }

    return Document(
        doc_id=doc_id,
        content=content,
        source_url=None,
        file_name=path.name,
        metadata=metadata,
    )


def load_documents() -> List[Document]:
    """Walk ``data/raw`` and return a list of :class:`Document` objects.

    The function is deliberately tolerant – files that cannot be parsed are
    skipped with a silent ``continue``. This makes the pipeline robust when the
    raw directory contains auxiliary artefacts such as ``.gitkeep``.
    """
    docs: List[Document] = []
    for path in RAW_ROOT.rglob("*"):
        doc = _build_document(path)
        if doc:
            docs.append(doc)
    return docs

