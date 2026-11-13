"""data_pipeline/ingestion.py

Utilities for loading raw documents from the ``data/raw`` directory.

Supported formats:
- Markdown ``.md``
- Plain text ``.txt``
- HTML ``.html``

Each file is turned into a :class:`data_pipeline.models.Document` instance.
"""

import json
import pathlib
import re
import uuid
from typing import Any, Dict, List, Optional

from .cleaner import html_to_markdown
from .models import Document

RAW_ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
MANIFEST_PATH = RAW_ROOT.parent / "source_manifest.json"


def load_source_manifest(path: pathlib.Path = MANIFEST_PATH) -> Dict[str, Dict[str, Any]]:
    """Load explicit provenance supplied by the document collector.

    URLs are never guessed from a filename or heading.  A manifest entry is keyed
    by the raw filename and may contain ``source_url``, ``title``, ``publisher``,
    and ``section_urls`` (a heading-to-canonical-URL mapping).
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("documents", payload)
    if not isinstance(entries, dict):
        raise ValueError("source manifest must contain an object named 'documents'")
    return {str(name): dict(value) for name, value in entries.items() if isinstance(value, dict)}


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


def _build_document(path: pathlib.Path, provenance: Optional[Dict[str, Any]] = None) -> Optional[Document]:
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
    if path.suffix.lower() == ".html" or re.match(r"\s*(?:<!doctype\s+html\b|<html\b)", content, re.IGNORECASE):
        content = html_to_markdown(content)
    if not content:
        return None

    provenance = provenance or {}
    # A stable ID lets persisted vector indexes and citations survive re-runs.
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))
    metadata = {
        "file_path": str(path),
        "size_bytes": path.stat().st_size,
        "source_file": path.name,
        "publisher": provenance.get("publisher"),
        "section_urls": provenance.get("section_urls", {}),
    }

    return Document(
        doc_id=doc_id,
        content=content,
        source_url=provenance.get("source_url"),
        file_name=provenance.get("title") or path.name,
        metadata=metadata,
    )


def load_documents(
    raw_root: pathlib.Path = RAW_ROOT,
    manifest_path: pathlib.Path = MANIFEST_PATH,
) -> List[Document]:
    """Walk ``data/raw`` and return a list of :class:`Document` objects.

    The function is deliberately tolerant – files that cannot be parsed are
    skipped with a silent ``continue``. This makes the pipeline robust when the
    raw directory contains auxiliary artefacts such as ``.gitkeep``.
    """
    docs: List[Document] = []
    manifest = load_source_manifest(manifest_path)
    for path in raw_root.rglob("*"):
        doc = _build_document(path, manifest.get(path.name))
        if doc:
            docs.append(doc)
    return docs

