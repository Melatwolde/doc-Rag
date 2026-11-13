"""Download explicitly approved official documentation into ``data/raw``.

The script accepts only URLs listed in a user-maintained manifest. It makes no
claim that a URL is official and never invents a source URL or section anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect approved documentation URLs")
    parser.add_argument("manifest", type=Path, help="JSON manifest with a 'documents' object")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = manifest.get("documents", manifest)
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for filename, entry in documents.items():
        url = entry.get("source_url")
        if not url:
            raise ValueError(f"{filename} has no source_url")
        destination = raw_dir / filename
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
        request = Request(url, headers={"User-Agent": "DocuRAG documentation collector"})
        with urlopen(request, timeout=30) as response:
            destination.write_bytes(response.read())
        provenance[filename] = entry
        print(f"Saved {url} -> {destination}")
    (ROOT / "data" / "source_manifest.json").write_text(json.dumps({"documents": provenance}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
