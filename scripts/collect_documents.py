"""Download explicitly approved official documentation into ``data/raw``.

The script accepts only URLs listed in a user-maintained manifest. It makes no
claim that a URL is official and never invents a source URL or section anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.html_to_markdown import html_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect approved documentation URLs")
    parser.add_argument("manifest", type=Path, help="JSON manifest with a 'documents' object")
    parser.add_argument("--overwrite", action="store_true", help="Refresh an existing collected document")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Manifest not found: {args.manifest}")
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
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}. Re-run with --overwrite to refresh it.")
        
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urlopen(request, timeout=30) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get_content_type()

        if content_type == "text/html" or "<html" in raw_text.lower()[:1000]:
            destination.write_text(html_to_markdown(raw_text, base_url=url), encoding="utf-8")
        else:
            destination.write_text(raw_text, encoding="utf-8")
        provenance[filename] = entry
        print(f"Saved {url} -> {destination}")
        
    (ROOT / "data" / "source_manifest.json").write_text(json.dumps({"documents": provenance}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
