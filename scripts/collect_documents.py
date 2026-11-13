"""Download explicitly approved official documentation into ``data/raw``.

The script accepts only URLs listed in a user-maintained manifest. It makes no
claim that a URL is official and never invents a source URL or section anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


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
        
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urlopen(request, timeout=30) as response:
            html_content = response.read().decode("utf-8")

        # Parse HTML and extract clean readable text
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        body = soup.find(role="main") or soup.find("div", class_="body") or soup.body
        clean_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

        destination.write_text(clean_text, encoding="utf-8")
        provenance[filename] = entry
        print(f"Saved & Cleaned {url} -> {destination}")
        
    (ROOT / "data" / "source_manifest.json").write_text(json.dumps({"documents": provenance}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()