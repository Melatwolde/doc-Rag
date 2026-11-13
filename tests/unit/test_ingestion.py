from __future__ import annotations

import json

from data_pipeline.ingestion import load_documents


def test_loader_extracts_html_structure_and_manifest_provenance(tmp_path):
	raw_dir = tmp_path / "raw"
	raw_dir.mkdir()
	html_path = raw_dir / "guide.md"
	html_path.write_text(
		"""<!doctype html>
<html><body>
<h1>JWT Guide</h1>
<p>Refresh with <a href="https://docs.example.test/auth#refresh">the endpoint</a>.</p>
<pre><code>refresh_access_token()</code></pre>
<script>ignore_this()</script>
</body></html>""",
		encoding="utf-8",
	)
	manifest_path = tmp_path / "source_manifest.json"
	manifest_path.write_text(
		json.dumps({
			"documents": {
				"guide.md": {
					"source_url": "https://docs.example.test/auth",
					"publisher": "Example Docs",
				}
			}
		}),
		encoding="utf-8",
	)

	documents = load_documents(raw_dir, manifest_path)

	assert len(documents) == 1
	document = documents[0]
	assert "# JWT Guide" in document.content
	assert "[the endpoint](https://docs.example.test/auth#refresh)" in document.content
	assert "```\nrefresh_access_token()\n```" in document.content
	assert "ignore_this" not in document.content
	assert document.source_url == "https://docs.example.test/auth"
	assert document.metadata["publisher"] == "Example Docs"


def test_loader_extracts_html_files(tmp_path):
	raw_dir = tmp_path / "raw"
	raw_dir.mkdir()
	(raw_dir / "page.html").write_text(
		"<html><body><h2>Install</h2><p>Run the command.</p></body></html>",
		encoding="utf-8",
	)

	documents = load_documents(raw_dir, tmp_path / "missing.json")

	assert len(documents) == 1
	assert documents[0].content == "## Install\nRun the command."
