from data_pipeline.html_to_markdown import html_to_markdown


def test_html_conversion_preserves_headings_links_and_code():
    markdown = html_to_markdown(
        '<h1>Guide</h1><h2>Install</h2><p>Use <a href="/install">this guide</a>.</p><pre>print("ok")</pre>',
        base_url="https://docs.example.com/page",
    )
    assert "# Guide" in markdown
    assert "## Install" in markdown
    assert "[this guide](https://docs.example.com/install)" in markdown
    assert "```\nprint(\"ok\")\n```" in markdown
