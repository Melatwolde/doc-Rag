from .cross_encoder import RerankedPassage


class ContextBuilder:
    """Assemble ranked passages into a source-labelled LLM context."""

    def __init__(self, max_context_chars: int = 12_000) -> None:
        if max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1")
        self.max_context_chars = max_context_chars

    def build(self, passages: list[RerankedPassage]) -> str:
        sections = []
        total_chars = 0

        for index, item in enumerate(passages, start=1):
            section = (
                f"[S{index}] Source ID: {item.source_id}\n"
                f"{item.text.strip()}\n"
            )
            separator = "\n" if sections else ""
            required = len(separator) + len(section)

            if total_chars + required > self.max_context_chars:
                break

            sections.append(section)
            total_chars += required

        return "\n".join(sections)
