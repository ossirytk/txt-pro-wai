"""Shared helpers for translating long-form text in safe-sized chunks."""

from collections.abc import Iterator

from deep_translator import GoogleTranslator

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

DEFAULT_MAX_CHARS = 4000
DEFAULT_SAFETY = 0.9


def _effective_limit(max_chars: int, safety: float) -> int:
    """Return the effective chunk size after applying the safety factor."""
    effective = int(max_chars * safety)
    if effective < 1:
        msg = "The effective translation limit must be at least 1 character"
        raise ValueError(msg)
    return effective


def _split_oversized_paragraph(paragraph: str, limit: int) -> Iterator[str]:
    """Yield word-aware slices from a paragraph that exceeds the size limit."""
    start = 0
    paragraph_len = len(paragraph)

    while start < paragraph_len:
        end = min(start + limit, paragraph_len)
        if end < paragraph_len:
            split_at = paragraph.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        piece = paragraph[start:end].strip()
        if piece:
            yield piece

        if end <= start:
            start += limit
        else:
            start = end

        while start < paragraph_len and paragraph[start].isspace():
            start += 1


def chunk_paragraphs(paragraphs: list[str], max_chars: int, safety: float) -> list[str]:
    """Group paragraphs into translation chunks that stay below the size limit."""
    effective = _effective_limit(max_chars, safety)
    chunks: list[str] = []
    current_batch: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()
        if not clean_paragraph:
            continue

        paragraph_len = len(clean_paragraph)
        separator_len = 2 if current_batch else 0

        if paragraph_len > effective:
            if current_batch:
                chunks.append("\n\n".join(current_batch))
                current_batch = []
                current_len = 0
            chunks.extend(_split_oversized_paragraph(clean_paragraph, effective))
            continue

        if current_len + paragraph_len + separator_len > effective:
            chunks.append("\n\n".join(current_batch))
            current_batch = [clean_paragraph]
            current_len = paragraph_len
            continue

        current_batch.append(clean_paragraph)
        current_len += paragraph_len + separator_len

    if current_batch:
        chunks.append("\n\n".join(current_batch))

    return chunks


def translate_paragraph_text(  # noqa: PLR0913
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    safety: float = DEFAULT_SAFETY,
    source_lang: str = "en",
    target_lang: str = "fi",
    progress_label: str | None = None,
) -> str:
    """Translate paragraph-based text while preserving paragraph breaks."""
    if not text.strip():
        return text

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return text

    chunks = chunk_paragraphs(paragraphs, max_chars=max_chars, safety=safety)
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated_parts: list[str] = []

    if progress_label and tqdm is not None:
        progress = tqdm(chunks, desc=progress_label, unit="chunk")
        try:
            for chunk in progress:
                translated = translator.translate(chunk)
                if translated is None:
                    msg = "Translator returned no content"
                    raise RuntimeError(msg)
                translated_parts.append(translated)
        finally:
            progress.close()
    else:
        for chunk in chunks:
            translated = translator.translate(chunk)
            if translated is None:
                msg = "Translator returned no content"
                raise RuntimeError(msg)
            translated_parts.append(translated)

    return "\n\n".join(translated_parts)
