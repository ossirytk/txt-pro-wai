"""Translate ebook content from English to Finnish.

Supports EPUB, TXT, HTML, and XHTML input files. EPUB content is read in spine
order and written to a plain text output file.

Usage:
    python ebook_translator.py book.epub
    uv run ebook_translator.py book.epub --out text_output
"""

import re
from pathlib import Path
from zipfile import BadZipFile

import click
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from loguru import logger

from translation_utils import DEFAULT_MAX_CHARS, DEFAULT_SAFETY, translate_paragraph_text

SUPPORTED_SUFFIXES = {".epub", ".txt", ".html", ".htm", ".xhtml"}
HTML_BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li")


def _normalize_block(text: str) -> str:
    """Collapse repeated whitespace inside an extracted text block."""
    return re.sub(r"\s+", " ", text).strip()


def _paragraphs_from_plain_text(text: str) -> list[str]:
    """Split plain text into normalized paragraphs."""
    paragraphs = [_normalize_block(part) for part in re.split(r"\n\s*\n", text) if part.strip()]
    return [paragraph for paragraph in paragraphs if paragraph]


def _extract_html_blocks(html_text: str | bytes) -> list[str]:
    """Extract readable blocks from HTML content."""
    soup = BeautifulSoup(html_text, "html.parser")
    blocks = [_normalize_block(tag.get_text(" ", strip=True)) for tag in soup.find_all(HTML_BLOCK_TAGS)]
    extracted_blocks = [block for block in blocks if block]

    if extracted_blocks:
        return extracted_blocks

    fallback_text = _normalize_block(soup.get_text("\n", strip=True))
    return [fallback_text] if fallback_text else []


def _iter_epub_documents(book: epub.EpubBook) -> list[epub.EpubHtml]:
    """Return EPUB document items in reading order."""
    document_items: list[epub.EpubHtml] = []
    for spine_item in book.spine:
        item_id = spine_item[0] if isinstance(spine_item, tuple) else spine_item
        if item_id == "nav":
            continue
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        document_items.append(item)

    if document_items:
        return document_items

    return list(book.get_items_of_type(ITEM_DOCUMENT))


def _extract_epub_paragraphs(input_path: Path) -> list[str]:
    """Read paragraphs from an EPUB file."""
    try:
        book = epub.read_epub(str(input_path))
    except (BadZipFile, OSError, KeyError, ValueError) as exc:
        msg = f"Failed to read EPUB file {input_path}: {exc}"
        raise click.ClickException(msg) from exc

    paragraphs: list[str] = []
    metadata = book.get_metadata("DC", "title")
    if metadata:
        title = _normalize_block(str(metadata[0][0]))
        if title:
            paragraphs.append(title)

    for item in _iter_epub_documents(book):
        paragraphs.extend(_extract_html_blocks(item.get_body_content()))

    return paragraphs


def _extract_input_paragraphs(input_path: Path) -> list[str]:
    """Read supported input formats into paragraph blocks."""
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        msg = f"Unsupported input format '{suffix}'. Supported formats: {supported}"
        raise click.ClickException(msg)

    if suffix == ".epub":
        return _extract_epub_paragraphs(input_path)

    try:
        source_text = input_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        msg = f"Failed to decode {input_path} as UTF-8"
        raise click.ClickException(msg) from exc
    except OSError as exc:
        msg = f"Failed to read {input_path}: {exc}"
        raise click.ClickException(msg) from exc

    if suffix in {".html", ".htm", ".xhtml"}:
        return _extract_html_blocks(source_text)

    return _paragraphs_from_plain_text(source_text)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("filename", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_dir",
    default="text_output",
    show_default=True,
    help="Output directory for translated ebook text files.",
)
@click.option(
    "--max-chars",
    default=DEFAULT_MAX_CHARS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum characters per translation request (soft limit).",
)
@click.option(
    "--safety",
    default=DEFAULT_SAFETY,
    show_default=True,
    type=click.FloatRange(0.0, 1.0),
    help="Safety multiplier (0..1) applied to --max-chars to avoid hitting API limits.",
)
@click.option(
    "--no-translate",
    is_flag=True,
    help="Skip translation and save extracted ebook text only.",
)
@click.option(
    "--source-lang",
    default="en",
    show_default=True,
    help="Translation source language (ISO 639-1 code, e.g. 'en', 'de').",
)
@click.option(
    "--target-lang",
    default="fi",
    show_default=True,
    help="Translation target language (ISO 639-1 code, e.g. 'fi', 'sv').",
)
@click.option(
    "--save-raw",
    is_flag=True,
    help="Also save extracted source text as {stem}.raw.txt.",
)
def translate_ebook(  # noqa: PLR0913
    filename: Path,
    out_dir: str,
    max_chars: int,
    safety: float,
    no_translate: bool,
    source_lang: str,
    target_lang: str,
    save_raw: bool,
) -> None:
    """Translate ebook content from English to Finnish and save it as plain text."""
    logger.info(f"Reading ebook content from {filename}")
    paragraphs = _extract_input_paragraphs(filename)
    if not paragraphs:
        msg = f"No readable text found in {filename}"
        raise click.ClickException(msg)

    raw_text = "\n\n".join(paragraphs)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_raw:
        raw_path = output_dir / f"{filename.stem}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")
        logger.info(f"Saved extracted source text to {raw_path}")

    if no_translate:
        output_text = raw_text
        output_path = output_dir / f"{filename.stem}.txt"
    else:
        logger.info(f"Translating ebook from {source_lang} to {target_lang}")
        output_text = translate_paragraph_text(
            raw_text,
            max_chars=max_chars,
            safety=safety,
            source_lang=source_lang,
            target_lang=target_lang,
            progress_label="Translating ebook",
        )
        output_path = output_dir / f"{filename.stem}_{target_lang}.txt"

    output_path.write_text(output_text, encoding="utf-8")
    click.echo(f"[OK] {filename} -> {output_path}")


if __name__ == "__main__":
    translate_ebook()
