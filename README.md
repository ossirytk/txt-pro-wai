# txt-pro-wai

Text processing with AI.

## Scripts

- `subtitle_translator.py` translates subtitle files to Finnish.
- `book_scan_translator.py` scans text from page images and translates it to Finnish.
- `ebook_translator.py` reads ebook content and translates it to Finnish.

## Usage

Translate subtitles:

```bash
uv run subtitle_translator.py path/to/subtitles.srt
```

Translate scanned book page images:

```bash
uv run book_scan_translator.py "photos/*.jpg"
```

Translate an ebook file:

```bash
uv run ebook_translator.py path/to/book.epub
```

The ebook translator accepts `.epub`, `.txt`, `.html`, and `.xhtml` inputs and
writes translated plain text output to `text_output/` by default.
