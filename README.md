# txt-pro-wai

Text processing with AI.

## Scripts

- `subtitle_translator.py` translates subtitle files to Finnish.
- `book_scan_translator.py` scans text from page images and translates it to Finnish.
- `ebook_translator.py` reads ebook content and translates it to Finnish.

## Usage

Primary development workflow:

- Open the repository from the Windows dev drive in VS Code.
- Use the integrated PowerShell terminal for `uv run ...` commands.
- WSL/Ubuntu remains a supported alternative if you still use it.

Translate subtitles:

```powershell
uv run subtitle_translator.py path/to/subtitles.srt
```

Translate scanned book page images:

```powershell
uv run book_scan_translator.py "photos/*.jpg"
```

Translate an ebook file:

```powershell
uv run ebook_translator.py path/to/book.epub
```

The ebook translator accepts `.epub`, `.txt`, `.html`, and `.xhtml` inputs and
writes translated plain text output to `text_output/` by default.
