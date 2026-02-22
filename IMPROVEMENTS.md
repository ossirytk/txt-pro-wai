# Improvement Ideas for Book Scan Translator

## Current State

`book_scan_translator.py` provides a CLI pipeline that preprocesses photos of book pages, runs Tesseract OCR, and translates the extracted English text to Finnish using Google Translate.

---

## Improvement Ideas

### 1. PDF Input Support

Currently only individual image files are accepted. Adding PDF support (via `pdf2image` or `pymupdf`) would allow users to process entire scanned books without manually extracting pages first.

### 2. Alternative / Higher-Quality OCR Engines

Tesseract performs reasonably well but can struggle with unusual fonts, tight spacing, or degraded scans. Integrating alternatives such as:
- **EasyOCR** – good accuracy on many languages out of the box.
- **PaddleOCR** – strong on dense text layouts common in books.
- **Google Cloud Vision / AWS Textract** – cloud-based; higher accuracy at cost.

### 3. Better Translation Quality for Literary Text

Google Translate is adequate for technical text but can produce awkward results for literary prose. Alternatives:
- **DeepL API** – generally considered superior for European languages.
- **LibreTranslate** – self-hosted, avoids rate-limiting issues.

Allowing the user to select the translation backend via a `--backend` option would keep the script flexible.

### 4. Configurable Source and Target Languages

The translation is currently hard-coded to English → Finnish. Exposing `--source-lang` and `--target-lang` options (passed through to the translator) would make the script useful for any language pair supported by the chosen backend.

### 5. Automatic Source Language Detection

Instead of requiring users to specify the OCR and translation source language, the script could auto-detect the language of the extracted text (e.g., using `langdetect` or the translator's auto-detect mode) and skip translation if the text is already in Finnish.

### 6. Hyphenation and Line-Break Correction

Book OCR often produces artifacts where:
- End-of-line hyphens split words across lines (e.g. `mo-\nment` should become `moment`).
- Paragraph-final hard line breaks break the sentence flow.

A post-OCR cleanup step that joins hyphenated words and re-flows paragraphs would improve translation quality significantly.

### 7. Multi-Column Layout Handling

Many books, newspapers, and academic papers use two or more columns per page. Tesseract's default segmentation (PSM 3/6) often reads across columns rather than down each column. Detecting column boundaries and splitting the image before OCR would give much better results for such layouts.

### 8. OCR Confidence Filtering

Tesseract can output per-word confidence scores (using `--tsv` output format). Low-confidence words could be flagged, highlighted in a separate report, or excluded from translation so the user knows which passages need manual review.

### 9. GPU-Accelerated Image Preprocessing

OpenCV supports CUDA-accelerated operations for operations such as Gaussian blur, morphological transforms, and resizing. For large batches, building OpenCV with CUDA support and enabling GPU preprocessing could substantially reduce processing time.

### 10. Resumable Batch Processing

For large books (hundreds of pages), interrupted runs currently require reprocessing from scratch. Saving a progress file (e.g., a JSON manifest of completed pages) and skipping already-processed images on re-run would make the tool more robust for long jobs.

### 11. Structured Output Formats

Plain `.txt` output loses the page structure. Supporting richer output formats would improve usability:
- **EPUB** – reflowable e-book format, ideal for translated books.
- **DOCX** – editable in word processors.
- **HTML** – easy to read in a browser with per-page sections.

### 12. Per-Page Quality Report

After processing, produce a summary report (e.g., JSON or CSV) listing each page, the detected word count, average OCR confidence, character count before and after translation, and any errors encountered. This helps users assess overall scan quality without reading every output file.
