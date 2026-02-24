"""Book page photo scanner and translator.

Scans photos of book pages using OCR (Tesseract) and translates the extracted
text using GoogleTranslator. Defaults to English → Finnish.

Usage:
    python book_scan_translator.py "photos/*.jpg"
    uv run book_scan_translator.py "photos/*.jpg" --out translated_books
"""

import csv
import glob
import io
import json
import re
import subprocess
from pathlib import Path

import click
import cv2
import numpy as np
from deep_translator import GoogleTranslator
from loguru import logger

from image_clean import preprocess_image

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

DEFAULT_MAX_CHARS = 4000
DEFAULT_SAFETY = 0.9
DEFAULT_CONF_THRESHOLD = 0


def _run_tesseract_stdout(img_path: str, lang: str, psm: int, oem: int, timeout: int | None) -> str:
    """Run Tesseract OCR on a preprocessed image and return extracted text."""
    cmd = [
        "tesseract",
        img_path,
        "stdout",
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "-l",
        lang,
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    return result.stdout


def _run_tesseract_tsv(img_path: str, lang: str, psm: int, oem: int, timeout: int | None) -> str:
    """Run Tesseract OCR on a preprocessed image and return TSV output with per-word confidence data."""
    cmd = [
        "tesseract",
        img_path,
        "stdout",
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "-l",
        lang,
        "tsv",
    ]
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    return result.stdout


def _parse_tsv_confidence(
    tsv_text: str,
    conf_threshold: int = 0,
) -> tuple[str, float, int, list[str]]:
    """Parse Tesseract TSV output and extract text with confidence statistics.

    Words with a confidence score below *conf_threshold* (when > 0) are replaced
    by ``[???]`` in the returned text and collected in *low_confidence_words*.

    Returns:
        A tuple of ``(text, avg_confidence, word_count, low_confidence_words)``.
    """
    if not tsv_text.strip():
        return "", 0.0, 0, []

    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")

    # Map (page_num, block_num, par_num, line_num) -> ordered word list
    structure: dict[tuple[int, int, int, int], list[str]] = {}
    confidences: list[float] = []
    low_conf_words: list[str] = []

    for row in reader:
        if row.get("level") != "5":  # level 5 = word
            continue
        try:
            conf = float(row.get("conf", -1))
        except (ValueError, TypeError):
            continue
        word = (row.get("text") or "").strip()
        if not word or conf < 0:
            continue

        try:
            key = (int(row["page_num"]), int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
        except (KeyError, ValueError):
            continue

        confidences.append(conf)

        display_word = word
        if conf_threshold > 0 and conf < conf_threshold:
            low_conf_words.append(word)
            display_word = "[???]"

        if key not in structure:
            structure[key] = []
        structure[key].append(display_word)

    # Reconstruct text preserving paragraph and line structure
    sorted_keys = sorted(structure.keys())
    parts: list[str] = []
    prev_para: tuple[int, int, int] | None = None

    for key in sorted_keys:
        para = key[:3]
        if prev_para is not None and para != prev_para:
            parts.append("")  # blank line = paragraph break
        parts.append(" ".join(structure[key]))
        prev_para = para

    text = "\n".join(parts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf, len(confidences), low_conf_words


def _fix_hyphenation(text: str) -> str:
    """Fix common OCR hyphenation and line-break artifacts.

    - Joins end-of-line hyphenated words (e.g. ``'mo-\\nment'`` → ``'moment'``).
    - Converts isolated single newlines within a paragraph to spaces so that
      sentences flow correctly into the translator.
    """
    # Join hyphenated line-breaks: "word-\nnext" -> "wordnext"
    # Replace lone newlines (not part of a paragraph break) with a space
    return re.sub(r"(?<!\n)\n(?!\n)", " ", re.sub(r"(\w)-\n(\w)", r"\1\2", text))


def _translate_text(text: str, max_chars: int, safety: float, source_lang: str = "en", target_lang: str = "fi") -> str:
    """Translate plain text in paragraph-based chunks, respecting the translator character limit.

    Respects the translator character limit by batching paragraphs.
    """
    if not text.strip():
        return text

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    effective = int(max_chars * safety)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return text

    translated_parts: list[str] = []
    current_batch: list[str] = []
    current_len = 0

    for para in paragraphs:
        p_len = len(para)
        sep = 2 if current_batch else 0

        if p_len > effective:
            # Flush current batch before handling oversized paragraph
            if current_batch:
                chunk = "\n\n".join(current_batch)
                translated = translator.translate(chunk)
                translated_parts.append(translated or chunk)
                current_batch = []
                current_len = 0
            # Translate oversized paragraph in word-boundary-aware slices
            start = 0
            while start < p_len:
                end = min(start + effective, p_len)
                if end < p_len:
                    window = para[start:end]
                    split_at = window.rfind(" ")
                    if split_at > 0:
                        end = start + split_at
                piece = para[start:end].strip()
                if not piece:
                    start = end + 1
                    continue
                translated = translator.translate(piece)
                translated_parts.append(translated or piece)
                start = end
        elif current_len + p_len + sep > effective:
            # Flush and start a new batch
            chunk = "\n\n".join(current_batch)
            translated = translator.translate(chunk)
            translated_parts.append(translated or chunk)
            current_batch = [para]
            current_len = p_len
        else:
            current_batch.append(para)
            current_len += p_len + sep

    if current_batch:
        chunk = "\n\n".join(current_batch)
        translated = translator.translate(chunk)
        translated_parts.append(translated or chunk)

    return "\n\n".join(translated_parts)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_glob", type=str)
@click.option(
    "--out",
    "out_dir",
    default="book_output",
    show_default=True,
    help="Output directory for translated text files and intermediate .pre.png files.",
)
@click.option(
    "--lang",
    default="eng",
    show_default=True,
    help="Tesseract OCR language(s), e.g. 'eng' or 'fin+eng'.",
)
@click.option(
    "--psm",
    default=3,
    show_default=True,
    type=click.IntRange(0, 13),
    help="Tesseract page segmentation mode (0..13). Default 3 = fully automatic page segmentation.",
)
@click.option(
    "--oem",
    default=1,
    show_default=True,
    type=click.IntRange(0, 3),
    help="Tesseract OCR engine mode (0..3). Default 1 = LSTM only.",
)
@click.option(
    "--threshold",
    "threshold_method",
    type=click.Choice(["sauvola", "otsu"], case_sensitive=False),
    default="sauvola",
    show_default=True,
    help="Binarization method for image preprocessing.",
)
@click.option(
    "--upscale",
    "upscale_factor",
    default=2.0,
    show_default=True,
    type=click.FloatRange(1.0, 6.0),
    help="Upscale factor (bicubic). 2.0 is a good default for small fonts.",
)
@click.option(
    "--no-perspective",
    is_flag=True,
    help="Disable perspective correction.",
)
@click.option(
    "--rotate",
    default="0",
    show_default=True,
    type=click.Choice(["0", "90", "180", "270"]),
    help="Rotate image clockwise by angle before processing.",
)
@click.option(
    "--sauvola-window",
    default=25,
    show_default=True,
    type=click.IntRange(9, 101),
    help="Sauvola window size (odd number recommended).",
)
@click.option(
    "--sauvola-k",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0.05, 0.8),
    help="Sauvola k parameter.",
)
@click.option(
    "--sharpen-amount",
    default=0.8,
    show_default=True,
    type=click.FloatRange(0.0, 2.0),
    help="Unsharp mask amount.",
)
@click.option(
    "--sharpen-radius",
    default=1.2,
    show_default=True,
    type=click.FloatRange(0.3, 5.0),
    help="Unsharp mask radius.",
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
    help="Skip translation and save raw OCR text only.",
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
    "--conf-threshold",
    default=DEFAULT_CONF_THRESHOLD,
    show_default=True,
    type=click.IntRange(0, 100),
    help="OCR confidence threshold (0-100). Words below this score are replaced with [???]. 0 disables filtering.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Skip pages already listed in {out}/progress.json from a previous run.",
)
@click.option(
    "--no-report",
    is_flag=True,
    help="Disable the per-page quality report (quality_report.json).",
)
@click.option(
    "--keep-pre",
    is_flag=True,
    help="Keep preprocessed .pre.png images after OCR.",
)
@click.option(
    "--save-raw",
    is_flag=True,
    help="Also save raw OCR text (before translation) as {stem}.raw.txt.",
)
@click.option(
    "--ocr-timeout",
    default=300,
    show_default=True,
    type=int,
    help="Timeout in seconds for each Tesseract OCR call (0 = no timeout).",
)
def main(  # noqa: PLR0912, PLR0913, PLR0915
    input_glob: str,
    out_dir: str,
    lang: str,
    psm: int,
    oem: int,
    threshold_method: str,
    upscale_factor: float,
    no_perspective: bool,
    rotate: str,
    sauvola_window: int,
    sauvola_k: float,
    sharpen_amount: float,
    sharpen_radius: float,
    max_chars: int,
    safety: float,
    no_translate: bool,
    source_lang: str,
    target_lang: str,
    conf_threshold: int,
    resume: bool,
    no_report: bool,
    keep_pre: bool,
    save_raw: bool,
    ocr_timeout: int,
) -> None:
    """
    Scan book page photos with OCR and translate text between any language pair.

    INPUT_GLOB example: "book_photos/*.jpg"
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    imgs = sorted(glob.glob(input_glob))  # noqa: PTH207
    if not imgs:
        msg = f"No images found for pattern: {input_glob}"
        raise click.ClickException(msg)

    use_perspective = not no_perspective
    rotate_int = int(rotate)

    stems = [Path(p).stem for p in imgs]
    has_duplicates = len(stems) != len(set(stems))
    if has_duplicates:
        click.echo(
            "[WARN] Multiple input images share the same filename stem. "
            "Output files will be prefixed with a page index to prevent collisions.",
            err=True,
        )

    pbar = tqdm(total=len(imgs), desc="Processing pages", unit="page") if tqdm is not None else None

    # Resumable batch processing: load previously-completed pages
    progress_path = out_path / "progress.json"
    completed_pages: set[str] = set()
    if resume and progress_path.exists():
        try:
            raw = json.loads(progress_path.read_text(encoding="utf-8"))
            completed_pages = set(raw.get("completed", []))
            logger.info(f"Resuming: {len(completed_pages)} page(s) already completed")
        except Exception as exc:
            logger.warning(f"Could not load progress file: {exc}")

    # Per-page quality report data
    quality_data: list[dict[str, object]] = []

    try:
        for idx, img_path in enumerate(imgs, 1):
            stem = Path(img_path).stem
            output_stem = f"{idx:04d}_{stem}" if has_duplicates else stem
            pre_path = out_path / f"{output_stem}.pre.png"

            # Resumable processing: skip already-completed pages
            if resume and img_path in completed_pages:
                logger.info(f"Skipping (already processed): {img_path}")
                if pbar is not None:
                    pbar.update(1)
                else:
                    logger.info(f"Progress: {idx}/{len(imgs)} pages processed")
                continue

            word_count = 0
            avg_conf = 0.0

            try:
                # Step 1: Preprocess image
                logger.info(f"Preprocessing {img_path}")
                th = preprocess_image(
                    img_path,
                    upscale_factor=upscale_factor,
                    threshold_method=threshold_method,
                    use_perspective=use_perspective,
                    sauvola_window=sauvola_window,
                    sauvola_k=sauvola_k,
                    sharpen_amount=sharpen_amount,
                    sharpen_radius=sharpen_radius,
                    rotate=rotate_int,
                )
                if th.dtype != np.uint8:
                    th = th.astype(np.uint8)
                cv2.imwrite(str(pre_path), th, [cv2.IMWRITE_PNG_COMPRESSION, 9])

                # Step 2: OCR with per-word confidence data
                logger.info(f"Running OCR on {pre_path}")
                timeout = ocr_timeout if ocr_timeout > 0 else None
                tsv_text = _run_tesseract_tsv(str(pre_path), lang=lang, psm=psm, oem=oem, timeout=timeout)
                raw_text, avg_conf, word_count, low_conf_words = _parse_tsv_confidence(tsv_text, conf_threshold)

                if low_conf_words:
                    logger.warning(f"{len(low_conf_words)} low-confidence word(s) flagged in {img_path}")

                # Step 2b: Fix hyphenation and line-break artifacts
                raw_text = _fix_hyphenation(raw_text)

                if save_raw:
                    raw_out = out_path / f"{output_stem}.raw.txt"
                    raw_out.write_text(raw_text, encoding="utf-8")
                    logger.info(f"Raw OCR saved to {raw_out}")

                # Step 3: Translate (or save raw)
                if no_translate:
                    output_text = raw_text
                    out_file = out_path / f"{output_stem}.txt"
                else:
                    logger.info(f"Translating text for {img_path}")
                    output_text = _translate_text(
                        raw_text,
                        max_chars=max_chars,
                        safety=safety,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    out_file = out_path / f"{output_stem}_{target_lang}.txt"

                out_file.write_text(output_text, encoding="utf-8")
                click.echo(f"[OK] {img_path} -> {out_file}")

                if not keep_pre and pre_path.exists():
                    pre_path.unlink()

                # Update progress manifest
                if resume:
                    completed_pages.add(img_path)
                    progress_path.write_text(
                        json.dumps({"completed": sorted(completed_pages)}, indent=2),
                        encoding="utf-8",
                    )

                quality_data.append(
                    {
                        "page": img_path,
                        "word_count": word_count,
                        "avg_confidence": round(avg_conf, 2),
                        "chars_before_translation": len(raw_text),
                        "chars_after_translation": len(output_text),
                        "status": "ok",
                        "error": None,
                    }
                )

            except subprocess.TimeoutExpired as e:
                click.echo(f"[TESSERACT TIMEOUT] {img_path}: {e}", err=True)
                quality_data.append(
                    {
                        "page": img_path,
                        "word_count": word_count,
                        "avg_confidence": round(avg_conf, 2),
                        "chars_before_translation": 0,
                        "chars_after_translation": 0,
                        "status": "error",
                        "error": f"Tesseract timeout: {e}",
                    }
                )
            except subprocess.CalledProcessError as e:
                msg = str(e)
                stderr = getattr(e, "stderr", None)
                if stderr:
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode(errors="ignore")
                    msg = f"{msg}\nTesseract stderr:\n{stderr}"
                click.echo(f"[TESSERACT FAIL] {img_path}: {msg}", err=True)
                quality_data.append(
                    {
                        "page": img_path,
                        "word_count": word_count,
                        "avg_confidence": round(avg_conf, 2),
                        "chars_before_translation": 0,
                        "chars_after_translation": 0,
                        "status": "error",
                        "error": msg,
                    }
                )
            except Exception as e:
                click.echo(f"[FAIL] {img_path}: {e}", err=True)
                quality_data.append(
                    {
                        "page": img_path,
                        "word_count": word_count,
                        "avg_confidence": round(avg_conf, 2),
                        "chars_before_translation": 0,
                        "chars_after_translation": 0,
                        "status": "error",
                        "error": str(e),
                    }
                )

            if pbar is not None:
                pbar.update(1)
            else:
                logger.info(f"Progress: {idx}/{len(imgs)} pages processed")
    finally:
        if pbar is not None:
            pbar.close()

    # Write per-page quality report
    if not no_report and quality_data:
        report_path = out_path / "quality_report.json"
        report_path.write_text(json.dumps(quality_data, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"[REPORT] Quality report written to {report_path}")


if __name__ == "__main__":
    main()
