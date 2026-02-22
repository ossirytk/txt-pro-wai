"""Book page photo scanner and translator.

Scans photos of book pages using OCR (Tesseract) and translates the extracted
text from English to Finnish using GoogleTranslator.

Usage:
    python book_scan_translator.py "photos/*.jpg"
    uv run book_scan_translator.py "photos/*.jpg" --out translated_books
"""

import glob as glob_module
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


def _translate_text(text: str, max_chars: int, safety: float) -> str:
    """Translate English plain text to Finnish in paragraph-based chunks.

    Respects the translator character limit by batching paragraphs.
    """
    if not text.strip():
        return text

    translator = GoogleTranslator(source="en", target="fi")
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
            # Translate oversized paragraph in slices
            for i in range(0, p_len, effective):
                piece = para[i : i + effective]
                translated = translator.translate(piece)
                translated_parts.append(translated or piece)
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
    type=int,
    help="Maximum characters per translation request (soft limit).",
)
@click.option(
    "--safety",
    default=DEFAULT_SAFETY,
    show_default=True,
    type=float,
    help="Safety multiplier (0..1) applied to --max-chars to avoid hitting API limits.",
)
@click.option(
    "--no-translate",
    is_flag=True,
    help="Skip translation and save raw OCR text only.",
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
def main(  # noqa: PLR0912, PLR0913
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
    keep_pre: bool,
    save_raw: bool,
    ocr_timeout: int,
) -> None:
    """
    Scan book page photos with OCR and translate English text to Finnish.

    INPUT_GLOB example: "book_photos/*.jpg"
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    imgs = sorted(glob_module.glob(input_glob))  # noqa: PTH207
    if not imgs:
        msg = f"No images found for pattern: {input_glob}"
        raise click.ClickException(msg)

    use_perspective = not no_perspective
    rotate_int = int(rotate)

    pbar = tqdm(total=len(imgs), desc="Processing pages", unit="page") if tqdm is not None else None

    try:
        for idx, img_path in enumerate(imgs, 1):
            stem = Path(img_path).stem
            pre_path = out_path / f"{stem}.pre.png"

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

                # Step 2: OCR
                logger.info(f"Running OCR on {pre_path}")
                timeout = ocr_timeout if ocr_timeout > 0 else None
                raw_text = _run_tesseract_stdout(str(pre_path), lang=lang, psm=psm, oem=oem, timeout=timeout)

                if save_raw:
                    raw_out = out_path / f"{stem}.raw.txt"
                    raw_out.write_text(raw_text, encoding="utf-8")
                    logger.info(f"Raw OCR saved to {raw_out}")

                # Step 3: Translate (or save raw)
                if no_translate:
                    output_text = raw_text
                    out_file = out_path / f"{stem}.txt"
                else:
                    logger.info(f"Translating text for {img_path}")
                    output_text = _translate_text(raw_text, max_chars=max_chars, safety=safety)
                    out_file = out_path / f"{stem}_fi.txt"

                out_file.write_text(output_text, encoding="utf-8")
                click.echo(f"[OK] {img_path} -> {out_file}")

                if not keep_pre and pre_path.exists():
                    pre_path.unlink()

            except subprocess.CalledProcessError as e:
                click.echo(f"[TESSERACT FAIL] {img_path}: {e}", err=True)
            except Exception as e:
                click.echo(f"[FAIL] {img_path}: {e}", err=True)

            if pbar is not None:
                pbar.update(1)
            else:
                logger.info(f"Progress: {idx}/{len(imgs)} pages processed")
    finally:
        if pbar is not None:
            pbar.close()


if __name__ == "__main__":
    main()
