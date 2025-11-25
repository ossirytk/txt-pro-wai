import sys
from pathlib import Path

import click
from deep_translator import GoogleTranslator
from loguru import logger

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _read_subtitle_file(subs: Path) -> str:
    """Read and validate subtitle file."""
    if not subs.exists():
        logger.error(f"File not found: {subs}")
        sys.exit(1)
    if not subs.is_file():
        logger.error(f"Path is not a file: {subs}")
        sys.exit(1)

    logger.info(f"Reading subs from {subs}")
    try:
        with subs.open(encoding="utf8") as sub_file:
            raw_subtitles = sub_file.read()
            logger.debug("Read subs")
    except UnicodeDecodeError:
        logger.error(f"Failed to decode {subs} as UTF-8")
        sys.exit(1)
    except OSError as e:
        logger.error(f"Error reading file: {e}")
        sys.exit(1)

    if not raw_subtitles.strip():
        logger.error("Subtitle file is empty")
        sys.exit(1)

    return raw_subtitles


def _parse_subtitles(raw_subtitles: str) -> tuple[list[dict], list[str]]:
    """Parse subtitles into items and content lines."""
    subtitle_items = raw_subtitles.split("\n\n")
    total_items = []
    all_lines = []

    logger.info(f"Translating {len(subtitle_items)} items")
    for idx, item in enumerate(subtitle_items):
        if not item.strip():
            continue
        lines = item.split("\n")
        if len(lines) < 2:
            logger.warning(f"Subtitle item {idx} has insufficient lines, skipping")
            continue

        sub_id = lines.pop(0)
        sub_time = lines.pop(0)
        line_number = len(lines)
        if line_number == 0:
            logger.warning(f"Subtitle item {idx} has no content lines, skipping")
            continue
        all_lines.extend(lines)
        sub_dict = {"sub_id": sub_id, "sub_time": sub_time, "line_number": line_number}
        total_items.append(sub_dict)

    if not all_lines:
        logger.error("No valid subtitle lines found")
        sys.exit(1)

    return total_items, all_lines


def _write_translated_subtitles(
    output_path: Path,
    total_items: list[dict],
    translated_lines: list[str],
) -> None:
    """Write translated subtitles to output file."""
    try:
        index = 0
        with output_path.open("w", encoding="utf8") as out:
            for subtitle_item in total_items:
                out.write(subtitle_item["sub_id"] + "\n")
                out.write(subtitle_item["sub_time"] + "\n")
                for _n in range(subtitle_item["line_number"]):
                    out.write(translated_lines[index] + "\n")
                    index += 1
                out.write("\n")
    except OSError as e:
        logger.error(f"Error writing output file: {e}")
        sys.exit(1)


def _batched_lines(lines: list[str], batch_size: int):
    """Yield successive batches of `batch_size` from `lines`."""
    for i in range(0, len(lines), batch_size):
        yield lines[i : i + batch_size]


def _split_line_by_chars(line: str, limit: int) -> list[str]:
    """Hard-split a single long line into pieces no longer than `limit`."""
    if not line:
        return [""]
    parts: list[str] = []
    start = 0
    while start < len(line):
        parts.append(line[start : start + limit])
        start += limit
    return parts


def _batched_by_chars(lines: list[str], max_chars: int, safety: float):
    """Yield sub-batches of `lines` whose joined character length stays under limit.

    Uses `safety` (0..1) to leave a margin under `max_chars`.
    """
    effective = int(max_chars * safety)
    batch: list[str] = []
    batch_len = 0
    for line in lines:
        l = line.strip()
        l_len = len(l) if l else 1

        if l_len > effective:
            # flush current batch
            if batch:
                yield batch
                batch = []
                batch_len = 0
            # split the too-long line into pieces
            for part in _split_line_by_chars(l, effective):
                yield [part]
            continue

        # +1 for a separator when joining lines
        sep = 1 if batch else 0
        if batch_len + l_len + sep > effective:
            yield batch
            batch = [l]
            batch_len = l_len
        else:
            batch.append(l)
            batch_len += l_len + sep

    if batch:
        yield batch


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--batch-size",
    type=int,
    default=500,
    show_default=True,
    help=(
        "Number of subtitle lines to send per request to GoogleTranslator. "
        "Increase for fewer requests; decrease if you hit size limits."
    ),
)
@click.option(
    "--max-chars",
    type=int,
    default=4000,
    show_default=True,
    help=(
        "Maximum characters per request accepted by the translator (soft limit)."
    ),
)
@click.option(
    "--safety",
    type=float,
    default=0.9,
    show_default=True,
    help=(
        "Safety multiplier (0..1) applied to `--max-chars` to avoid hitting limits."
    ),
)
def translate(filename: str, batch_size: int, max_chars: int, safety: float) -> None:
    """
    Translate an English Subtitles file into Finnish.
    Run with python -m subtitle_translator <path to sub file>
    """
    try:
        logger.info("Subs translation started")
        subs = Path(filename)

        raw_subtitles = _read_subtitle_file(subs)
        total_items, all_lines = _parse_subtitles(raw_subtitles)

        translator = GoogleTranslator(source="en", target="fi")
        try:
            total = len(all_lines)
            logger.info(f"Translating {total} lines with GoogleTranslator in batches")

            # Batch size controls per-request payload; adjust if needed.
            translated_lines: list[str] = []

            batches = list(_batched_lines(all_lines, batch_size))

            if tqdm is not None:
                pbar = tqdm(total=len(batches), desc="Translating batches", unit="batch")
                try:
                    for batch in batches:
                        # further split by chars/safety to avoid API size limits
                        for sub in _batched_by_chars(batch, max_chars, safety):
                            translated_batch = translator.translate_batch(batch=sub)
                            translated_lines.extend(translated_batch)
                        pbar.update(1)
                finally:
                    pbar.close()
            else:
                for idx, batch in enumerate(batches, start=1):
                    for sub in _batched_by_chars(batch, max_chars, safety):
                        translated_batch = translator.translate_batch(batch=sub)
                        translated_lines.extend(translated_batch)
                    done = len(translated_lines)
                    pct = done / total * 100 if total else 100.0
                    logger.info(f"Progress: {done}/{total} lines translated ({pct:.1f}%)")

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            sys.exit(1)

        output_filename = subs.stem + "_fin.srt"
        output_dir = Path("subs_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        _write_translated_subtitles(output_path, total_items, translated_lines)

        logger.info(f"Translation completed successfully. Output saved to {output_path}")

    except KeyboardInterrupt:
        logger.warning("Translation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    translate()
