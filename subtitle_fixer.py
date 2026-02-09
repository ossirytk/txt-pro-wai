from __future__ import annotations

import re
from pathlib import Path

import click

TIMECODE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})",
)


def _normalize_timecode(value: str) -> str:
    return value.replace(".", ",")


def fix_srt_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    matches = list(TIMECODE_RE.finditer(text))
    entries: list[tuple[str, str, list[str]]] = []

    for index, match in enumerate(matches):
        start = _normalize_timecode(match.group("start"))
        end = _normalize_timecode(match.group("end"))
        block_start = match.end()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block_text = text[block_start:block_end]

        cleaned_lines: list[str] = []
        for raw_line in block_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            cleaned_lines.append(line)

        if not cleaned_lines:
            continue

        entries.append((start, end, cleaned_lines))

    if not entries:
        return ""

    output_lines: list[str] = []
    for index, (start, end, lines) in enumerate(entries, start=1):
        output_lines.append(str(index))
        output_lines.append(f"{start} --> {end}")
        output_lines.extend(lines)
        output_lines.append("")

    return "\n".join(output_lines).rstrip() + "\n"


def _default_output_path(input_path: Path, suffix: str) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.{suffix}{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}.{suffix}")


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path. Defaults to <stem>.<suffix><ext> in the same folder.",
)
@click.option("--suffix", default="fixed", show_default=True, help="Identifier added to the output file name.")
def main(input_path: Path, output: Path | None, suffix: str) -> None:
    raw_text = input_path.read_text(encoding="utf-8-sig", errors="replace")
    fixed_text = fix_srt_text(raw_text)

    if not fixed_text:
        msg = "No valid timecodes were found in the input file."
        raise click.ClickException(msg)

    output_path = output or _default_output_path(input_path, suffix)
    output_path.write_text(fixed_text, encoding="utf-8")
    click.echo(f"Saved fixed subtitles to {output_path}")


if __name__ == "__main__":
    main()
