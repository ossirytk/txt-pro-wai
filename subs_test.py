import sys
from pathlib import Path
from loguru import logger
from icecream import ic

logger.remove(0)
logger.add(sys.stdout, format="{level} {message}", level="INFO")

def main() -> None:
    logger.info("Processing subs")
    subs = Path() / "src_subs" / "HDMs3e1.srt"
    logger.debug("Reading subs from {subs}")
    with subs.open() as sub_file:
        raw_subtitles = sub_file.read()
        logger.debug("Read subs")

    ic(len(raw_subtitles))

    subtitle_items = raw_subtitles.split("\n\n")
    ic(len(subtitle_items))

    logger.info("Done")

if __name__ == "__main__":
    main()
