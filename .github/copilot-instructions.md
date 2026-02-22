# Copilot Instructions

This repository is a Python project for processing subtitles and images.
It is developed using VS Code and NvChad Neovim; prefer CLI-first workflows
and keep diffs minimal, deterministic, and compliant with the tooling rules below.

---

## Tooling

### Linting & Formatting (Python)
- **Ruff** is the only formatter and linter. Do NOT use Black, isort, yapf, or any other formatter.
- Run `ruff check .` to lint and `ruff format .` to format.
- The authoritative Ruff configuration lives in `pyproject.toml`.
- Pyrefly is editor-only; do not modify code solely to appease it.

### Package Management
- Use **uv** (`uv add`, `uv run`, etc.) to manage dependencies and run scripts.
- Do not edit `uv.lock` manually.

### Python Version
- Target Python **3.11+** (`requires-python = ">=3.11"` in `pyproject.toml`).

---

## Code Style

- `line-length = 120` (enforced by Ruff).
- All relative imports are banned (`ban-relative-imports = "all"`).
- Unused imports (`F401`) must be removed manually; the auto-fix is disabled.
- Ignored rules: `FBT001`, `G004`, `COM812`.

---

## Repository Layout

| Path | Purpose |
|---|---|
| `image_clean.py` | Image cleaning/processing script |
| `subtitle_fixer.py` | Subtitle correction script |
| `subtitle_translator.py` | Subtitle translation script |
| `src_subs/` | Source subtitle files |
| `pyproject.toml` | Project metadata and Ruff config |
| `requirements.txt` | Pinned dependencies (legacy reference) |

---

## Key Principles

- Make the **smallest possible change** that satisfies the requirement.
- Keep all changes reproducible from the terminal.
- Do not commit secrets, credentials, or large binary files.
