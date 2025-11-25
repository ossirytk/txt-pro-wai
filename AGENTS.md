# AGENTS.md — Project Rules for AI Assistants (Python + Rust Workspace)

This repository is developed using VS Code and NvChad Neovim.  
Contributors run tooling from the terminal. Prefer CLI-first workflows and
keep diffs minimal, deterministic, and compliant with the repo’s tooling rules.

GitHub Copilot agents and other LLM-based assistants use this file to align with project-specific practices. [1](https://github.com/spanmartina/Text-Recognition-and-Translation-MLKit)  
VS Code’s agentic AI features can apply multi-file coordinated changes, so the rules below constrain that behavior. [2](https://developers.google.com/ml-kit/vision/text-recognition/v2/android)

---

## 1. Authoritative Tools & Source of Truth

### Python
- Ruff is the ONLY formatter + linter.
- Ruff’s configuration in `pyproject.toml` is authoritative.
- Do NOT reformat using Black, isort, yapf, or any editor formatter.
- Pyrefly is editor-only; do not modify code solely to appease it.

### Rust (Cargo Workspace)
- Formatting: `cargo fmt --all`
- Linting: `cargo clippy --workspace --all-targets` (default severity; no `-D warnings` unless instructed)
- If a `rustfmt.toml` exists, treat it as the single source of truth.

### Cross-Editor Compatibility
- Contributors use VS Code and Neovim.  
- All changes must be reproducible via terminal commands.

---

## 2. Python: Canonical Ruff Configuration

```toml
[tool.ruff]
target-version = "py313"
line-length = 120
lint.select = [
    "F","E","W","I","N","UP","YTT","ANN","ASYNC","S","FBT","B",
    "A","COM","C4","DTZ","EM","EXE","FA","ISC","ICN","LOG","G",
    "INP","PIE","T20","PYI","Q","RSE","RET","SLF","SLOT","SIM",
    "TID","TC","INT","ARG","PTH","PD","PGH","PL","TRY","FLY",
    "NPY","PERF","FURB","RUF",
]
lint.ignore = ["FBT001", "S603", "G004", "S607"]
lint.unfixable = [
    "F401",
]
