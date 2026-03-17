# Contributing to OpenAIReview

## Getting Started

```bash
git clone https://github.com/ChicagoHAI/OpenAIReview.git
cd OpenAIReview
uv venv && uv pip install -e ".[dev]"
```

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run tests: `pytest tests/`
4. Open a PR against `main`

## Releasing

Releases are automated via CI. To trigger a release:

- **Bump the version** in both `pyproject.toml` and `src/reviewer/__init__.py` as part of your PR
- When the PR is merged to `main`, CI auto-tags, creates a GitHub release, and publishes to PyPI
- Use [semantic versioning](https://semver.org/): patch (0.2.x) for fixes, minor (0.x.0) for features, major (x.0.0) for breaking changes

Do **not** create version bump commits after merging — include the bump in the PR itself.

## Code Style

- Python 3.12+
- Keep changes focused — one feature or fix per PR
- Avoid adding dependencies unless necessary; use optional extras (`[mistral]`, `[deepseek]`) for heavy deps
- No docstrings or type annotations required for small changes, but keep existing ones consistent

## Testing

- Unit tests go in `tests/`
- Integration tests that call external APIs should check for the relevant API key and skip if not set
- Run `pytest tests/` before submitting

## PDF Parsing

If modifying PDF parsing (`parsers.py`):
- Test with at least one math-heavy and one table-heavy PDF
- OCR engines (Mistral, DeepSeek) are optional — code must work without them installed
- The fallback chain (Mistral > DeepSeek > Marker > pymupdf4llm) should always end with pymupdf4llm

## Questions

Open an issue at https://github.com/ChicagoHAI/OpenAIReview/issues
