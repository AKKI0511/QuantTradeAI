# Repository Guidelines

## Project Structure & Module Organization
- Code lives in `quanttradeai/` (CLI in `quanttradeai/main.py`): `data/`, `features/`, `models/`, `backtest/`, `trading/`, `streaming/`, `utils/`.
- Config files in `config/` (e.g., `model_config.yaml`, `backtest_config.yaml`).
- Tests mirror the package in `tests/` (e.g., `tests/models/`, `tests/streaming/`).
- Datasets and cache in `data/`; trained artifacts in `models/`; docs in `docs/`.

## Build, Test, and Development Commands
- Install dev deps: `poetry install --with dev`
- Format code: `make format` (Black)
- Lint code: `make lint` (Flake8)
- Run tests: `make test` or `poetry run pytest`
- Run pipeline: `make pipeline` or `poetry run quanttradeai train -c config/model_config.yaml`
- CLI help: `poetry run quanttradeai --help`
- Package build (optional): `poetry build`

## Coding Style & Naming Conventions
- Python 3.11+, format with Black (line length 88). Lint via Flake8 (ignores: E203, W503, etc. per `.flake8`).
- Naming: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE_CASE (constants). Files: `snake_case.py`.
- Tests named `test_*.py`; keep public imports curated in `quanttradeai/__init__.py`.
- Avoid unused imports/variables; prefer pure, side‑effect‑free helpers in `utils/`.

## Testing Guidelines
- Framework: Pytest. Place unit tests under matching folders in `tests/`.
- Conventions: files `test_<unit>.py`, tests `test_<behavior>`; use fixtures/mocks for I/O, streaming, and external APIs.
- Cover edge cases: NaNs/gaps in data, feature windows, execution costs/slippage, websocket disconnects.
- Run locally: `pytest -q` or `make test`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat(backtest): add execution cost modelling`, `test: cover streaming safeguards`).
- PRs include: clear description, linked issues, CLI/API changes, before/after metrics or plots when relevant, and tests for new behavior.
- Ensure `pre-commit run --all-files` (format, lint, test) passes before requesting review.

## Security & Configuration Tips
- Keep secrets in env vars (e.g., `OPENAI_API_KEY`); never commit `.env`.
- Centralize knobs in YAML under `config/`; validate changes with `verify_config_loading.py`.
- Write outputs to `data/`, `models/`, `reports/` (these paths are gitignored by default).

## Evaluation & Splitting (Important)
- Train/test splits are time-aware:
  - Prefer `data.test_start` and optional `data.test_end` in `config/model_config.yaml`.
  - If unset, the last `training.test_size` fraction is used as test chronologically (no shuffle).
- Hyperparameter tuning uses `TimeSeriesSplit(n_splits=training.cv_folds)` to avoid leakage.
- Ensure any configured test window falls within `data.start_date`/`data.end_date`.
