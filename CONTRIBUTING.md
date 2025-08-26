# Contributing to QuantTradeAI

Thanks for your interest in contributing! This guide covers setup, workflow, and review expectations.

## Setup
- Prerequisites: Python 3.11+, Poetry.
- Install deps: `poetry install --with dev`
- Install hooks: `poetry run pre-commit install`

## Development Workflow
1. Create a feature branch: `git checkout -b feat/<scope>-<short-desc>`
2. Write code and tests under `quanttradeai/` and `tests/` (mirror structure).
3. Run quality gates locally:
   - Format: `make format`
   - Lint: `make lint`
   - Test: `make test`
4. Optional: quick sanity run
   - CLI help: `poetry run quanttradeai --help`
   - Pipeline: `poetry run quanttradeai train -c config/model_config.yaml`

## Style & Conventions
- Formatting: Black (line length 88); lint: Flake8 (see `.flake8`).
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- Commits: Conventional Commits, e.g. `feat(backtest): add execution cost modelling`, `test: cover streaming safeguards`.

## Pull Requests
- Template: PRs auto-use `.github/pull_request_template.md`.
- Include: clear summary, list of changes, tests, and any screenshots/metrics when relevant.
- Ensure `pre-commit run --all-files` passes before requesting review.

## Configuration & Secrets
- Keep credentials in env vars (e.g., `OPENAI_API_KEY`); do not commit `.env`.
- Use YAML in `config/` to change behavior; validate with `verify_config_loading.py` if modified.

## Getting Help
- Check `README.md` and `docs/` for usage details.
- Open a GitHub Discussion/Issue with a minimal repro if blocked.

