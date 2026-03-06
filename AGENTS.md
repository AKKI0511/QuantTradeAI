# Agent Guidelines for QuantTradeAI

## Mission and Product Principles
- Build for the happy path: production-ready for most users, avoid niche edge cases.
- Prioritize reliability, clear defaults, and end-to-end usability.
- Keep features simple and explainable; prefer small, complete workflows.

## Source of Truth
- Public roadmap lives in `roadmap.md` and should stay current.
- If roadmap direction conflicts with legacy docs or old config patterns, follow the roadmap.

## Project Structure
- Core package: `quanttradeai/` (CLI entry points in `quanttradeai/main.py` and `quanttradeai/cli.py`).
- Modules: `data/`, `features/`, `models/`, `backtest/`, `trading/`, `streaming/`, `utils/`.
- Configs: current codebase still uses multiple YAML files under `config/`; target UX in the roadmap is a single `config/project.yaml` entrypoint.
- Tests mirror modules under `tests/`.
- Outputs: `data/`, `models/`, `reports/` (gitignored).

## Build, Test, and Dev Commands
- Install dev deps: `poetry install --with dev`
- Format: `make format`
- Lint: `make lint`
- Tests: `make test` or `poetry run pytest`
- Pipeline: `poetry run quanttradeai train -c config/model_config.yaml`
- CLI help: `poetry run quanttradeai --help`

## Engineering Expectations
- Preserve time-aware evaluation and avoid data leakage.
- Keep training and serving feature pipelines consistent.
- Favor config-driven behavior; update `quanttradeai/utils/config_schemas.py` when adding new knobs.
- AI features must be optional, explainable, and safe by default.
- Avoid heavy infrastructure; keep deployment and scaling simple.
- Use clear logging and human-readable summaries.

## Testing Guidelines
- Use pytest with deterministic tests; mock external APIs and streaming.
- Cover edge cases: missing data, NaNs, time splits, execution costs, disconnects.
- Add tests for AI features that validate behavior without real API calls.

## Documentation and UX
- Update `roadmap.md` when scope or stage changes.
- Update `README.md` and `docs/` when CLI or config behavior changes.
- Keep docs focused on concepts and user workflows, not internal details.
- Keep the CLI surface small and intuitive; prefer clear flags over adding niche commands.

## Security and Configuration
- Keep secrets in env vars; never commit `.env`.
- Validate configs with `poetry run quanttradeai validate-config` when changing YAML.
- Write outputs to `data/`, `models/`, and `reports/`.
