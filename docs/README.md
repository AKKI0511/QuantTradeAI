# QuantTradeAI Documentation

This documentation set covers the implemented CLI workflows, configuration formats, and public APIs in QuantTradeAI.

## Start Here

- [Getting Started](getting-started.md) for the canonical `project.yaml` workflow and the legacy multi-file workflow
- [Configuration Guide](configuration.md) for supported config files and validation commands
- [Quick Reference](quick-reference.md) for CLI examples and common API patterns
- [LLM Sentiment Analysis](llm-sentiment.md) for optional sentiment features
- [API Documentation](api/) for module-level reference material
- [Main README](../README.md) for repository-level overview

## Quick Start

### Canonical project workflow

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
```

Use this path when you want the CLI to generate and validate `config/project.yaml` for the Stage 1 research workflow.

### Legacy multi-file workflow

```bash
poetry run quanttradeai fetch-data -c config/model_config.yaml
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -c config/model_config.yaml -m models/experiments/<timestamp>/<SYMBOL>
```

Use this path when you want to run the existing `model_config.yaml` and `features_config.yaml` pipeline directly.

## What Lives Where

- `getting-started.md` explains the end-to-end onboarding flow.
- `configuration.md` documents config files, templates, and validation.
- `quick-reference.md` summarizes common commands and code snippets.
- `api/README.md` indexes the API reference by subsystem.

## Core Runtime Artifacts

- `runs/<timestamp>/` contains `research run` summaries and resolved runtime configs.
- `models/experiments/<timestamp>/` contains training outputs, validation reports, and per-symbol saved models.
- `reports/backtests/<timestamp>/` contains saved-model backtest outputs.
