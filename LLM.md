# QuantTradeAI Agent Guide

QuantTradeAI is built around one canonical project file: `config/project.yaml`.
AI coding agents should treat that file as the source of truth for research,
agent runs, promotion, and deployment.

## Primary Workflow

Use the small CLI surface first:

```bash
uvx quanttradeai init research-lab --template research
cd research-lab
uv sync
uv run quanttradeai doctor
uv run quanttradeai validate -c config/project.yaml
uv run quanttradeai research run -c config/project.yaml
uv run quanttradeai research run -c config/project.yaml --sweep <sweep_name> --max-concurrency 4
uv run quanttradeai runs list --scoreboard --sort-by net_sharpe
uv run quanttradeai promote --run research/<run_id> -c config/project.yaml
```

Agent projects follow the same pattern:

```bash
uvx quanttradeai init llm-lab --template llm-agent
cd llm-lab
uv sync
uv run quanttradeai doctor
uv run quanttradeai validate -c config/project.yaml
uv run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
uv run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
uv run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper
uv run quanttradeai promote --run agent/paper/<run_id> -c config/project.yaml --to live --acknowledge-live breakout_gpt
```

Deployment stays under the same command:

```bash
uv run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target local
uv run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
uv run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target render -o deployments/breakout-render
```

## Product Objects

- `project`: name, profile, and environment metadata.
- `data`: symbols, time windows, historical data, and streaming settings.
- `features`: reusable feature definitions shared by research and agents.
- `research`: labels, model training, evaluation, backtest, and promotion rules.
- `agents`: first-class `rule`, `model`, `llm`, and `hybrid` trading agents.
- `risk` and `position_manager`: live safety and runtime controls.
- `deployment`: local, Docker Compose, or Render bundle metadata.
- `runs`: persisted artifacts for research, backtest, paper, live, batch, and sweep runs.

## Agent Design Rules

- Prefer YAML changes in `config/project.yaml` over new config files.
- Keep the CLI path small: `init`, `validate`, `research run`, `agent run`, `runs list`, `promote`, and `deploy`.
- Preserve time-aware evaluation and avoid data leakage.
- Keep training and serving feature definitions aligned.
- Make LLM behavior auditable through prompt files, context blocks, decisions, executions, and prompt samples.
- Do not remove working legacy utility commands unless a canonical replacement already exists.

## Deployment Notes

`deploy --target render` generates a Render Background Worker bundle with:

- `render.yaml`
- `Dockerfile`
- `.env.example`
- `resolved_project_config.yaml`
- `deployment_manifest.json`
- `assets/` for selected-agent prompts, notes, and model artifacts

Render bundles use `sync: false` secret placeholders and a persistent `/app/runs` disk. For Git-backed Render deploys, generate the bundle into a tracked directory such as `deployments/<agent>-render` or force-add the default `reports/deployments/...` output.

## Development Checks

Before handing off a code change, run the narrow tests for the touched area, then the broader suite if practical:

```bash
poetry check
poetry run pytest tests/integration/test_deploy_cli.py -q
poetry run pytest tests/integration/test_cli_smoke.py tests/test_project_config_cli.py -q
```

Use `make format`, `make lint`, and `make test` for full local verification.
