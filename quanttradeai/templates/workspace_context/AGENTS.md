# QuantTradeAI Project Workspace

This is a disposable QuantTradeAI project workspace. Keep project-specific state here and use the globally installed QuantTradeAI plugin for reusable framework workflows.

## Local Project Files

- `config/project.yaml` is the canonical project config.
- `pyproject.toml` pins QuantTradeAI as a local uv dependency.
- `.env.example` documents optional local environment variables. Keep real secrets in `.env` or exported environment variables.
- `runs/`, `reports/`, `data/`, and `models/` are local outputs.

## Common Commands

```bash
uv sync
uv run quanttradeai validate -c config/project.yaml
uv run quanttradeai agent run --all -c config/project.yaml --mode backtest
```

Use the QuantTradeAI plugin skills for research loops, run analysis, promotion, and deployment guidance. Do not copy or regenerate plugin skills inside this workspace.
