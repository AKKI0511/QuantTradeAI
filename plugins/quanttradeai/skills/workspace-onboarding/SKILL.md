---
name: workspace-onboarding
description: Use when a user wants to start with QuantTradeAI from outside an initialized workspace, create a new QuantTradeAI lab, validate the generated project YAML, or asks for a natural-language quant research workflow that should begin by creating a workspace. Handles the agent-first onboarding path with `uvx quanttradeai init`, `uv sync`, and handoff into strategy, model, or run-analysis workflows.
---

# Workspace Onboarding

Use this skill when the user asks to start QuantTradeAI from a normal folder rather than from an existing QuantTradeAI workspace.

The goal is to create a clean workspace, verify it, then continue with the appropriate research workflow. Do not clone the QuantTradeAI source repository for normal users. Do not create one-off backtest scripts.

## First Check

Check whether the current directory is already a QuantTradeAI workspace:

- `config/project.yaml`
- `pyproject.toml`
- `.quanttradeai/workspace.yaml`

If those exist, do not run `init` again unless the user explicitly asks to regenerate or overwrite. Continue with `uv sync`, `uv run quanttradeai doctor`, and the relevant research commands.

## Create The Workspace

If the user gave a workspace name, use it. If not, choose a short name from the request, such as `vibe-lab`, `strategy-lab`, or `research-lab`.

Default template:

```bash
uvx quanttradeai init <workspace>
cd <workspace>
```

Template examples:

```bash
uvx quanttradeai init <workspace> --template strategy-lab
uvx quanttradeai init <workspace> --template research
uvx quanttradeai init <workspace> --template rule-agent
uvx quanttradeai init <workspace> --template model-agent
uvx quanttradeai init <workspace> --template llm-agent
uvx quanttradeai init <workspace> --template hybrid
```

Use `strategy-lab` for general strategy comparison, sweeps, RSI, SMA, or "vibe quant research" prompts. Use `research` when the request is clearly about training historical ML models. Use an agent template only when the user specifically asks for that agent type.

If `uvx quanttradeai ...` fails because the package is not available from the configured package registry, report that package installation is blocked. Do not switch to `pip install` or clone the source unless the user asks for contributor setup.

## Set Up uv

Inside the workspace:

```bash
uv sync
uv run quanttradeai doctor
```

If `doctor` reports warnings, fix what is actionable before running experiments. Common fixes:

- Run `uv sync` if `.venv` is missing.
- Use the Python version from `.python-version` when possible.
- Keep the generated `quanttradeai==<version>` pin unless the user intentionally changes package versions.

## Validate The Project

Before research or backtests:

```bash
uv run quanttradeai validate -c config/project.yaml
```

If validation fails, edit `config/project.yaml` and validate again. Keep edits small and tied to the user's research request.

## Continue Into Research

For broad strategy prompts, use the strategy experiment workflow:

```bash
uv run quanttradeai agent run --all -c config/project.yaml --mode backtest
uv run quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency <n>
uv run quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe
```

For model prompts, use the model research workflow:

```bash
uv run quanttradeai research run -c config/project.yaml
uv run quanttradeai research run -c config/project.yaml --sweep <sweep_name> --max-concurrency <n>
uv run quanttradeai runs list --type research --scoreboard
```

After any run, inspect durable artifacts under `runs/` before recommending a winner. Terminal output alone is not enough.

## Reporting Standards

When reporting back, include:

- Workspace path created or reused.
- Template used.
- Setup commands run.
- `doctor` and validation status.
- Config changes made.
- Experiments run.
- Artifact paths inspected.
- Best candidate or `no clear winner`.
- Caveats about dates, costs, risk, sparse trades, failures, and overfitting.

Stay in backtest/research mode unless the user explicitly asks for paper, deployment, or live trading.
