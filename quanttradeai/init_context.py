# flake8: noqa: E501
"""Generated workspace guidance files for `quanttradeai init`."""

AGENTS_MD_CONTENT = """# QuantTradeAI Workspace

This is a QuantTradeAI workspace for running strategy research and agent workflows through the `quanttradeai` CLI and YAML.

## Primary Interface

- Treat `config/project.yaml` as the canonical project file.
- Prefer editing YAML and running QuantTradeAI CLI commands over writing custom Python scripts.
- Run `quanttradeai validate -c config/project.yaml` after every YAML change.
- Do not edit QuantTradeAI package or framework internals unless the human explicitly asks to modify QuantTradeAI itself.

## Default Workflow

- Default to `backtest` and replay-backed `paper` workflows.
- Never run `live` mode, broker-backed execution, or live deployment unless the human explicitly asks for it.
- Prefer sweeps and batch runs for strategy research and optimization.
- Use `quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4` for batch backtests.
- Use `quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency 4` for configured sweeps.
- Promote only successful runs after reviewing their artifacts.

## Artifact Review

- Do not rely only on terminal output when making recommendations.
- Inspect machine-readable artifacts before summarizing results.
- Read `summary.json.run_result` first for a single run.
- Read `scoreboard.json` for rankings across batch or sweep runs.
- Read `metrics.json` and `results.json` to validate the ranking details.
- Compare top runs before recommending a winner.
- Explain recommendations using metrics, assumptions, and tradeoffs from the artifacts.

## Useful Commands

```bash
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai runs list --scoreboard --sort-by net_sharpe
quanttradeai promote --run agent/backtest/<winning_run_id> -c config/project.yaml
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode paper
```
"""


CLAUDE_MD_CONTENT = """@AGENTS.md

## Claude Code

- Follow `AGENTS.md` as the main QuantTradeAI operating guide.
- Use the project skill at `.claude/skills/quanttradeai-research/SKILL.md` when the user asks to research, test, compare, optimize, paper trade, or deploy strategies.
- Live trading requires explicit user approval.
"""


CLAUDE_SKILL_MD_CONTENT = """---
name: quanttradeai-research
description: Use QuantTradeAI CLI and config/project.yaml to research, backtest, sweep, compare, promote, paper trade, or prepare deployment for trading strategies.
---

# QuantTradeAI Research

## When To Use

Use this skill when the human asks to research, test, compare, optimize, sweep, promote, paper trade, or prepare deployment for trading strategies in this workspace.

## Primary Interface

- Use the `quanttradeai` CLI and `config/project.yaml`.
- Prefer YAML edits and CLI runs over custom scripts.
- Validate after every YAML change.

## Default Experiment Loop

1. Read `config/project.yaml`.
2. Edit symbols, features, agents, sweeps, or risk settings in YAML.
3. Run `quanttradeai validate -c config/project.yaml`.
4. Run backtests or sweeps.
5. Read JSON artifacts before judging results.
6. Compare top runs and recommend a winner only with artifact evidence.
7. Promote only successful runs.

## Command Patterns

See `workflow.md` for detailed command sequences.

## Artifact Reading Order

See `artifacts.md`. Start with `summary.json.run_result` for single runs and `scoreboard.json` for batch or sweep rankings.

## Safety Rules

See `safety.md`. Backtest is the default. Paper mode is allowed unless the user restricts it. Live trading and broker-backed execution require explicit human approval.

## Responding To The Human

- State what was changed in YAML.
- State which commands ran and whether they passed.
- Summarize the winning run and closest alternatives using artifact metrics.
- Call out missing data, failed runs, or validation issues plainly.
"""


CLAUDE_WORKFLOW_MD_CONTENT = """# QuantTradeAI Workflow

## Strategy Research

Start with `config/project.yaml`. Update symbols, date windows, features, agents, sweeps, and risk settings there. Prefer existing template structure and explainable rules before adding complexity.

```bash
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
```

## YAML Editing

After every YAML edit, validate before running experiments.

```bash
quanttradeai validate -c config/project.yaml
```

If validation fails, fix the YAML first. Do not continue to promotion or deployment with an invalid project file.

## Backtest

Run all configured agents in backtest mode when comparing strategies.

```bash
quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
```

Run one agent when isolating a specific candidate.

```bash
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode backtest
```

## Sweep

Use configured sweeps for parameter research.

```bash
quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency 4
```

## Scoreboard

Rank runs with the scoreboard before choosing a winner.

```bash
quanttradeai runs list --scoreboard --sort-by net_sharpe
```

Use the metric that matches the user's goal when they specify one.

## Compare

Compare top runs directly before making a recommendation.

```bash
quanttradeai runs list --compare agent/backtest/<run_a> --compare agent/backtest/<run_b> --sort-by net_sharpe
```

## Promotion

Promote only successful runs after reviewing artifacts.

```bash
quanttradeai promote --run agent/backtest/<winning_run_id> -c config/project.yaml
```

For research model promotion, use the research run id requested by the CLI output or artifacts.

## Paper Mode

Paper mode is the next step after a successful backtest and promotion.

```bash
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode paper
```

Prefer replay-backed paper mode unless the human explicitly asks for broker-backed behavior.

## Deployment Preparation

If the user asks to deploy, clarify whether they mean a deployment bundle or actual live trading. Preparing a deployment bundle is not the same as starting live trading. Keep live mode gated on explicit approval.
"""


CLAUDE_ARTIFACTS_MD_CONTENT = """# QuantTradeAI Artifacts

Use JSON artifacts as the source of truth. Do not rely only on terminal output.

## Reading Order

1. `summary.json`
2. `summary.json.run_result`
3. `metrics.json`
4. `scoreboard.json`
5. `results.json`
6. `decisions.jsonl`
7. `executions.jsonl`
8. `resolved_project_config.yaml`

## Single Runs

For single runs, read `summary.json.run_result` first, then `metrics.json`. Use `resolved_project_config.yaml` to confirm what settings actually ran.

## Batch And Sweeps

For batch or sweep results, read `scoreboard.json` for rankings and `results.json` for run details. Compare the top candidates before recommending a winner.

## LLM And Hybrid Runs

Inspect prompt samples only when needed to explain a decision, debug behavior, or verify prompt wiring. Avoid reading huge files unless necessary.

## Large Files

Prefer compact summaries and JSON records. Avoid loading full logs, large JSONL files, or generated data unless a specific question requires it.
"""


CLAUDE_SAFETY_MD_CONTENT = """# QuantTradeAI Safety

- Backtest mode is safe by default.
- Replay-backed paper mode is allowed unless the user restricts it.
- Live mode requires explicit human approval.
- Broker-backed execution requires explicit human approval.
- Never promote to live unless the user asks.
- Never run live commands just because a previous step succeeded.
- If the user asks for "deploy", clarify whether they mean generate a deployment bundle or actually live trade.
- Prefer simulated and replay workflows first.
"""


INIT_CONTEXT_FILES = {
    "AGENTS.md": AGENTS_MD_CONTENT,
    "CLAUDE.md": CLAUDE_MD_CONTENT,
    ".claude/skills/quanttradeai-research/SKILL.md": CLAUDE_SKILL_MD_CONTENT,
    ".claude/skills/quanttradeai-research/workflow.md": CLAUDE_WORKFLOW_MD_CONTENT,
    ".claude/skills/quanttradeai-research/artifacts.md": CLAUDE_ARTIFACTS_MD_CONTENT,
    ".claude/skills/quanttradeai-research/safety.md": CLAUDE_SAFETY_MD_CONTENT,
}
