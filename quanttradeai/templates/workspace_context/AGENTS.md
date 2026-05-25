# QuantTradeAI Workspace

This is a QuantTradeAI workspace for agent-driven quant research.

QuantTradeAI is operated through `config/project.yaml` and the `quanttradeai` CLI. A human gives the research objective. You, the coding agent, should translate that objective into YAML changes, run experiments, inspect machine-readable artifacts, and report evidence-backed results.

## Primary Interface

- Treat `config/project.yaml` as the canonical project file.
- Prefer editing YAML and running `quanttradeai` CLI commands over writing custom research scripts.
- Run `quanttradeai validate -c config/project.yaml` after every YAML change.
- Do not edit QuantTradeAI package/framework internals unless the human explicitly asks to modify QuantTradeAI itself.
- Keep experiments reproducible. Prefer checked-in YAML changes over ad hoc terminal-only state.

## Default Research Loop

1. Read `config/project.yaml`.
2. Identify the current symbols, dates, features, agents, sweeps, risk settings, and deployment mode.
3. Modify YAML to express the user's requested strategy research.
4. Validate the project.
5. Run backtests, batches, or sweeps.
6. Read JSON artifacts before judging results.
7. Compare top candidates.
8. Recommend the next experiment, winner, or promotion step with metric evidence.

## Common Commands

```bash
quanttradeai validate -c config/project.yaml
quanttradeai agent run --all -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai agent run --sweep <sweep_name> -c config/project.yaml --mode backtest --max-concurrency 4
quanttradeai runs list --scoreboard --sort-by net_sharpe
quanttradeai runs list --compare agent/backtest/<run_a> --compare agent/backtest/<run_b> --sort-by net_sharpe
quanttradeai promote --run agent/backtest/<winning_run_id> -c config/project.yaml
quanttradeai agent run --agent <agent_name> -c config/project.yaml --mode paper
```

Use `research run` when the project is training/promoting model artifacts. Use `agent run` when comparing deployable rule/model/LLM/hybrid agents.

## Artifact Reading Order

Do not rely only on terminal output.

For a single run:

1. `summary.json`
2. `summary.json.run_result`
3. `metrics.json`
4. `resolved_project_config.yaml`

For batch or sweep runs:

1. batch `summary.json`
2. batch `scoreboard.json`
3. batch `results.json`
4. top child run `summary.json`
5. top child run `metrics.json`

Use `resolved_project_config.yaml` to confirm what actually ran. Use `decisions.jsonl`, `executions.jsonl`, and prompt samples only when needed to explain behavior or debug a run.

## Strategy Research Defaults

* Prefer `strategy-lab` style workflows for early exploration.
* Prefer backtest sweeps before paper runs.
* Prefer simple explainable strategy variants before adding complexity.
* Use `net_sharpe` for default backtest ranking unless the human specifies another objective.
* Consider drawdown, total PnL, execution count, failure count, and overfitting risk before declaring a winner.
* If the requested strategy cannot be represented cleanly in the current YAML, explain the gap before writing custom code.

## Safety Rules

* Backtest mode is safe by default.
* Replay-backed paper mode is allowed unless the human restricts it.
* Live mode requires explicit human approval.
* Broker-backed execution requires explicit human approval.
* Never run `--mode live` just because a backtest or paper run succeeded.
* Never promote to live unless the human explicitly asks.
* If the human says "deploy", clarify whether they mean generating a deployment bundle or starting live trading.

## Response Expectations

When reporting back:

* State what YAML/config changed.
* State which commands ran and whether they passed.
* Summarize the best run and closest alternatives using artifact metrics.
* Mention failed runs or validation warnings.
* Recommend the next concrete step.
* Be explicit about assumptions and limitations.

## More Detailed Skill Guidance

Claude Code users can use the project skill at:

`.claude/skills/quanttradeai-research/SKILL.md`

Other coding agents should follow this `AGENTS.md` file and may read the skill files as reference documentation.
