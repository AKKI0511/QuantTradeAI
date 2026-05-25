# QuantTradeAI Artifacts

QuantTradeAI writes machine-readable artifacts for every meaningful run. Treat them as the source of truth.

## Single Run Reading Order

1. `summary.json`
2. `summary.json.run_result`
3. `metrics.json`
4. `resolved_project_config.yaml`

Use `summary.json.run_result` for the compact result contract. Use `metrics.json` for detailed values. Use `resolved_project_config.yaml` to verify the exact config that ran.

## Batch And Sweep Reading Order

1. batch `summary.json`
2. batch `scoreboard.json`
3. batch `results.json`
4. top child run `summary.json`
5. top child run `metrics.json`
6. top child run `resolved_project_config.yaml`

Use `scoreboard.json` to rank candidates. Use `results.json` to map candidates to run IDs, run directories, parameters, failures, and artifacts.

## Important Files

* `summary.json`: run metadata, status, artifacts, and `run_result`
* `summary.json.run_result`: compact machine-readable result summary
* `metrics.json`: detailed metrics
* `scoreboard.json`: ranked batch/sweep records
* `results.json`: batch/sweep child run details
* `resolved_project_config.yaml`: exact config used at runtime
* `decisions.jsonl`: agent decisions
* `executions.jsonl`: simulated, paper, or live executions
* `prompt_samples.json`: sampled prompt payloads for LLM/hybrid agents
* `replay_manifest.json`: replay-backed paper run details

## Interpretation Rules

* Do not declare a winner from terminal output alone.
* Do not declare a winner from one metric alone if other risk metrics are poor.
* Check status and failures before trusting rankings.
* Check the resolved config before comparing runs.
* Prefer concise JSON summaries over large logs.
* Read JSONL files only when decision-level or execution-level explanation is needed.
* For LLM/hybrid agents, inspect prompt samples only when debugging prompt/context behavior.

## Default Ranking Guidance

For backtests:

* Start with `net_sharpe`
* Check `net_pnl`
* Check `net_mdd` or drawdown fields
* Check trade/execution count if available
* Watch for suspiciously sparse or overfit results

For paper/live:

* Start with `total_pnl`
* Check portfolio value
* Check risk status
* Check execution count
* Check failures and warnings
