# Run Inspection

`quanttradeai runs list` discovers local run records under `runs/`, filters them, optionally attaches metrics, and renders a table, JSON payload, or explicit run comparison.

## When To Use

Use it:

- after research or agent experiments
- before promoting a candidate
- to rank runs with scoreboard metrics
- to compare 2-4 compatible candidates
- when a coding agent needs normalized run metadata

## Syntax

```bash
quanttradeai runs list [OPTIONS]
```

```bash
quanttradeai runs list
quanttradeai runs list --type research
quanttradeai runs list --type agent --mode backtest --scoreboard
quanttradeai runs list --sort-by net_sharpe --scoreboard
quanttradeai runs list --json
quanttradeai runs list --compare agent/backtest/run_a --compare agent/backtest/run_b
```

## Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `--type TEXT` | `all` | No | Run type filter. Allowed values: `all`, `research`, `agent`, `batch`. |
| `--mode TEXT` | `all` | No | Run mode filter. Allowed values: `all`, `research`, `backtest`, `paper`, `live`. |
| `--status TEXT` | `all` | No | Run status filter. Allowed values: `all`, `success`, `failed`. |
| `--limit INTEGER` | `20` | No | Maximum runs to show. Must be at least `1`. |
| `--scoreboard` | `false` | No | Render a metrics-aware table using `metrics.json` artifacts. |
| `--sort-by TEXT` | `started_at` | No | Sort field. Allowed values are listed below. |
| `--ascending` | `false` | No | Sort ascending instead of using the default direction for the selected field. |
| `--json` | `false` | No | Emit normalized run records or comparison payload as JSON. |
| `--compare TEXT` | `None` | No | Explicit run id to compare. Repeat 2-4 times. |

Allowed `--sort-by` values:

```text
started_at
name
status
accuracy
f1
net_sharpe
net_pnl
total_pnl
execution_count
decision_count
```

Sorting by a metric field attaches scoreboard data even if `--scoreboard` is not set. Missing metric values sort last.

## Reads

The command recursively discovers:

```text
runs/**/summary.json
```

When scoreboard data is needed, it also reads `metrics.json` using either the `summary.json` artifact path or the default file under the run directory.

Comparison mode also reads:

- `summary.json`
- `metrics.json`
- `resolved_project_config.yaml`

## Normal Listing

Without `--scoreboard`, the CLI renders a compact run table:

```text
RUN_ID  TYPE  MODE  STATUS  NAME  STARTED_AT  SYMBOLS
```

Example:

```bash
quanttradeai runs list --type agent --mode paper --status success
```

## Scoreboard Listing

With `--scoreboard`, the table adapts to the run family:

- research runs show `ACC`, `F1`, `NET_SHARPE`, and `NET_PNL`
- agent backtests show `NET_SHARPE`, `NET_PNL`, `NET_MDD`, and `DECISIONS`
- paper/live agent runs show `TOTAL_PNL`, `PORTFOLIO`, `EXEC`, `DECISIONS`, and `RISK`
- mixed runs show shared primary/PNL/Sharpe/execution columns where available

Example:

```bash
quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe
```

## JSON Output

Use `--json` when another program or coding agent needs structured output:

```bash
quanttradeai runs list --type research --json
```

In normal mode, JSON output is a list of normalized run records.

## Comparing Runs

Comparison mode requires 2-4 explicit `--compare` values. All compared runs must be from the same run family:

- `research`
- `agent/backtest`
- `agent/paper`
- `agent/live`

Example:

```bash
quanttradeai runs list \
  --compare agent/backtest/20260525_120000_rsi_reversion \
  --compare agent/backtest/20260525_130000_rsi_reversion
```

Comparison mode does not support these listing flags at the same time:

```text
--type
--mode
--status
--limit
--scoreboard
```

It does support `--sort-by`, `--ascending`, and `--json`.

The rendered comparison includes:

- scoreboard metrics
- config differences extracted from each run's resolved config
- artifact paths
- warnings

## Expected Outcome

The command either prints a table, prints JSON, prints a comparison report, or prints:

```text
No runs found.
```

Invalid filters, unsupported sort fields, missing run ids, duplicate comparison ids, and incompatible comparison families fail with a clear error.

## Common Mistakes

| Mistake | Why it matters |
|---|---|
| Comparing incompatible runs | Comparison only supports one run family at a time. Rank mixed runs with `--scoreboard` first. |
| Using scoreboard without checking child artifacts | Scoreboard values are summaries; inspect each child `summary.json` and `metrics.json` before promotion. |
| Sorting by metrics that do not exist for a run type | Missing values sort last, which can hide otherwise relevant runs. |
| Combining `--compare` with listing filters | Compare mode rejects filters such as `--type`, `--mode`, and `--limit`. |
| Passing only one `--compare` value | Compare mode requires at least two and at most four run ids. |

## Related Docs

- [`docs/artifacts.md`](../artifacts.md)
- [`docs/config/`](../config/)
