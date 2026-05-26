# Sweeps

Sweeps run a Cartesian product of scalar parameter values without mutating the source YAML file. QuantTradeAI supports research sweeps and agent backtest sweeps.

## Used By

| Command | Sweep kind |
|---|---|
| `quanttradeai validate` | Expands and validates all configured sweeps. |
| `quanttradeai research run --sweep <name>` | Runs `kind: research_run` sweeps. |
| `quanttradeai agent run --sweep <name> --mode backtest` | Runs `kind: agent_backtest` sweeps. |
| `quanttradeai promote --run agent/backtest/<run>` | Materializes a successful sweep-generated backtest child into the base agent for paper mode. |

Agent sweeps are backtest-only in the current CLI. `agent run --sweep` rejects `--mode paper` and `--mode live`.

## Supported Fields

```yaml
sweeps:
  - name: rsi_threshold_grid
    kind: agent_backtest
    agent: rsi_reversion
    parameters:
      - path: rule.buy_below
        values: [25, 30]
      - path: rule.sell_above
        values: [70, 75]
```

| Field | Required | Notes |
|---|---:|---|
| `name` | Yes | Must be non-empty and unique. |
| `kind` | Yes | `agent_backtest` or `research_run`. |
| `agent` | For `agent_backtest` | Must reference an existing `agents[].name`. |
| `parameters` | Yes | At least one parameter. |
| `parameters[].path` | Yes | Dot-separated path to an existing scalar leaf. |
| `parameters[].values` | Yes | Non-empty list of scalar values: string, integer, float, boolean, or null. |

Sweep paths may not modify `name`, `kind`, or `mode`.

## Agent Backtest Sweeps

Agent sweep paths are relative to the selected agent config.

```yaml
sweeps:
  - name: sma_risk_grid
    kind: agent_backtest
    agent: sma_trend
    parameters:
      - path: risk.max_position_pct
        values: [0.03, 0.05, 0.07]
```

Examples of supported agent paths:

| Path | Meaning |
|---|---|
| `rule.buy_below` | RSI threshold parameter. |
| `rule.sell_above` | RSI threshold parameter. |
| `risk.max_position_pct` | Agent position sizing parameter. |
| `llm.prompt_file` | Prompt path, if every value points to a valid prompt file. |

The path must already exist in the base agent and resolve to a scalar leaf.

## Research Sweeps

Research sweep paths are relative to the project config.

```yaml
sweeps:
  - name: rsi_research_grid
    kind: research_run
    parameters:
      - path: research.labels.horizon
        values: [3, 5]
      - path: features.rsi_14.params.period
        values: [7, 14]
      - path: research.backtest.costs.bps
        values: [1, 5]
```

Supported research sweep paths:

| Path family | Supported leaves |
|---|---|
| `data.<leaf>` | `timeframe`, `start_date`, `end_date`, `test_start`, `test_end`, `cache_path`, `cache_dir`, `cache_expiration_days`, `use_cache`, `refresh`, `max_workers` |
| `research.labels.<leaf>` | Existing scalar label leaves such as `horizon`, `buy_threshold`, `sell_threshold`. |
| `research.backtest.costs.<leaf>` | Existing scalar cost leaves such as `enabled` and `bps`. |
| `research.model.tuning.<leaf>` | `enabled`, `trials` |
| `research.evaluation.use_configured_test_window` | Boolean values. |
| `features.<feature_name>.params.<param>` | Existing scalar feature parameter leaves. |

`data.symbols`, `research.model.kind`, and `research.model.family` are not supported sweep paths.

## Variant Creation

QuantTradeAI expands parameter values as a Cartesian product. For example, two paths with two values each produce four child runs.

Variant names are deterministic and include the base project or agent name, sweep name, leaf names, and scalar values. Child project configs are materialized for run execution; the source YAML file is not edited.

## Expected Outcomes

Research sweep batches write under `runs/research/batches/...`:

- `summary.json`
- `results.json`
- `scoreboard.json`
- `resolved_project_config.yaml`
- child `runs/research/...` directories
- child research runtime configs and experiment artifacts

Agent sweep batches write under `runs/agent/batches/...`:

- `summary.json`
- `results.json`
- `scoreboard.json`
- `resolved_project_config.yaml`
- `variants/<variant>/project.yaml`
- child `runs/agent/backtest/...` directories
- child decisions, metrics, equity curves, and summaries

For sweep-generated agent backtest children, batch results include a promotion command. Promoting the winning child materializes that child's scalar parameters back into the base agent, sets the base agent to `mode: paper`, and sets `deployment.mode: paper`.

## Common Mistakes

| Mistake | Result |
|---|---|
| Running `agent run --sweep` with `--mode paper` or `--mode live` | The CLI rejects the command. |
| Using a path that does not already exist | Validation fails. |
| Pointing to a list or mapping instead of a scalar leaf | Validation fails. |
| Expecting the checked-in YAML to mutate during the sweep | Sweeps write child configs and run artifacts, not source config edits. |
| Promoting a sweep child without inspecting `scoreboard.json` and child artifacts | You may materialize the wrong parameter combination. |
| Trying to promote a sweep backtest child directly to live | Promotion requires materializing to paper first, then running/promoting a successful paper run. |

