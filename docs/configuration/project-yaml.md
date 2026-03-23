# Project Config (`project.yaml`)

`config/project.yaml` is the canonical config for the current happy path.

It is the file used by:

- `poetry run quanttradeai init --template ...`
- `poetry run quanttradeai validate -c config/project.yaml`
- `poetry run quanttradeai research run -c config/project.yaml`
- `poetry run quanttradeai agent run --agent ... -c config/project.yaml --mode backtest`

It is **not** the only config file in the product. Live trading still depends on the runtime YAMLs documented in [Runtime and Live Trading Configs](live-runtime-files.md).

## Happy-Path Flow

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

### Agent Backtest

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
```

## What Belongs in `project.yaml`

| Section | Required | Used at runtime today | Notes |
| --- | --- | --- | --- |
| `project` | Yes | Partly | Name is used in run folders and summaries |
| `profiles` | Yes | Minimal | Required and validated, but mostly metadata today |
| `data` | Yes | Yes | Drives date ranges, symbols, cache settings, and test windows |
| `features` | Yes | Yes | Compiled into the runtime feature config |
| `research` | Yes | Yes | Controls labels, tuning, split behavior, and research backtest costs |
| `agents` | Yes | Yes | Used by `quanttradeai agent run` |
| `deployment` | Yes | Minimal | Required and validated, but mostly metadata today |

## A Practical Starting Example

```yaml
project:
  name: research_lab
  profile: research

profiles:
  research:
    mode: research
  paper:
    mode: paper
  live:
    mode: live

data:
  symbols: ["AAPL", "MSFT"]
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  timeframe: "1d"
  test_start: "2024-09-01"
  test_end: "2024-12-31"
  cache_dir: "data/raw"
  cache_path: "data/raw"
  cache_expiration_days: 7
  use_cache: true
  refresh: false
  max_workers: 1

features:
  definitions:
    - name: rsi_14
      type: technical
      params:
        period: 14
    - name: mean_reversion_10
      type: custom
      params:
        kind: mean_reversion
        lookback: [10]

research:
  enabled: true
  labels:
    type: forward_return
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: classifier
    family: voting
    tuning:
      enabled: true
      trials: 50
  evaluation:
    split: time_aware
    use_configured_test_window: true
  backtest:
    costs:
      enabled: true
      bps: 5

agents: []

deployment:
  target: docker-compose
  mode: paper
```

## Section by Section

### `project`

| Field | Required | What it does today |
| --- | --- | --- |
| `name` | Yes | Used in summaries and research run directory names |
| `profile` | Yes | Validated and surfaced in summaries; not a major runtime switch yet |

### `profiles`

This section is required and validated, but it is mostly descriptive in the current implementation.

```yaml
profiles:
  research:
    mode: research
  paper:
    mode: paper
  live:
    mode: live
```

Keep the values readable and aligned with the workflow you intend to document for your team.

### `data`

These are the `project.yaml` data knobs that are wired into the current canonical runtime:

| Field | Required | What it controls |
| --- | --- | --- |
| `symbols` | Yes | Universe of tickers for research and agent runs |
| `start_date` | Yes | Historical start date |
| `end_date` | Yes | Historical end date |
| `timeframe` | No | Primary bar timeframe, default `1d` |
| `test_start` | No | Start of the time-aware test window |
| `test_end` | No | End of the time-aware test window |
| `cache_dir` | No | Cache directory for fetched data |
| `cache_path` | No | Alternate cache directory key used by the loader |
| `cache_expiration_days` | No | Cache freshness window |
| `use_cache` | No | Enable or disable cache reads |
| `refresh` | No | Force fresh downloads |
| `max_workers` | No | Parallel symbol fetch count |

### Symbol Shapes

You can write `symbols` as plain tickers:

```yaml
data:
  symbols: ["AAPL", "MSFT"]
```

Or as mappings:

```yaml
data:
  symbols:
    - ticker: AAPL
      asset_class: equities
    - ticker: ES
      asset_class: futures
```

The validator accepts the mapping style and checks `asset_class` values against `config/impact_config.yaml`.

Important current limitation:

- The canonical runtime compiler currently turns symbols into plain tickers for research and agent runs
- That means per-symbol `asset_class` choices are **validated on input** but are **not fully propagated** through the canonical runtime path yet

### Time-Aware Splitting

- If `test_start` and `test_end` are present, the canonical research run uses that window
- If only `test_start` is present, train data is everything before it and test data is everything from that point onward
- If the requested window validates but your downloaded data does not fully cover it, the pipeline falls back to a chronological split using `training.test_size`
- `research.evaluation.use_configured_test_window: false` disables the explicit window entirely

### Accepted but Not Fully Wired Yet

The project schema can accept more than the canonical compiler uses. Two important examples:

- `secondary_timeframes` is accepted by the data schema, but the canonical `project.yaml` compiler does not currently pass it through to runtime configs
- Per-symbol `asset_class` metadata is validated, but it is not fully preserved through canonical runtime compilation

### `features`

Canonical projects define features through `features.definitions`.

```yaml
features:
  definitions:
    - name: rsi_14
      type: technical
      params:
        period: 14
    - name: breakout_20
      type: custom
      params:
        kind: volatility_breakout
        lookback: [20]
        threshold: 2.0
```

### Feature Definition Fields

| Field | Required | Notes |
| --- | --- | --- |
| `name` | Yes | User-facing name referenced by agent context |
| `type` | Yes | Use `technical` or `custom` for the current canonical runtime |
| `params` | No | Per-feature settings |

### `technical` Feature Params That Work Today

| Param | Meaning |
| --- | --- |
| `price_features` | Price-ratio features such as `close_to_open` and `price_range` |
| `period` or `rsi_period` | RSI period |
| `macd_params` | MACD settings |
| `stoch_params` | Stochastic settings |
| `atr_periods` | ATR lookback list |
| `bollinger_bands` | Bollinger Band settings |
| `keltner_channels` | Keltner Channel settings |

Example:

```yaml
features:
  definitions:
    - name: technical_core
      type: technical
      params:
        price_features:
          - close_to_open
          - high_to_low
        rsi_period: 14
        macd_params:
          fast: 12
          slow: 26
          signal: 9
        stoch_params:
          k: 14
          d: 3
        atr_periods: [14]
        bollinger_bands:
          period: 20
          std_dev: 2
```

### `custom` Feature Families That Work Today

The canonical compiler supports these custom families:

- `price_momentum`
- `volume_momentum`
- `mean_reversion`
- `volatility_breakout`

Use `params.kind` when the feature name is not self-explanatory:

```yaml
features:
  definitions:
    - name: volume_spike_20
      type: custom
      params:
        kind: volume_momentum
        periods: [20]
    - name: mean_reversion_10
      type: custom
      params:
        kind: mean_reversion
        lookback: [10]
```

Important current limitation:

- Other `type` values may validate as generic strings, but the canonical runtime only compiles `technical` and `custom` definitions today

### `research`

The `research` block is the operational heart of the canonical research workflow.

| Field | What it does |
| --- | --- |
| `enabled` | Must be `true` for `quanttradeai research run` |
| `labels.type` | Currently `forward_return` only |
| `labels.horizon` | Forward-return lookahead |
| `labels.buy_threshold` | Threshold for label `1` |
| `labels.sell_threshold` | Threshold for label `-1` |
| `model.kind` | Currently `classifier` |
| `model.family` | Currently `voting` |
| `model.tuning.enabled` | Enables or disables Optuna tuning |
| `model.tuning.trials` | Number of Optuna trials |
| `evaluation.split` | Currently `time_aware` |
| `evaluation.use_configured_test_window` | Use or ignore `data.test_start` / `data.test_end` |
| `backtest.costs.enabled` | Enables transaction costs for research and agent backtests |
| `backtest.costs.bps` | Basis points used for research and agent backtests |

Example:

```yaml
research:
  enabled: true
  labels:
    type: forward_return
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: classifier
    family: voting
    tuning:
      enabled: true
      trials: 50
  evaluation:
    split: time_aware
    use_configured_test_window: true
  backtest:
    costs:
      enabled: true
      bps: 5
```

### `agents`

`project.yaml` can define agents directly, but the current CLI only runs a narrower subset than the schema allows.

### What `agent run` Supports Today

- `kind`: `llm` and `hybrid`
- `mode`: `backtest`

The schema accepts more values (`rule`, `model`, `paper`, `live`), but the current runtime does not execute them through `quanttradeai agent run`.

### Agent Fields

| Field | Used at runtime today | Notes |
| --- | --- | --- |
| `name` | Yes | The value passed to `--agent` |
| `kind` | Yes | Must be `llm` or `hybrid` for the current runtime |
| `mode` | Yes | Must be `backtest` for the current runtime |
| `llm.provider` | Yes | Used to build the LiteLLM model name |
| `llm.model` | Yes | Used to build the LiteLLM model name |
| `llm.prompt_file` | Yes | Required for `llm` and `hybrid`; resolved relative to project root |
| `llm.api_key_env_var` | Yes | Optional override for the provider API key env var |
| `llm.extra` | Yes | Passed through to LiteLLM |
| `context.market_data` | Yes | Adds recent OHLCV bars to prompt context |
| `context.features` | Yes | Adds requested feature values to prompt context |
| `context.model_signals` | Yes | Adds model signal payloads from `model_signal_sources` |
| `context.positions` | Yes | Adds target-position state |
| `context.risk_state` | Yes | Adds decision count, current direction, and `risk` dict |
| `context.orders` | No | Accepted by schema but not populated by the current backtest runtime |
| `context.news` | No | Accepted by schema but not populated by the current backtest runtime |
| `context.memory` | No | Accepted by schema but not populated by the current backtest runtime |
| `context.notes` | No | Accepted by schema but not populated by the current backtest runtime |
| `tools` | Partly | Allowed values are `get_quote`, `get_position`, and `place_order`; they are included in prompt context, but not executed as real tools by the runtime |
| `risk.max_position_pct` | Yes | Used to size positions |
| Other `risk` keys | Partly | Included in prompt context, but not enforced by the backtest engine |
| `model_signal_sources` | Yes | Must be objects with `name` and `path` for runtime use |

### LLM Agent Example

```yaml
agents:
  - name: breakout_gpt
    kind: llm
    mode: backtest
    llm:
      provider: openai
      model: gpt-5.3
      prompt_file: prompts/breakout.md
      api_key_env_var: OPENAI_API_KEY
      extra: {}
    context:
      market_data:
        enabled: true
        timeframe: "1d"
        lookback_bars: 20
      features: ["rsi_14"]
      positions: true
      risk_state: true
    tools: ["get_quote", "get_position", "place_order"]
    risk:
      max_position_pct: 0.05
      max_daily_loss_pct: 0.02
```

### Hybrid Agent Example

```yaml
agents:
  - name: hybrid_swing_agent
    kind: hybrid
    mode: backtest
    llm:
      provider: openai
      model: gpt-5.3
      prompt_file: prompts/hybrid_swing.md
    context:
      features: ["rsi_14"]
      model_signals: ["aapl_daily_classifier"]
      positions: true
    model_signal_sources:
      - name: aapl_daily_classifier
        path: models/trained/aapl_daily_classifier
    tools: ["get_quote", "place_order"]
    risk:
      max_position_pct: 0.05
```

### Path Resolution Rules

- `llm.prompt_file` is resolved relative to the project root
- `model_signal_sources[].path` is resolved relative to the project root
- Absolute paths are also accepted

Important current limitation:

- Deprecated string-only `model_signal_sources` entries may still validate with a warning, but `quanttradeai agent run` rejects them at runtime

### `deployment`

`deployment` is required and validated, but it is mostly descriptive today.

```yaml
deployment:
  target: docker-compose
  mode: paper
```

Keep it present and readable, but do not expect it to drive the current research or agent execution path.

## What Gets Written at Runtime

### `quanttradeai validate`

Writes timestamped artifacts under `reports/config_validation/<timestamp>/`:

- `resolved_project_config.yaml`
- `summary.json`

### `quanttradeai research run`

Writes a timestamped run folder under `runs/research/<timestamp>_<project>/` with:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `runtime_backtest_config.yaml`
- `summary.json`
- `metrics.json`
- `backtest_summary.json` when backtests succeed

### `quanttradeai agent run`

Writes a timestamped run folder under `runs/agent/backtest/<timestamp>_<agent>/` with:

- `resolved_project_config.yaml`
- `runtime_model_config.yaml`
- `runtime_features_config.yaml`
- `summary.json`
- `metrics.json`
- `equity_curve.csv`
- `decisions.jsonl`
- `prompt_samples.json`

## Compatibility Knobs That Still Work

If you are migrating from older YAMLs, the canonical research compiler can still read some non-canonical top-level sections such as:

- `news`
- `training`
- `trading`
- `execution`
- `models`

These are best treated as migration aids, not as the preferred shape for a new `project.yaml`.

Important current limitation:

- Those compatibility sections are mainly respected by the research compiler
- The current agent runtime uses a different project-to-runtime path, so compatibility settings are not applied there in the same way

## Bottom Line

If you want the least surprising `project.yaml` today:

- keep `data`, `features`, `research`, and `agents` explicit
- use `technical` and `custom` feature definitions only
- use object-based `model_signal_sources`
- treat `profiles` and `deployment` as required metadata
- keep live-trading settings in the runtime YAML files, not in `project.yaml`
