# Features

The `features` section declares feature definitions in project YAML. QuantTradeAI compiles those definitions into `runtime_features_config.yaml`, then `DataProcessor` generates the actual columns for research and agent runs.

## Used By

| Command | How features are used |
|---|---|
| `quanttradeai validate` | Checks agent context references against `features.definitions`. |
| `quanttradeai research run` | Compiles definitions into runtime feature config for model training and model backtests. |
| `quanttradeai agent run --mode backtest` | Generates historical features for rule, LLM, hybrid, and model agent backtests. |
| `quanttradeai agent run --mode paper` | Generates features over bootstrap and streaming/replay bars where applicable. |
| `quanttradeai agent run --mode live` | Generates live runtime features and writes live runtime snapshots. |

## Supported Fields

Each feature definition has the same shape:

```yaml
features:
  definitions:
    - name: rsi_14
      type: technical
      params:
        period: 14
```

| Field | Type | Notes |
|---|---|---|
| `name` | string | Human-facing feature name used by agents and sweeps. |
| `type` | string | Runtime-supported values are `technical` and `custom`. |
| `params` | mapping | Parameters compiled into runtime feature config. |

### Technical Features

Any `type: technical` definition enables the technical indicator step. If no `price_features` are provided, QuantTradeAI enables the standard price ratios:

- `close_to_open`
- `high_to_low`
- `close_to_high`
- `close_to_low`
- `price_range`

Supported technical params:

| Param | Runtime effect |
|---|---|
| `price_features` | List of supported price ratio columns. |
| `sma_periods` | Integer or list of integers used for generated `sma_<period>` columns. |
| `period` | Sets RSI period. |
| `rsi_period` | Sets RSI period; same runtime target as `period`. |
| `macd_params` | Mapping with `fast`, `slow`, and `signal`. |
| `stoch_params` | Mapping with `k` and `d`. |
| `atr_periods` | Integer or list of integers for ATR columns. |
| `bollinger_bands` | Mapping with `period` and `std_dev`. |
| `keltner_channels` | Mapping with `periods` and `atr_multiple`. |

SMA names are meaningful. A feature named `sma_20` or `sma_50` adds that period to runtime SMA generation even if `params` is empty.

```yaml
features:
  definitions:
    - name: rsi_14
      type: technical
      params:
        period: 14
        macd_params: {fast: 12, slow: 26, signal: 9}
        stoch_params: {k: 14, d: 3}
        atr_periods: [14]
        bollinger_bands: {period: 20, std_dev: 2.0}
    - name: sma_20
      type: technical
      params: {}
    - name: sma_50
      type: technical
      params: {}
```

If technical features exist but no explicit SMA periods are configured, the runtime feature schema supplies default SMA periods: `5`, `10`, `20`, `50`, and `200`.

### Custom Features

Supported custom feature kinds:

| Kind | Generated column pattern | Params |
|---|---|---|
| `price_momentum` | `price_momentum_<period>` | `periods`, `lookback`, `period`, or `window` |
| `volume_momentum` | `volume_momentum_<period>` | `periods`, `lookback`, `period`, or `window` |
| `mean_reversion` | `mean_reversion_<period>` | `periods`, `lookback`, `period`, or `window` |
| `volatility_breakout` | `volatility_breakout_<lookback>` and `volatility_breakout` | `lookback`, `periods`, `period`, or `window`; optional `threshold` |

Use `params.kind` when the name is not self-describing:

```yaml
features:
  definitions:
    - name: momentum_10
      type: custom
      params:
        kind: price_momentum
        window: 10
    - name: volume_spike_20
      type: custom
      params:
        kind: volume_momentum
        periods: [20]
    - name: breakout_20
      type: custom
      params:
        kind: volatility_breakout
        lookback: [20]
        threshold: 2.0
```

Without `params.kind`, QuantTradeAI can infer custom kinds only from names that match the supported kinds, start with `price_` or `volume_`, contain `reversion`, or contain `breakout`.

## Runtime Compilation

Project definitions are compiled into a runtime config with these sections:

| Runtime section | Source |
|---|---|
| `pipeline.steps` | Technical/custom definitions decide whether generation steps are included. |
| `price_features` | Technical params and SMA name parsing. |
| `momentum_features` | RSI, MACD, and stochastic params. |
| `volatility_features` | ATR, Bollinger Bands, and Keltner params. |
| `custom_features` | Supported custom feature definitions. |
| `feature_selection` | Runtime default: recursive selection with `n_features: 20`. |
| `preprocessing` | Runtime default: standard scaling and winsorized outlier handling. |

## Expected Outcomes

Runs that compile features write:

- `runtime_features_config.yaml`
- generated feature columns inside the training/backtest pipeline
- `preprocessing.json` under the research experiment directory
- `preprocessing_summary.json` and preprocessor artifacts under trained model directories

Agent artifacts can include feature values inside `decisions.jsonl` when an agent requests them in `context.features`.

## Common Mistakes

| Mistake | Result |
|---|---|
| Using a custom feature name that cannot map to a supported kind | Runtime config compilation fails. |
| Omitting periods for custom period-based features | The feature compiles with no useful periodized column. |
| Expecting an agent to see a feature that is not listed in `context.features` | The feature is generated but not included in that agent's decision context. |
| Referencing `sma_200` in a rule without defining or generating it | Validation fails for SMA crossover rules. |
| Adding a definition with a type other than `technical` or `custom` | The runtime compiler does not generate a feature step from it. |

