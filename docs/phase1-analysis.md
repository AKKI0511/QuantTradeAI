# Phase 1 Analysis & Codex Prompt

## Phase 1 status (current)

QuantTradeAI already includes several Phase 1 roadmap items:

- **Streaming hardening**: gateway orchestration, health monitoring, alerting, metrics, and recovery management are implemented under `quanttradeai/streaming`. See the streaming gateway and monitoring modules for reconnection logic and metrics collection.
- **Backtesting realism**: execution modeling includes transaction costs, slippage, liquidity constraints, market impact, borrow fees, intrabar fills, and a trade ledger.
- **Risk & portfolio controls**: drawdown guard, turnover limits, and position sizing hooks exist for backtests and live position management.
- **Multi‑timeframe groundwork**: secondary timeframe ingestion and resampling in the data loader plus derived multi‑timeframe feature generation in the data processor.

## Gaps / inconsistencies identified

1. **Feature configuration is disconnected from training/backtests**
   - `DataProcessor` always loads `config/features_config.yaml` by default.
   - The CLI and pipeline (`run_pipeline`, `run_model_backtest`, `evaluate_model`) do not accept a features config path, so users cannot run experiments with alternate feature definitions without editing the default file.
   - `config/model_config.yaml` still has a legacy `features:` block that is never read, which can mislead users about which settings are active.

2. **Backtest YAML missing available execution knobs**
   - `config/backtest_config.yaml` omits `borrow_fee` and `intrabar` sections even though the execution engine and schemas support them.

3. **Streaming health config flag is unused**
   - `streaming_health.monitoring.enabled` is present in `config/streaming.yaml`, but `StreamingGateway` does not consult it, so the flag does not actually disable monitoring.

## Proposed end‑to‑end feature for Codex

**Make feature configuration explicit and experiment‑friendly.**

Add a `features_config` path to `model_config.yaml` (and CLI overrides) so training, evaluation, backtesting, and live trading all use the same feature set without editing the default YAML. This aligns with Phase 1 “stabilize & polish” and reduces configuration drift.

### Success criteria

- `model_config.yaml` can include a `features_config` path, defaulting to `config/features_config.yaml` when absent.
- `quanttradeai train`, `evaluate`, `backtest-model`, and `live-trade` accept `--features-config` CLI overrides.
- `DataProcessor` is initialized with the chosen features config path in all pipeline entrypoints.
- Config validation reports include the resolved features config path and validate it.
- Documentation explains the new path and deprecates the unused `features:` block in `model_config.yaml`.
- Tests cover the CLI override and model config loading behavior.

## Codex prompt

Implement the following end‑to‑end feature in QuantTradeAI:

**Goal:** Make feature configuration explicit and consistent across training, evaluation, backtesting, and live trading by allowing a configurable `features_config` path.

**Scope & changes**

1. **Configuration changes**
   - Add an optional `features_config` field to `ModelConfigSchema` in `quanttradeai/utils/config_schemas.py` (string path, default `config/features_config.yaml`).
   - Update `config/model_config.yaml` to include `features_config: config/features_config.yaml` and add a short comment explaining its purpose.
   - Deprecate or remove the unused `features:` block in `model_config.yaml` (documented behavior is to use `features_config.yaml`). If you keep it, add a warning log when it’s present.

2. **Pipeline wiring**
   - Update `run_pipeline`, `evaluate_model`, `run_model_backtest`, and `run_live_pipeline` in `quanttradeai/main.py` to initialize `DataProcessor` with the resolved features config path.
   - Ensure that `run_model_backtest` uses the same features config that the model was trained with (prefer the model config value unless a CLI override is provided).

3. **CLI updates**
   - Add `--features-config` option to the `train`, `evaluate`, `backtest-model`, and `live-trade` commands in `quanttradeai/cli.py`.
   - CLI override should take precedence over the model config value.

4. **Config validation**
   - Update `quanttradeai/utils/config_validator.py` so the validation summary for `model_config` includes the resolved features config path.
   - Ensure validation fails with a helpful error if the referenced features config file is missing.

5. **Docs**
   - Update `docs/configuration.md` (and `docs/quick-reference.md` if applicable) to document the new `features_config` field and CLI flag.
   - Remove or clearly mark the legacy `features:` block in `model_config.yaml` as deprecated.

6. **Tests**
   - Add/extend pytest coverage to confirm that:
     - A model config with `features_config` loads correctly.
     - CLI `--features-config` overrides the model config value.
     - The pipeline uses the intended features config path (can be validated via a small fixture or by patching `DataProcessor`).

**Notes**
- Keep changes focused on the “happy path” (no over‑engineering).
- Preserve existing defaults so current users are not broken.
- Run `make format`, `make lint`, and `make test` (or `pre-commit run --all-files`).

Deliver as a single PR with clear commit message and a concise summary in the PR body.
