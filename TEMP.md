PR Scope: Model-Integrated Backtesting CLI (backtest-model)

Summary
- Add an end-to-end CLI subcommand `backtest-model` that loads a saved model, generates time-aware test predictions, converts them into trading signals, and runs the vectorized backtester with execution costs/slippage/liquidity. Persist metrics, equity curve, and (optional) trade ledger. This stitches the ML pipeline to the backtesting engine per the Phase 1 roadmap item “Enhanced Backtesting,” providing a single-command, reproducible evaluation of real PnL with costs.

Why This Now
- Current state: Training/evaluation exists (classification metrics), and a powerful backtester exists, but they are decoupled. The README Roadmap emphasizes Enhanced Backtesting in Phase 1. Integrating model outputs with the backtester delivers immediate, decision-grade value and a clear demo path for users (max brownie points).
- Benefits: Single command to quantify strategy viability; avoids confusion between classification scores vs. trading performance; leverages existing time-aware split, execution modeling, and config schemas.

Goals
- Provide `poetry run quanttradeai backtest-model -m <model_dir> [-c <model_cfg>] [-b <bt_cfg>] [flags...]` that:
  - Loads model+scaler+feature_columns from `<model_dir>`.
  - Loads data via `model_config.yaml`, processes features, creates labels, and performs time-aware split using existing logic.
  - Uses the saved `feature_columns` to build X_test and generate predictions for the test window without mutating `feature_columns`.
  - Converts predictions to signals (-1/0/1) aligned to `Close` (and `Volume` for liquidity).
  - Runs `simulate_trades(..., execution=...)` and `compute_metrics(...)` with execution options from a `backtest_config.yaml` and/or CLI overrides.
  - Saves: `metrics.json`, `equity_curve.csv`, `ledger.csv` under `reports/backtests/<timestamp>/<symbol>/`.
  - Prints a concise summary to stdout.

Non-Goals
- No new models or feature engineering changes.
- No broker/exchange order execution integration.
- No refactor of `MomentumClassifier` internals (we only read its saved `feature_columns`).

Design & Changes
- CLI: Add a new subparser `backtest-model` in `quanttradeai/main.py` with args:
  - `-c/--config` (default `config/model_config.yaml`)
  - `-b/--backtest-config` (optional; default `config/backtest_config.yaml` if present)
  - `-m/--model-path` (required)
  - Optional overrides mirroring existing `backtest` command: `--cost-bps`, `--cost-fixed`, `--slippage-bps`, `--slippage-fixed`, `--liquidity-max-participation`
- Implementation (new function `run_model_backtest`):
  1) Load `model_config.yaml` and create `DataLoader`/`DataProcessor`.
  2) Fetch data; for each configured symbol: process, label, time-aware split (reuse `time_aware_split`).
  3) Load trained model via `MomentumClassifier.load_model(model_path)`.
  4) Build `X_test` strictly from `model.feature_columns`; get `y_test` from split df; generate predictions via `model.predict(X_test)`.
  5) Construct a minimal DataFrame for backtesting: copy `test_df[['Close','Volume']]` + set `label = predictions`.
  6) Load backtest `execution` config (from `--backtest-config` or default), apply CLI overrides (same pattern as existing `backtest` subcommand).
  7) Call `simulate_trades(df_bt, execution=exec_cfg)` and `compute_metrics`.
  8) Persist artifacts under `reports/backtests/<timestamp>/<symbol>/`.
  9) Print concise summary (Sharpe, MDD, CAGR if available) to stdout.

Edge Handling
- Feature column alignment: never call `prepare_data` after `load_model` (it would overwrite `feature_columns`). Select columns using `model.feature_columns` with exact order.
- Missing `Volume`: if absent, inject a large constant volume to keep liquidity logic functional.
- Label space: assumes training used {-1,0,1} from `generate_labels`. Predictions are used as-is; if binary {0,1}, we map `{0:0, 1:1}` and never emit -1.
- Time windows: reuse `time_aware_split` rules (supports `data.test_start`/`data.test_end` or fallback to tail fraction).

Tests
- Add `tests/integration/test_backtest_model_cli.py`:
  - Mocks `DataLoader.fetch_data` to return a tiny OHLCV frame; uses a trivial stub model saved to disk with known `feature_columns` and deterministic predictions.
  - Runs the CLI via `main.main()` with args for `backtest-model` and validates:
    - No exception, stdout contains “Sharpe”/“Max Drawdown”.
    - Artifacts created under `reports/backtests/...` with expected files.
  - Parametrize with/without backtest overrides to ensure they plumb through.

Docs
- README “Usage” and docs/quick-reference.md: add a “Backtest a saved model” snippet and brief explanation of where outputs are saved.
- docs/getting-started.md: add a short step after Train: “Backtest a trained model.”

Acceptance Criteria
- `poetry run quanttradeai backtest-model -m models/experiments/<ts>/<SYM>` runs end-to-end without code edits, saves artifacts, and prints metrics.
- Works with time-aware test windows if set; otherwise uses last `training.test_size` as test chronologically.
- CLI overrides for costs/slippage/liquidity take effect in results.

Risks & Mitigations
- Mismatch feature columns: enforce using saved `feature_columns` only; error clearly if missing any required column.
- Intraday timeframes: labeling uses bar counts (forward_returns). That’s consistent with current behavior; we will document this and keep configuration of label horizon for a future PR.

Out of Scope (future PRs)
- Configurable labeling horizon/threshold in YAML.
- Multi-timeframe feature fusion (e.g., 1h + 1d features).
- Portfolio-level multi-symbol backtest from model predictions.

