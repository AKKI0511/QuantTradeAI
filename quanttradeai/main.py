import warnings

# Suppress pandas_ta pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

"""High-level pipeline functions and legacy CLI bridge.

This module exposes programmatic functions (run_pipeline, evaluate_model, etc.)
used by tests and scripts. The interactive CLI is implemented with Typer in
``quanttradeai/cli.py``. The ``main()`` function here simply delegates to the
Typer app to preserve the existing console entry points.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
import json
from datetime import datetime
import os

from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics
from quanttradeai.models.classifier import MomentumClassifier
from quanttradeai.trading.drawdown_guard import DrawdownGuard
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.impact_loader import (
    load_impact_config,
    merge_execution_with_impact,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories(cache_dir: str = "data/raw"):
    """Create project directories used for caching and artifacts.

    Example
    -------
    >>> setup_directories("data/raw")
    # Ensures data/, models/, and reports/ subfolders exist
    """
    dirs = [
        cache_dir,
        "data/processed",
        "models/trained",
        "models/experiments",
        "reports/figures",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def _write_validation_report(report_path: Path, report: dict) -> None:
    """Persist validation results to JSON and CSV formats."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    try:
        import pandas as pd

        rows = []
        for symbol, details in report.items():
            row = {"symbol": symbol, **details}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(report_path.with_suffix(".csv"), index=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write CSV validation report: %s", exc)


def _write_coverage_report(report_path: Path, coverage: dict) -> None:
    """Persist test-window coverage results to JSON and CSV formats."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(coverage, fh, indent=2)

    try:
        rows = []
        for symbol, details in coverage.items():
            row = {"symbol": symbol, **details}
            rows.append(row)
        pd.DataFrame(rows).to_csv(report_path.with_suffix(".csv"), index=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to write CSV coverage report: %s", exc)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a DatetimeIndex.

    If the index is not datetime-like, try common fallbacks such as a
    'Date'/'Datetime' column. Raise a clear error if conversion fails.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    # Try to promote a Date/Datetime column
    for col in ("Datetime", "Date", "timestamp", "time"):
        if col in df.columns:
            out = df.copy()
            out.index = pd.to_datetime(out[col])
            return out
    # Try to parse current index
    try:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(
            "DataFrame must have a DatetimeIndex or a Date/Datetime column"
        ) from exc


def time_aware_split(
    df_labeled: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return chronological train/test splits using config windows.

    Rules
    -----
    - If data.test_start and data.test_end are provided:
      train = idx < test_start; test = test_start <= idx <= test_end
    - If only data.test_start: train = idx < test_start; test = idx >= test_start
    - Else: fallback to last `training.test_size` fraction as test (no shuffle).
    """
    df = _ensure_datetime_index(df_labeled)
    data_cfg = (cfg or {}).get("data", {})
    train_cfg = (cfg or {}).get("training", {})
    test_start = data_cfg.get("test_start")
    test_end = data_cfg.get("test_end")

    def _fraction_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_size = float(train_cfg.get("test_size", 0.2))
        if not 0 < test_size < 1:
            raise ValueError(
                "training.test_size must be between 0 and 1 for chronological fallback splits."
            )
        n = len(df)
        split_idx = max(1, int(n * (1 - test_size)))
        train_part = df.iloc[:split_idx]
        test_part = df.iloc[split_idx:]
        return train_part, test_part

    def _window_has_full_coverage(
        start_dt: pd.Timestamp, end_dt: pd.Timestamp | None, test_slice: pd.DataFrame
    ) -> bool:
        """Return True if the dataset fully spans the requested test window."""

        if end_dt is None:
            return True

        if df.index.min() > start_dt or df.index.max() < end_dt:
            return False

        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq:
            expected_points = len(pd.date_range(start_dt, end_dt, freq=inferred_freq))
            return len(test_slice) >= expected_points

        return True

    fallback_used = False
    coverage_ok: bool | None = True if test_start else None
    data_start = df.index.min()
    data_end = df.index.max()
    split_strategy = "window" if test_start else "fraction"
    if test_start:
        start_dt = pd.to_datetime(test_start)
        end_dt = pd.to_datetime(test_end) if test_end else None
        train_df = df[df.index < start_dt]
        test_df = (
            df[(df.index >= start_dt) & (df.index <= end_dt)]
            if end_dt
            else df[df.index >= start_dt]
        )

        coverage_ok = _window_has_full_coverage(start_dt, end_dt, test_df)

        if len(train_df) == 0 or len(test_df) == 0 or not coverage_ok:
            fallback_used = True
            fallback_test_size = float(train_cfg.get("test_size", 0.2))
            logger.warning(
                "Requested test window %s to %s not fully present in data; falling back to chronological split using training.test_size=%.3f.",
                test_start,
                test_end or "data end",
                fallback_test_size,
            )
            train_df, test_df = _fraction_split()
    else:
        train_df, test_df = _fraction_split()

    if len(train_df) == 0 or len(test_df) == 0:
        window_msg = (
            "chronological fallback split"
            if fallback_used or not test_start
            else "requested test window"
        )
        raise ValueError(
            f"Invalid train/test window produced empty split when using {window_msg}. Adjust data.test_* or training.test_size."
        )
    coverage: dict = {
        "data_start": data_start.isoformat(),
        "data_end": data_end.isoformat(),
        "test_start": pd.to_datetime(test_start).isoformat() if test_start else None,
        "test_end": pd.to_datetime(test_end).isoformat() if test_end else None,
        "train_start": train_df.index.min().isoformat(),
        "train_end": train_df.index.max().isoformat(),
        "test_start_actual": test_df.index.min().isoformat(),
        "test_end_actual": test_df.index.max().isoformat(),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "coverage_ok": coverage_ok,
        "fallback_used": fallback_used,
        "split_strategy": split_strategy if not fallback_used else "fraction_fallback",
    }
    return train_df, test_df, coverage


def _validate_or_raise(
    *,
    loader: DataLoader,
    data: dict,
    report_path: Path,
    skip_validation: bool,
) -> dict:
    """Validate fetched data and persist a report unless skipped."""

    if skip_validation:
        logger.warning(
            "Skipping data validation as requested; downstream results may be unreliable."
        )
        return {}

    validation_result = loader.validate_data(data)
    if isinstance(validation_result, tuple) and len(validation_result) == 2:
        is_valid, report = validation_result
    else:
        is_valid = bool(validation_result)
        report = {}
    _write_validation_report(report_path, report)

    if not is_valid:
        failed = [symbol for symbol, res in report.items() if not res.get("passed")]
        raise ValueError(
            f"Data validation failed for symbols: {', '.join(failed) if failed else 'unknown'}"
        )

    logger.info(
        "Data validation passed for %d symbols. Report saved to %s",
        len(report),
        report_path,
    )
    return report


def run_pipeline(
    config_path: str = "config/model_config.yaml",
    *,
    skip_validation: bool = False,
    include_coverage: bool = False,
):
    """Run the end-to-end training pipeline.

    Loads data, generates features and labels, tunes hyperparameters,
    trains the ensemble model per symbol, evaluates, and persists artifacts.

    Example
    -------
    >>> # CLI equivalent: `poetry run quanttradeai train -c config/model_config.yaml`
    >>> results = run_pipeline("config/model_config.yaml")
    >>> sorted(results.keys())  # doctest: +ELLIPSIS
    ...

    Set ``include_coverage=True`` to also receive coverage metadata:

    >>> results, coverage = run_pipeline("config/model_config.yaml", include_coverage=True)
    >>> sorted(coverage.keys())
    ['fallback_symbols', 'path']
    """

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    cache_dir = config.get("data", {}).get("cache_dir", "data/raw")
    setup_directories(cache_dir)

    # Initialize components
    data_loader = DataLoader(config_path)
    data_processor = DataProcessor()
    model = MomentumClassifier(config_path)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"models/experiments/{timestamp}"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 1. Fetch Data
        logger.info("Fetching data...")
        refresh = config.get("data", {}).get("refresh", False)
        data_dict = data_loader.fetch_data(refresh=refresh)

        # Validate data quality
        validation_path = Path(experiment_dir) / "validation.json"
        _validate_or_raise(
            loader=data_loader,
            data=data_dict,
            report_path=validation_path,
            skip_validation=skip_validation,
        )

        # Process each stock
        results = {}
        coverage_report: dict[str, dict] = {}
        for symbol, df in data_dict.items():
            logger.info(f"\nProcessing {symbol}...")

            # 2. Generate Features
            df_processed = data_processor.process_data(df)

            # 3. Generate Labels
            df_labeled = data_processor.generate_labels(df_processed)

            # 4. Time-aware Split
            train_df, test_df, coverage = time_aware_split(df_labeled, config)
            coverage_report[symbol] = coverage
            X_train, y_train = model.prepare_data(train_df)
            X_test, y_test = model.prepare_data(test_df)
            # Log split summary
            logger.info(
                "Split %s -> train[%s..%s]=%d, test[%s..%s]=%d",
                symbol,
                str(train_df.index.min()),
                str(train_df.index.max()),
                len(train_df),
                str(test_df.index.min()),
                str(test_df.index.max()),
                len(test_df),
            )

            # 5. Optimize Hyperparameters
            logger.info(f"Optimizing hyperparameters for {symbol}...")
            best_params = model.optimize_hyperparameters(X_train, y_train, n_trials=50)

            # 6. Train Model
            logger.info(f"Training model for {symbol}...")
            model.train(X_train, y_train, params=best_params)

            # 7. Evaluate Performance
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)

            # Save results
            results[symbol] = {
                "hyperparameters": best_params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

            # Save model
            model_path = f"{experiment_dir}/{symbol}"
            Path(model_path).mkdir(parents=True, exist_ok=True)
            model.save_model(model_path)

            logger.info(f"\n{symbol} Results:")
            logger.info(f"Train Metrics: {train_metrics}")
            logger.info(f"Test Metrics: {test_metrics}")

        coverage_path = Path(experiment_dir) / "test_window_coverage.json"
        _write_coverage_report(coverage_path, coverage_report)
        fallback_symbols = [
            symbol
            for symbol, details in coverage_report.items()
            if details.get("fallback_used")
        ]
        if fallback_symbols:
            logger.warning(
                "Fallback chronological split applied for symbols: %s. Coverage report: %s",
                ", ".join(fallback_symbols),
                coverage_path,
            )
        logger.info("Coverage report saved to %s", coverage_path)

        # Save experiment results
        with open(f"{experiment_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info("\nPipeline completed successfully!")
        if include_coverage:
            return (
                results,
                {
                    "path": coverage_path.as_posix(),
                    "fallback_symbols": fallback_symbols,
                },
            )

        return results

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


def fetch_data_only(config_path: str, refresh: bool = False) -> None:
    """Fetch and cache historical data.

    Example
    -------
    >>> fetch_data_only("config/model_config.yaml", refresh=True)
    """
    data_loader = DataLoader(config_path)
    data = data_loader.fetch_data(refresh=refresh)
    data_loader.save_data(data)


def evaluate_model(
    config_path: str, model_path: str, *, skip_validation: bool = False
) -> None:
    """Evaluate a persisted model on current config’s dataset.

    Example
    -------
    >>> evaluate_model("config/model_config.yaml", "models/experiments/20250101_000000/AAPL")
    """
    data_loader = DataLoader(config_path)
    data_processor = DataProcessor()
    model = MomentumClassifier(config_path)
    model.load_model(model_path)

    data_dict = data_loader.fetch_data()
    validation_path = Path(model_path) / "validation.json"
    _validate_or_raise(
        loader=data_loader,
        data=data_dict,
        report_path=validation_path,
        skip_validation=skip_validation,
    )
    results = {}
    for symbol, df in data_dict.items():
        df_processed = data_processor.process_data(df)
        df_labeled = data_processor.generate_labels(df_processed)
        X, y = model.prepare_data(df_labeled)
        metrics = model.evaluate(X, y)
        results[symbol] = metrics
        logger.info(f"{symbol} Evaluation: {metrics}")

    with open(os.path.join(model_path, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=4)


async def run_live_pipeline(
    *,
    model_config: str = "config/model_config.yaml",
    model_path: str,
    streaming_config: str = "config/streaming.yaml",
    risk_config: str | None = "config/risk_config.yaml",
    position_manager_config: str | None = "config/position_manager.yaml",
    enable_health_api: bool | None = None,
    initial_capital: float = 1_000_000.0,
    history_window: int = 512,
    min_history_for_features: int = 220,
    stop_loss_pct: float = 0.01,
) -> None:
    """Run end-to-end real-time trading using the streaming gateway."""

    from quanttradeai.streaming.live_trading import LiveTradingEngine

    engine = LiveTradingEngine(
        model_config=model_config,
        model_path=model_path,
        streaming_config=streaming_config,
        risk_config=risk_config,
        position_manager_config=position_manager_config,
        enable_health_api=enable_health_api,
        initial_capital=initial_capital,
        history_window=history_window,
        min_history_for_features=min_history_for_features,
        stop_loss_pct=stop_loss_pct,
    )
    await engine.start()


def _load_execution_cfg(
    backtest_config_path: str | None,
    *,
    cost_bps: float | None = None,
    cost_fixed: float | None = None,
    slippage_bps: float | None = None,
    slippage_fixed: float | None = None,
    liquidity_max_participation: float | None = None,
) -> dict:
    """Load execution config and apply CLI-style overrides."""
    exec_cfg: dict = {}
    if backtest_config_path and os.path.exists(backtest_config_path):
        with open(backtest_config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        exec_cfg = cfg.get("execution", {})

    if cost_bps is not None:
        exec_cfg.setdefault("transaction_costs", {})
        exec_cfg["transaction_costs"].update(
            {"enabled": True, "mode": "bps", "value": cost_bps}
        )
    if cost_fixed is not None:
        exec_cfg.setdefault("transaction_costs", {})
        exec_cfg["transaction_costs"].update(
            {"enabled": True, "mode": "fixed", "value": cost_fixed}
        )
    if slippage_bps is not None:
        exec_cfg.setdefault("slippage", {})
        exec_cfg["slippage"].update(
            {"enabled": True, "mode": "bps", "value": slippage_bps}
        )
    if slippage_fixed is not None:
        exec_cfg.setdefault("slippage", {})
        exec_cfg["slippage"].update(
            {"enabled": True, "mode": "fixed", "value": slippage_fixed}
        )
    if liquidity_max_participation is not None:
        exec_cfg.setdefault("liquidity", {})
        exec_cfg["liquidity"].update(
            {"enabled": True, "max_participation": liquidity_max_participation}
        )
    return exec_cfg


def run_model_backtest(
    *,
    model_config: str = "config/model_config.yaml",
    model_path: str,
    backtest_config: str | None = "config/backtest_config.yaml",
    risk_config: str | None = "config/risk_config.yaml",
    cost_bps: float | None = None,
    cost_fixed: float | None = None,
    slippage_bps: float | None = None,
    slippage_fixed: float | None = None,
    liquidity_max_participation: float | None = None,
    skip_validation: bool = False,
) -> dict:
    """Backtest a saved model’s predictions on the configured test window.

    Returns a dict keyed by symbol with metrics and artifact paths.
    """
    # Validate required paths
    if not model_config or not os.path.exists(model_config):
        raise FileNotFoundError(f"Model config not found: {model_config}")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Load configs and components
    with open(model_config, "r") as f:
        cfg = yaml.safe_load(f)
    setup_directories(cfg.get("data", {}).get("cache_dir", "data/raw"))
    loader = DataLoader(model_config)
    processor = DataProcessor()
    clf = MomentumClassifier(model_config)
    clf.load_model(model_path)

    # Backtest execution config
    exec_cfg = _load_execution_cfg(
        backtest_config,
        cost_bps=cost_bps,
        cost_fixed=cost_fixed,
        slippage_bps=slippage_bps,
        slippage_fixed=slippage_fixed,
        liquidity_max_participation=liquidity_max_participation,
    )
    impact_defaults = load_impact_config()

    drawdown_guard: DrawdownGuard | None = None
    if risk_config:
        risk_path = Path(risk_config)
        if risk_path.is_file():
            try:
                drawdown_guard = DrawdownGuard(config_path=str(risk_path))
                logger.info("Loaded risk config from %s", risk_path)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to load risk config %s: %s. Continuing without drawdown guard.",
                    risk_path,
                    exc,
                )
        else:
            logger.info(
                "Risk config not found at %s; continuing without drawdown guard.",
                risk_path,
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dict = loader.fetch_data()
    validation_path = Path(f"reports/backtests/{timestamp}") / "validation.json"
    _validate_or_raise(
        loader=loader,
        data=data_dict,
        report_path=validation_path,
        skip_validation=skip_validation,
    )
    base_dir = Path(f"reports/backtests/{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    prepared_data: dict[str, pd.DataFrame] = {}
    artifact_dirs: dict[str, Path] = {}
    coverage_report: dict[str, dict] = {}

    trading_cfg = (cfg or {}).get("trading", {})
    stop_loss = trading_cfg.get("stop_loss")
    take_profit = trading_cfg.get("take_profit")

    def _resolve_float(
        value: float | int | None, default: float, *, key: str, positive: bool = False
    ) -> float:
        if value is None:
            return default
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid trading.%s=%r; defaulting to %.4f", key, value, default
            )
            return default
        if positive and resolved <= 0:
            logger.warning(
                "trading.%s must be positive; defaulting to %.4f", key, default
            )
            return default
        if not positive and resolved < 0:
            logger.warning(
                "trading.%s must be non-negative; defaulting to %.4f", key, default
            )
            return default
        return resolved

    initial_capital = _resolve_float(
        trading_cfg.get("initial_capital"),
        100_000.0,
        key="initial_capital",
        positive=True,
    )
    max_risk_per_trade = _resolve_float(
        trading_cfg.get("max_risk_per_trade"),
        0.02,
        key="max_risk_per_trade",
        positive=True,
    )
    max_portfolio_risk = _resolve_float(
        trading_cfg.get("max_portfolio_risk"),
        0.10,
        key="max_portfolio_risk",
        positive=True,
    )

    def _execution_for(symbol: str) -> dict:
        asset_class = loader.asset_classes.get(symbol, "equities")
        return merge_execution_with_impact(exec_cfg, impact_defaults, asset_class)

    for symbol, df in data_dict.items():
        try:
            df_proc = processor.process_data(df)
            df_lbl = processor.generate_labels(df_proc)
            train_df, test_df, coverage = time_aware_split(df_lbl, cfg)
            coverage_report[symbol] = coverage
            # Build features from saved order
            missing = [
                c for c in (clf.feature_columns or []) if c not in test_df.columns
            ]
            if missing:
                raise ValueError(f"Missing required features for {symbol}: {missing}")
            X_test = test_df[clf.feature_columns].values
            preds = clf.predict(X_test)

            bt_df = test_df[
                [c for c in ["Close", "Volume"] if c in test_df.columns]
            ].copy()
            if "Volume" not in bt_df.columns:
                bt_df["Volume"] = 1e12  # effectively infinite liquidity
            bt_df["label"] = preds

            prepared_data[symbol] = bt_df
            artifact_dirs[symbol] = base_dir / symbol
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol, exc)
            summary[symbol] = {"error": str(exc)}

    if prepared_data:
        # Instantiate the portfolio manager without wiring the drawdown guard so
        # that turnover enforcement continues to operate on the historical bar
        # timestamps recorded during simulation rather than the wall-clock times
        # used when seeding portfolio allocations.
        portfolio_manager = PortfolioManager(
            capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade,
            max_portfolio_risk=max_portfolio_risk,
        )
        exec_cfg_by_symbol = {
            symbol: _execution_for(symbol) for symbol in prepared_data
        }
        try:
            results = simulate_trades(
                prepared_data,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                execution=exec_cfg_by_symbol,
                portfolio=portfolio_manager,
                drawdown_guard=drawdown_guard,
            )
        except Exception as exc:
            logger.error("Portfolio backtest failed: %s", exc)
            summary["portfolio"] = {"error": str(exc)}
            return summary

        for symbol, result_df in results.items():
            out_dir: Path = artifact_dirs.get(symbol, base_dir / symbol)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                cols_to_save = [
                    c
                    for c in ["strategy_return", "equity_curve"]
                    if c in result_df.columns
                ]
                if cols_to_save:
                    result_df[cols_to_save].to_csv(
                        out_dir / "equity_curve.csv", index=True
                    )
                ledger = result_df.attrs.get("ledger")
                if ledger is not None and not ledger.empty:
                    ledger.to_csv(out_dir / "ledger.csv", index=False)
                metrics = compute_metrics(result_df)
                metrics_text = json.dumps(metrics, indent=2)
                with open(out_dir / "metrics.json", "w") as f:
                    f.write(metrics_text)
            except Exception as exc:  # noqa: PERF203 - specific to artifact persistence
                logger.error("Backtest artifact write failed for %s: %s", symbol, exc)
                summary[symbol] = {"error": str(exc)}
                continue

            logger.info("%s backtest metrics: %s", symbol, metrics)
            summary[symbol] = {
                "metrics": metrics,
                "output_dir": out_dir.as_posix(),
            }

    coverage_path = base_dir / "test_window_coverage.json"
    _write_coverage_report(coverage_path, coverage_report)
    fallback_symbols = [
        symbol
        for symbol, details in coverage_report.items()
        if details.get("fallback_used")
    ]
    if fallback_symbols:
        logger.warning(
            "Fallback chronological split applied for symbols: %s. Coverage report: %s",
            ", ".join(fallback_symbols),
            coverage_path,
        )
    logger.info("Coverage report saved to %s", coverage_path)
    summary["coverage_report"] = {
        "path": coverage_path.as_posix(),
        "fallback_symbols": fallback_symbols,
    }

    return summary


def main():
    # Lazy import Typer CLI to avoid introducing a hard dependency in unit tests
    # that import this module for functions only.
    from .cli import app  # type: ignore

    app()


if __name__ == "__main__":
    main()
