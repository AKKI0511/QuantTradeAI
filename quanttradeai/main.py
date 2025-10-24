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
from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.data.datasource import WebSocketDataSource
import pandas as pd
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics
from quanttradeai.models.classifier import MomentumClassifier
from quanttradeai.trading.drawdown_guard import DrawdownGuard
from typing import Tuple
import yaml
import json
from datetime import datetime
import os

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    if test_start:
        start_dt = pd.to_datetime(test_start)
        if test_end:
            end_dt = pd.to_datetime(test_end)
            train_df = df[df.index < start_dt]
            test_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        else:
            train_df = df[df.index < start_dt]
            test_df = df[df.index >= start_dt]
    else:
        test_size = float(train_cfg.get("test_size", 0.2))
        n = len(df)
        split_idx = max(1, int(n * (1 - test_size)))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "Invalid train/test window produced empty split. Adjust test_start/test_end or test_size."
        )
    return train_df, test_df


def run_pipeline(config_path: str = "config/model_config.yaml"):
    """Run the end-to-end training pipeline.

    Loads data, generates features and labels, tunes hyperparameters,
    trains the ensemble model per symbol, evaluates, and persists artifacts.

    Example
    -------
    >>> # CLI equivalent: `poetry run quanttradeai train -c config/model_config.yaml`
    >>> results = run_pipeline("config/model_config.yaml")
    >>> sorted(results.keys())  # doctest: +ELLIPSIS
    ...
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

        # Process each stock
        results = {}
        for symbol, df in data_dict.items():
            logger.info(f"\nProcessing {symbol}...")

            # 2. Generate Features
            df_processed = data_processor.process_data(df)

            # 3. Generate Labels
            df_labeled = data_processor.generate_labels(df_processed)

            # 4. Time-aware Split
            train_df, test_df = time_aware_split(df_labeled, config)
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

        # Save experiment results
        with open(f"{experiment_dir}/results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info("\nPipeline completed successfully!")
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


def evaluate_model(config_path: str, model_path: str) -> None:
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


async def run_live_pipeline(config_path: str, url: str) -> None:
    """Run a minimal real-time pipeline using WebSocket input.

    Example
    -------
    >>> # asyncio.run(run_live_pipeline("config/model_config.yaml", "wss://example"))
    """

    processor = DataProcessor()
    loader = DataLoader(config_path, data_source=WebSocketDataSource(url))

    def handle(df: pd.DataFrame) -> None:
        logger.info("Received update:\n%s", df.tail(1))

    await loader.stream_data(processor, callback=handle)


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

    data_dict = loader.fetch_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(f"reports/backtests/{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for symbol, df in data_dict.items():
        try:
            df_proc = processor.process_data(df)
            df_lbl = processor.generate_labels(df_proc)
            train_df, test_df = time_aware_split(df_lbl, cfg)
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

            trading_cfg = (cfg or {}).get("trading", {})
            stop_loss = trading_cfg.get("stop_loss")
            take_profit = trading_cfg.get("take_profit")

            result_df = simulate_trades(
                bt_df,
                stop_loss_pct=stop_loss,
                take_profit_pct=take_profit,
                execution=exec_cfg,
                drawdown_guard=drawdown_guard,
            )
            metrics = compute_metrics(result_df)

            # Persist artifacts
            out_dir = base_dir / symbol
            out_dir.mkdir(parents=True, exist_ok=True)
            result_df[["strategy_return", "equity_curve"]].to_csv(
                out_dir / "equity_curve.csv",
                index=True,
            )
            ledger = result_df.attrs.get("ledger")
            if ledger is not None and not ledger.empty:
                ledger.to_csv(out_dir / "ledger.csv", index=False)
            # Write metrics atomically to avoid partial JSON on serialization errors
            metrics_text = json.dumps(metrics, indent=2)
            with open(out_dir / "metrics.json", "w") as f:
                f.write(metrics_text)

            logger.info("%s backtest metrics: %s", symbol, metrics)
            summary[symbol] = {
                "metrics": metrics,
                "output_dir": str(out_dir),
            }
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol, exc)
            summary[symbol] = {"error": str(exc)}

    return summary


def main():
    # Lazy import Typer CLI to avoid introducing a hard dependency in unit tests
    # that import this module for functions only.
    from .cli import app  # type: ignore

    app()


if __name__ == "__main__":
    main()
