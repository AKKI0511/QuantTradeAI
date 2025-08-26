import warnings

# Suppress pandas_ta pkg_resources deprecation warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

"""Command line interface entry points.

Provides convenience functions and CLI commands to run the end-to-end
pipeline, fetch data or evaluate models.
"""

import logging
from pathlib import Path
from quanttradeai.data.loader import DataLoader
from quanttradeai.data.processor import DataProcessor
from quanttradeai.data.datasource import WebSocketDataSource
import pandas as pd
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics
from quanttradeai.models.classifier import MomentumClassifier
from typing import Tuple
import yaml
import json
from datetime import datetime
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="QuantTradeAI command line interface")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch data and cache it")
    fetch_parser.add_argument(
        "-c", "--config", default="config/model_config.yaml", help="Path to config file"
    )
    fetch_parser.add_argument(
        "--refresh", action="store_true", help="Force refresh of cached data"
    )

    train_parser = subparsers.add_parser("train", help="Run full training pipeline")
    train_parser.add_argument(
        "-c", "--config", default="config/model_config.yaml", help="Path to config file"
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument(
        "-c", "--config", default="config/model_config.yaml", help="Path to config file"
    )
    eval_parser.add_argument(
        "-m", "--model-path", required=True, help="Directory containing saved model"
    )

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument(
        "-c",
        "--config",
        default="config/backtest_config.yaml",
        help="Path to backtest config",
    )
    grp_cost = backtest_parser.add_mutually_exclusive_group()
    grp_cost.add_argument("--cost-bps", type=float, help="Transaction cost in bps")
    grp_cost.add_argument("--cost-fixed", type=float, help="Fixed transaction cost")
    grp_slip = backtest_parser.add_mutually_exclusive_group()
    grp_slip.add_argument("--slippage-bps", type=float, help="Slippage in bps")
    grp_slip.add_argument("--slippage-fixed", type=float, help="Fixed slippage amount")
    backtest_parser.add_argument(
        "--liquidity-max-participation",
        type=float,
        help="Override liquidity max participation",
    )

    live_parser = subparsers.add_parser(
        "live-trade", help="Run real-time trading pipeline"
    )
    live_parser.add_argument(
        "-c", "--config", default="config/model_config.yaml", help="Path to config file"
    )
    live_parser.add_argument(
        "--url", required=True, help="WebSocket URL for streaming data"
    )

    args = parser.parse_args()

    if args.command == "fetch-data":
        fetch_data_only(args.config, args.refresh)
    elif args.command == "train":
        run_pipeline(args.config)
    elif args.command == "evaluate":
        evaluate_model(args.config, args.model_path)
    elif args.command == "backtest":
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        exec_cfg = cfg.get("execution", {})
        if args.cost_bps is not None:
            exec_cfg.setdefault("transaction_costs", {})
            exec_cfg["transaction_costs"].update(
                {"enabled": True, "mode": "bps", "value": args.cost_bps}
            )
        if args.cost_fixed is not None:
            exec_cfg.setdefault("transaction_costs", {})
            exec_cfg["transaction_costs"].update(
                {"enabled": True, "mode": "fixed", "value": args.cost_fixed}
            )
        if args.slippage_bps is not None:
            exec_cfg.setdefault("slippage", {})
            exec_cfg["slippage"].update(
                {"enabled": True, "mode": "bps", "value": args.slippage_bps}
            )
        if args.slippage_fixed is not None:
            exec_cfg.setdefault("slippage", {})
            exec_cfg["slippage"].update(
                {"enabled": True, "mode": "fixed", "value": args.slippage_fixed}
            )
        if args.liquidity_max_participation is not None:
            exec_cfg.setdefault("liquidity", {})
            exec_cfg["liquidity"].update(
                {"enabled": True, "max_participation": args.liquidity_max_participation}
            )
        df = pd.read_csv(cfg["data_path"])
        result = simulate_trades(df, execution=exec_cfg)
        metrics = compute_metrics(result)
        print(metrics)
    elif args.command == "live-trade":
        import asyncio

        asyncio.run(run_live_pipeline(args.config, args.url))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
