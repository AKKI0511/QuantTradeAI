import logging
from pathlib import Path
from data.loader import DataLoader
from data.processor import DataProcessor
from models.classifier import MomentumClassifier
from sklearn.model_selection import train_test_split
import yaml
import json
from datetime import datetime
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories(cache_dir: str = "data/raw"):
    """Create necessary directories if they don't exist."""
    dirs = [
        cache_dir,
        "data/processed",
        "models/trained",
        "models/experiments",
        "reports/figures",
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def run_pipeline(config_path: str = "config/model_config.yaml"):
    """Run the complete trading strategy pipeline."""

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

            # 4. Split Data
            X, y = model.prepare_data(df_labeled)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
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
    """Fetch data using DataLoader and store it in the cache directory."""
    data_loader = DataLoader(config_path)
    data = data_loader.fetch_data(refresh=refresh)
    data_loader.save_data(data)


def evaluate_model(config_path: str, model_path: str) -> None:
    """Load a saved model and evaluate it on the configured dataset."""
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

    args = parser.parse_args()

    if args.command == "fetch-data":
        fetch_data_only(args.config, args.refresh)
    elif args.command == "train":
        run_pipeline(args.config)
    elif args.command == "evaluate":
        evaluate_model(args.config, args.model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
