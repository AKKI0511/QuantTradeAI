import os
import tempfile

import pandas as pd
import yaml
from typer.testing import CliRunner
from unittest.mock import patch


def test_cli_fetch_data_smoke():
    from quanttradeai.cli import app

    runner = CliRunner()
    with patch("quanttradeai.cli.fetch_data_only") as mock_fn:
        result = runner.invoke(app, ["fetch-data", "-c", "config/model_config.yaml", "--refresh"])
        assert result.exit_code == 0
        mock_fn.assert_called_once()


def test_cli_train_smoke():
    from quanttradeai.cli import app

    runner = CliRunner()
    with patch("quanttradeai.cli.run_pipeline") as mock_fn:
        result = runner.invoke(app, ["train", "-c", "config/model_config.yaml"])
        assert result.exit_code == 0
        mock_fn.assert_called_once()


def test_cli_evaluate_smoke():
    from quanttradeai.cli import app

    runner = CliRunner()
    with patch("quanttradeai.cli.evaluate_model") as mock_fn:
        result = runner.invoke(app, ["evaluate", "-c", "config/model_config.yaml", "-m", "models/experiments/foo/AAPL"])
        assert result.exit_code == 0
        mock_fn.assert_called_once()


def test_cli_backtest_smoke_with_overrides():
    from quanttradeai.cli import app

    # Prepare a minimal backtest config and data CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "backtest.yaml")
        data_path = os.path.join(tmpdir, "data.csv")
        # The backtest command reads the CSV and then calls simulate_trades/compute_metrics
        pd.DataFrame({"Close": [1, 1.01, 1.02], "label": [0, 1, 0]}).to_csv(data_path, index=False)
        with open(cfg_path, "w") as f:
            yaml.safe_dump({"execution": {}, "data_path": data_path}, f)

        runner = CliRunner()
        with patch("quanttradeai.cli.simulate_trades") as mock_sim, \
             patch("quanttradeai.cli.compute_metrics", return_value={"sharpe_ratio": 1.0}):
            mock_sim.return_value = pd.DataFrame({"strategy_return": [0, 0], "equity_curve": [1, 1.0]})
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "-c",
                    cfg_path,
                    "--cost-bps",
                    "5",
                    "--slippage-fixed",
                    "0.01",
                    "--liquidity-max-participation",
                    "0.1",
                ],
            )
            assert result.exit_code == 0
            assert "sharpe_ratio" in result.stdout


def test_cli_backtest_model_smoke():
    from quanttradeai.cli import app

    runner = CliRunner()
    with patch("quanttradeai.cli.run_model_backtest", return_value={"AAA": {"metrics": {"sharpe_ratio": 2.0}}}):
        result = runner.invoke(
            app,
            [
                "backtest-model",
                "-m",
                "models/experiments/foo/AAPL",
                "-c",
                "config/model_config.yaml",
            ],
        )
        assert result.exit_code == 0
        assert "sharpe_ratio" in result.stdout

