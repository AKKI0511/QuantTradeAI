import pandas as pd
import pytest

from quanttradeai.backtest import backtester
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.utils.config_schemas import ModelConfigSchema
from quanttradeai.utils.impact_loader import merge_execution_with_impact


def test_merge_execution_with_impact_prioritizes_overrides():
    exec_cfg = {"impact": {"beta": 0.2, "enabled": False}}
    defaults = {"equities": {"alpha": 0.1, "beta": 0.05, "model": "linear"}}

    merged = merge_execution_with_impact(exec_cfg, defaults, "equities")

    assert merged["impact"]["alpha"] == 0.1
    assert merged["impact"]["beta"] == 0.2
    assert merged["impact"]["model"] == "linear"
    assert merged["impact"]["enabled"] is False


def test_unknown_asset_class_rejected_against_impact_config():
    cfg = {
        "data": {
            "symbols": [
                {"ticker": "ABC", "asset_class": "crypto"},
            ],
            "start_date": "2020-01-01",
            "end_date": "2020-02-01",
        }
    }

    with pytest.raises(ValueError, match="impact_config.yaml"):
        ModelConfigSchema(**cfg)


def test_simulate_trades_accepts_per_symbol_execution(monkeypatch):
    calls: dict[str, dict] = {}

    def fake_sim(
        data: pd.DataFrame,
        stop_loss_pct=None,
        take_profit_pct=None,
        execution=None,
        drawdown_guard=None,
    ) -> pd.DataFrame:
        calls[data.attrs.get("symbol", "")] = execution
        return pd.DataFrame(
            {
                "strategy_return": [0.0 for _ in range(len(data))],
                "equity_curve": [1.0 for _ in range(len(data))],
            },
            index=data.index,
        )

    monkeypatch.setattr(backtester, "_simulate_single", fake_sim)

    base_df = pd.DataFrame(
        {"Close": [10.0, 11.0], "Volume": [1_000, 1_100], "label": [1, 0]},
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )
    df_a = base_df.copy()
    df_a.attrs["symbol"] = "AAA"
    df_b = base_df.copy()
    df_b.attrs["symbol"] = "BBB"

    exec_map = {
        "AAA": {"impact": {"alpha": 0.1, "beta": 0.2}},
        "BBB": {"impact": {"alpha": 0.3, "beta": 0.4}},
    }

    backtester.simulate_trades(
        {"AAA": df_a, "BBB": df_b},
        execution=exec_map,
        portfolio=PortfolioManager(capital=100_000),
    )

    assert calls == exec_map

