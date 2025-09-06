import pandas as pd
import pytest

from quanttradeai.backtest.backtester import simulate_trades
from quanttradeai.backtest.engine import BacktestEngine
from quanttradeai.trading.portfolio import PortfolioManager
from quanttradeai.trading.risk_manager import RiskManager
from quanttradeai.utils.config_schemas import BorrowFeeConfig, MarketImpactConfig


def test_borrow_fee_applied():
    df = pd.DataFrame({"Close": [10, 9, 8], "label": [-1, -1, 0]})
    execution = {"borrow_fee": {"enabled": True, "rate_bps": 100}}
    res = simulate_trades(df, execution=execution)
    ledger = res.attrs["ledger"]
    borrow_entries = ledger[ledger["side"] == "borrow_fee"]
    assert borrow_entries["borrow_fee"].sum() > 0
    # first bar net return should reflect borrow cost 0.01
    assert pytest.approx(res["strategy_return"].iloc[0], rel=1e-6) == 0.09


def test_intrabar_fill_vwap():
    ticks = [[{"price": 10.0, "volume": 5}, {"price": 10.2, "volume": 5}], []]
    df = pd.DataFrame(
        {
            "Close": [10, 10],
            "label": [10, 0],
            "ticks": ticks,
            "Volume": [10, 10],
        }
    )
    execution = {"intrabar": {"enabled": True}}
    res = simulate_trades(df, execution=execution)
    ledger = res.attrs["ledger"]
    trade = ledger.iloc[0]
    assert pytest.approx(trade["fill_price"], rel=1e-6) == 10.1
    assert trade["tick_fills"] == 2
    assert pytest.approx(trade["liquidity_participation"], rel=1e-6) == 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        BorrowFeeConfig(enabled=True, rate_bps=-1)
    with pytest.raises(ValueError):
        MarketImpactConfig(liquidity_scale=-1)


def test_backtest_engine_wrapper():
    df = pd.DataFrame({"Close": [10, 11], "label": [1, 0]})
    pm = PortfolioManager(capital=1000)
    rm = RiskManager()
    engine = BacktestEngine(portfolio=pm, risk_manager=rm)
    res = engine.run(df)
    assert "equity_curve" in res.columns


def test_synthetic_intrabar_generation():
    df = pd.DataFrame(
        {
            "Close": [10, 10],
            "label": [5, 0],
            "Volume": [10, 10],
        }
    )
    execution = {
        "intrabar": {"enabled": True, "synthetic_ticks": 5, "volatility": 0.1}
    }
    res = simulate_trades(df, execution=execution)
    trade = res.attrs["ledger"].iloc[0]
    assert trade["tick_fills"] == 3


def test_limit_and_stop_orders():
    df = pd.DataFrame(
        {
            "Close": [10, 9, 8],
            "label": [5, 5, -5],
            "Volume": [10, 10, 10],
            "order_type": ["limit", "limit", "stop"],
            "limit_price": [9.5, 9.5, None],
            "stop_price": [None, None, 8.5],
        }
    )
    execution = {
        "liquidity": {
            "enabled": True,
            "order_book_depth": 20,
            "max_participation": 1.0,
        }
    }
    res = simulate_trades(df, execution=execution)
    ledger = res.attrs["ledger"]
    # limit order partially fills on second bar
    assert ledger.iloc[0]["order_type"] == "limit"
    assert pytest.approx(ledger.iloc[0]["qty"], rel=1e-6) == 5.0
    # stop order triggers on third bar with partial fill
    assert ledger.iloc[1]["order_type"] == "stop"
    assert pytest.approx(ledger.iloc[1]["qty"], rel=1e-6) == 2.5
    assert ledger.iloc[1]["timestamp"] == df.index[2]


def test_dynamic_spread_asymmetric_impact():
    df = pd.DataFrame(
        {
            "Close": [10, 10, 10, 10],
            "label": [1, 1, -1, 0],
            "Volume": [100, 100, 100, 100],
            "Volatility": [0.1, 0.1, 0.1, 0.1],
        }
    )
    execution = {
        "impact": {
            "enabled": True,
            "model": "linear",
            "alpha_buy": 0.1,
            "alpha_sell": 0.2,
            "spread_model": {"base": 0.01, "vol_coeff": 0.5},
            "average_daily_volume": 100,
        }
    }
    res = simulate_trades(df, execution=execution)
    ledger = res.attrs["ledger"]
    buy = ledger.iloc[0]
    sell = ledger.iloc[1]
    buy_per = buy["impact_temp_cost"] / buy["qty"]
    sell_per = sell["impact_temp_cost"] / sell["qty"]
    assert sell_per > buy_per
    expected_spread = 0.01 + 0.5 * 0.1
    assert buy["impact_cost"] >= expected_spread / 2
