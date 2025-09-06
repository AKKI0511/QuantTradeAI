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
