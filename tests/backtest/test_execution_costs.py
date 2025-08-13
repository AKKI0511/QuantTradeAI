import pandas as pd
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics


def make_df(prices, labels, volumes=None):
    index = pd.date_range("2020-01-01", periods=len(prices), freq="D")
    data = {"Close": prices, "label": labels}
    if volumes is not None:
        data["Volume"] = volumes
    return pd.DataFrame(data, index=index)


def test_slippage_bps_applies_side_correctly():
    df = make_df([100, 101], [1, 0])
    res = simulate_trades(
        df,
        execution={
            "slippage": {"enabled": True, "mode": "bps", "value": 10},
        },
    )
    ledger = res.attrs["ledger"]
    assert ledger.iloc[0]["side"] == "buy"
    assert ledger.iloc[0]["fill_price"] > ledger.iloc[0]["reference_price"]
    assert ledger.iloc[1]["side"] == "sell"
    assert ledger.iloc[1]["fill_price"] < ledger.iloc[1]["reference_price"]


def test_transaction_costs_bps_and_fixed_modes():
    df = make_df([100, 101], [1, 0])
    res_bps = simulate_trades(
        df,
        execution={
            "transaction_costs": {"enabled": True, "mode": "bps", "value": 100},
        },
    )
    cost_bps = res_bps.attrs["ledger"]["transaction_cost"].sum()
    assert abs(cost_bps - (100 * 0.01 + 101 * 0.01)) < 1e-6
    res_fixed = simulate_trades(
        df,
        execution={
            "transaction_costs": {
                "enabled": True,
                "mode": "fixed",
                "value": 1,
                "apply_on": "shares",
            }
        },
    )
    cost_fixed = res_fixed.attrs["ledger"]["transaction_cost"].sum()
    assert abs(cost_fixed - 2.0) < 1e-6


def test_liquidity_cap_partial_fills_and_carryover():
    df = make_df([100, 100, 100], [1, 1, 1], volumes=[1, 1, 1])
    res = simulate_trades(
        df,
        execution={
            "liquidity": {"enabled": True, "max_participation": 0.5},
        },
    )
    ledger = res.attrs["ledger"]
    assert ledger.iloc[0]["qty"] == 0.5
    assert ledger.iloc[1]["qty"] == 0.5


def test_metrics_net_vs_gross_consistency():
    df = make_df([100, 101, 102], [1, 0, 0])
    res = simulate_trades(
        df,
        execution={
            "transaction_costs": {
                "enabled": True,
                "mode": "fixed",
                "value": 1,
                "apply_on": "shares",
            },
            "slippage": {"enabled": True, "mode": "bps", "value": 100},
        },
    )
    metrics = compute_metrics(res)
    diff = metrics["gross_pnl"] - metrics["net_pnl"]
    total = metrics["total_costs"] + metrics["total_slippage_cost"]
    assert abs(diff - total) < 1e-3
