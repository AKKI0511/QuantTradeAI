import pandas as pd
from quanttradeai.backtest.backtester import simulate_trades, compute_metrics


def make_df(prices, labels, volumes=None):
    idx = pd.date_range("2021-01-01", periods=len(prices), freq="D")
    data = {"Close": prices, "label": labels}
    if volumes is not None:
        data["Volume"] = volumes
    return pd.DataFrame(data, index=idx)


def test_linear_impact_applied_to_trades():
    df = make_df([100, 101], [1, 0], volumes=[1000, 1000])
    res = simulate_trades(
        df,
        execution={
            "impact": {
                "enabled": True,
                "model": "linear",
                "alpha": 0.5,
                "beta": 0.0,
                "average_daily_volume": 1000,
                "spread": 0.02,
            }
        },
    )
    ledger = res.attrs["ledger"]
    assert "impact_cost" in ledger.columns
    assert ledger["impact_cost"].iloc[0] > 0
    metrics = compute_metrics(res)
    assert metrics["total_impact_cost"] > 0
