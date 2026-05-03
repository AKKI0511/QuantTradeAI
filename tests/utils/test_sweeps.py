import yaml

from quanttradeai.cli import PROJECT_TEMPLATES
from quanttradeai.utils.sweeps import expand_agent_backtest_sweep


def test_expand_agent_backtest_sweep_builds_cartesian_variants_with_deterministic_names():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["rule-agent"], sort_keys=False)
    )
    project_cfg["sweeps"] = [
        {
            "name": "rsi_threshold_grid",
            "kind": "agent_backtest",
            "agent": "rsi_reversion",
            "parameters": [
                {"path": "rule.buy_below", "values": [25.0, 30.0]},
                {"path": "rule.sell_above", "values": [70.0, 75.0]},
            ],
        }
    ]

    expansion = expand_agent_backtest_sweep(project_cfg, "rsi_threshold_grid")

    assert expansion["base_agent_name"] == "rsi_reversion"
    assert [variant["name"] for variant in expansion["variants"]] == [
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-70_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-75_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-70_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-75_0",
    ]
    assert expansion["variants"][0]["project_config"]["agents"][0]["name"] == (
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-70_0"
    )
    assert expansion["variants"][0]["project_config"]["agents"][0]["rule"] == {
        "preset": "rsi_threshold",
        "feature": "rsi_14",
        "buy_below": 25.0,
        "sell_above": 70.0,
    }
    assert "sweeps" not in expansion["variants"][0]["project_config"]


def test_strategy_lab_sma_risk_sweep_expands_against_sma_agent():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["strategy-lab"], sort_keys=False)
    )

    expansion = expand_agent_backtest_sweep(project_cfg, "sma_risk_grid")

    assert expansion["base_agent_name"] == "sma_trend"
    assert [variant["name"] for variant in expansion["variants"]] == [
        "sma_trend__sma_risk_grid__max_position_pct-0_03",
        "sma_trend__sma_risk_grid__max_position_pct-0_05",
        "sma_trend__sma_risk_grid__max_position_pct-0_07",
    ]
    assert expansion["variants"][0]["agent_config"]["rule"] == {
        "preset": "sma_crossover",
        "fast_feature": "sma_20",
        "slow_feature": "sma_50",
    }
    assert expansion["variants"][0]["agent_config"]["risk"] == {
        "max_position_pct": 0.03
    }
