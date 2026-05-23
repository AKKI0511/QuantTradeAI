import yaml
import pytest

from quanttradeai.cli import PROJECT_TEMPLATES
from quanttradeai.utils.sweeps import (
    expand_agent_backtest_sweep,
    expand_research_sweep,
)


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


def test_expand_research_sweep_builds_cartesian_variants_and_feature_params():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["research"]["backtest"]["costs"] = {"enabled": True, "bps": 5}
    project_cfg["sweeps"] = [
        {
            "name": "rsi_research_grid",
            "kind": "research_run",
            "parameters": [
                {"path": "research.labels.horizon", "values": [3, 5]},
                {"path": "features.rsi_14.params.period", "values": [7, 14]},
                {"path": "research.backtest.costs.bps", "values": [1]},
            ],
        }
    ]

    expansion = expand_research_sweep(project_cfg, "rsi_research_grid")

    assert expansion["kind"] == "research_run"
    assert [variant["name"] for variant in expansion["variants"]] == [
        "research_lab__rsi_research_grid__horizon-3__period-7__bps-1",
        "research_lab__rsi_research_grid__horizon-3__period-14__bps-1",
        "research_lab__rsi_research_grid__horizon-5__period-7__bps-1",
        "research_lab__rsi_research_grid__horizon-5__period-14__bps-1",
    ]
    first_variant = expansion["variants"][0]["project_config"]
    assert first_variant["research"]["labels"]["horizon"] == 3
    assert first_variant["research"]["backtest"]["costs"]["bps"] == 1
    assert first_variant["features"]["definitions"][0]["params"]["period"] == 7
    assert "sweeps" not in first_variant


def test_expand_research_sweep_rejects_unknown_feature_path():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["sweeps"] = [
        {
            "name": "bad_grid",
            "kind": "research_run",
            "parameters": [{"path": "features.missing.params.period", "values": [7]}],
        }
    ]

    with pytest.raises(ValueError, match="unknown feature"):
        expand_research_sweep(project_cfg, "bad_grid")


def test_expand_research_sweep_rejects_empty_values():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["sweeps"] = [
        {
            "name": "empty_grid",
            "kind": "research_run",
            "parameters": [{"path": "research.labels.horizon", "values": []}],
        }
    ]

    with pytest.raises(ValueError, match="at least one value"):
        expand_research_sweep(project_cfg, "empty_grid")


def test_expand_research_sweep_rejects_non_scalar_leaf():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["sweeps"] = [
        {
            "name": "non_scalar_grid",
            "kind": "research_run",
            "parameters": [{"path": "data.symbols", "values": ["AAPL"]}],
        }
    ]

    with pytest.raises(ValueError, match="not supported"):
        expand_research_sweep(project_cfg, "non_scalar_grid")
