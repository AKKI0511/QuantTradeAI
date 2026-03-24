from quanttradeai.utils.project_config import compile_research_runtime_configs
from quanttradeai.utils.project_runtime import project_to_runtime_configs


def test_project_runtime_honors_technical_feature_params():
    project_cfg = {
        "data": {"symbols": ["AAPL"]},
        "research": {"enabled": True},
        "features": {
            "definitions": [
                {
                    "name": "tech_custom",
                    "type": "technical",
                    "params": {
                        "price_features": ["close_to_open"],
                        "rsi_period": 21,
                        "atr_periods": [10, 20],
                        "bollinger_bands": {"period": 30, "std_dev": 2.5},
                        "keltner_channels": {"periods": [15], "atr_multiple": 1.8},
                    },
                }
            ]
        },
    }

    _, runtime_features_cfg = project_to_runtime_configs(project_cfg)

    assert runtime_features_cfg["price_features"] == ["close_to_open"]
    assert runtime_features_cfg["momentum_features"]["rsi_period"] == 21
    assert runtime_features_cfg["volatility_features"] == {
        "atr_periods": [10, 20],
        "bollinger_bands": {"period": 30, "std_dev": 2.5},
        "keltner_channels": {"periods": [15], "atr_multiple": 1.8},
    }


def test_project_runtime_matches_canonical_compiler_for_agent_flows():
    project_cfg = {
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "test_start": "2020-10-01",
            "test_end": "2020-12-31",
        },
        "features": {
            "definitions": [
                {"name": "rsi_14", "type": "technical", "params": {"period": 14}},
                {
                    "name": "volume_spike_20",
                    "type": "custom",
                    "params": {"kind": "volume_momentum", "window": 20},
                },
            ]
        },
        "research": {
            "enabled": False,
            "evaluation": {"use_configured_test_window": False},
            "backtest": {"costs": {"enabled": True, "bps": 7}},
        },
        "trading": {"initial_capital": 250000, "max_risk_per_trade": 0.03},
    }

    runtime_model_cfg, runtime_features_cfg = project_to_runtime_configs(
        project_cfg,
        require_research=False,
    )
    expected_model_cfg, expected_features_cfg, _ = compile_research_runtime_configs(
        project_cfg,
        require_research=False,
    )

    assert runtime_model_cfg == expected_model_cfg
    assert runtime_features_cfg == expected_features_cfg
    assert runtime_model_cfg["data"]["test_start"] is None
    assert runtime_model_cfg["data"]["test_end"] is None
