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
    assert runtime_features_cfg["volatility_features"] == [
        {
            "atr_periods": [10, 20],
            "bollinger_bands": {"period": 30, "std_dev": 2.5},
            "keltner_channels": {"periods": [15], "atr_multiple": 1.8},
        }
    ]
