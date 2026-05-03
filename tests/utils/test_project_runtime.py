from quanttradeai.utils.project_config import (
    compile_live_position_manager_runtime_config,
    compile_live_risk_runtime_config,
    compile_live_streaming_runtime_config,
    compile_paper_streaming_runtime_config,
    compile_research_runtime_configs,
    resolve_paper_replay_window,
)
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


def test_project_runtime_compiles_named_sma_feature_periods():
    project_cfg = {
        "data": {"symbols": ["AAPL"]},
        "research": {"enabled": False},
        "features": {
            "definitions": [
                {"name": "sma_10", "type": "technical", "params": {}},
                {"name": "sma_30", "type": "technical", "params": {}},
            ]
        },
    }

    _, runtime_features_cfg = project_to_runtime_configs(
        project_cfg,
        require_research=False,
    )

    assert runtime_features_cfg["price_features"]["sma_periods"] == [10, 30]
    assert runtime_features_cfg["price_features"]["close_to_open"] is True


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


def test_compile_live_runtime_configs_emit_canonical_streaming_risk_and_position_manager():
    project_cfg = {
        "data": {
            "symbols": ["AAPL"],
            "streaming": {
                "enabled": True,
                "provider": "alpaca",
                "websocket_url": "wss://example.test/stream",
                "auth_method": "api_key",
                "symbols": ["AAPL"],
                "channels": ["trades", "quotes"],
            },
        },
        "risk": {
            "drawdown_protection": {
                "enabled": True,
                "max_drawdown_pct": 0.1,
            },
            "turnover_limits": {"daily_max": 2.0},
        },
        "position_manager": {
            "impact": {"enabled": False},
            "reconciliation": {"intraday": "1m", "daily": "1d"},
            "mode": "live",
        },
    }

    streaming_cfg = compile_live_streaming_runtime_config(project_cfg)
    risk_cfg = compile_live_risk_runtime_config(project_cfg)
    position_manager_cfg = compile_live_position_manager_runtime_config(project_cfg)

    assert streaming_cfg["streaming"]["providers"][0]["name"] == "alpaca"
    assert risk_cfg["risk_management"]["drawdown_protection"]["enabled"] is True
    assert (
        position_manager_cfg["position_manager"]["risk_management"]["turnover_limits"][
            "daily_max"
        ]
        == 2.0
    )
    assert position_manager_cfg["position_manager"]["mode"] == "live"


def test_compile_live_runtime_accepts_legacy_nested_position_manager_risk():
    project_cfg = {
        "data": {
            "symbols": ["AAPL"],
            "streaming": {
                "enabled": True,
                "provider": "alpaca",
                "websocket_url": "wss://example.test/stream",
                "symbols": ["AAPL"],
                "channels": ["trades"],
            },
        },
        "position_manager": {
            "impact": {"enabled": False},
            "reconciliation": {"intraday": "1m", "daily": "1d"},
            "mode": "live",
            "risk_management": {
                "drawdown_protection": {
                    "enabled": True,
                    "max_drawdown_pct": 0.15,
                }
            },
        },
    }

    risk_cfg = compile_live_risk_runtime_config(project_cfg)
    position_manager_cfg = compile_live_position_manager_runtime_config(project_cfg)

    assert (
        risk_cfg["risk_management"]["drawdown_protection"]["max_drawdown_pct"] == 0.15
    )
    assert (
        position_manager_cfg["position_manager"]["risk_management"][
            "drawdown_protection"
        ]["max_drawdown_pct"]
        == 0.15
    )


def test_compile_paper_runtime_config_allows_replay_without_realtime_provider():
    project_cfg = {
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "test_start": "2024-02-01",
            "test_end": "2024-02-29",
            "streaming": {
                "enabled": True,
                "symbols": ["AAPL"],
                "channels": ["trades"],
                "replay": {
                    "enabled": True,
                    "pace_delay_ms": 0,
                },
            },
        }
    }

    streaming_cfg = compile_paper_streaming_runtime_config(project_cfg)

    assert "providers" not in streaming_cfg["streaming"]
    assert streaming_cfg["streaming"]["replay"] == {
        "enabled": True,
        "start_date": "2024-02-01",
        "end_date": "2024-02-29",
        "pace_delay_ms": 0,
    }


def test_resolve_paper_replay_window_prefers_explicit_dates_then_test_then_data():
    project_cfg = {
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "test_start": "2024-02-01",
            "test_end": "2024-02-29",
            "streaming": {
                "enabled": True,
                "symbols": ["AAPL"],
                "channels": ["trades"],
                "replay": {
                    "enabled": True,
                    "start_date": "2024-02-10",
                    "end_date": "2024-02-12",
                    "pace_delay_ms": 25,
                },
            },
        }
    }

    explicit_window = resolve_paper_replay_window(project_cfg)
    assert explicit_window is not None
    assert explicit_window.start_date == "2024-02-10"
    assert explicit_window.end_date == "2024-02-12"
    assert explicit_window.pace_delay_ms == 25

    project_cfg["data"]["streaming"]["replay"].pop("start_date")
    project_cfg["data"]["streaming"]["replay"].pop("end_date")
    test_window = resolve_paper_replay_window(project_cfg)
    assert test_window is not None
    assert test_window.start_date == "2024-02-01"
    assert test_window.end_date == "2024-02-29"

    project_cfg["data"].pop("test_start")
    project_cfg["data"].pop("test_end")
    data_window = resolve_paper_replay_window(project_cfg)
    assert data_window is not None
    assert data_window.start_date == "2024-01-01"
    assert data_window.end_date == "2024-03-31"
