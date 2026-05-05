import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from quanttradeai.cli import app


runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_run_bundle(
    root: Path,
    *,
    relative_dir: str,
    summary_payload: dict,
    metrics_payload: dict | None = None,
    resolved_config_payload: dict | None = None,
) -> None:
    run_dir = root / Path(relative_dir)
    summary = dict(summary_payload)
    if resolved_config_payload is not None:
        artifacts = dict(summary.get("artifacts") or {})
        artifacts.setdefault("resolved_project_config", "resolved_project_config.yaml")
        summary["artifacts"] = artifacts
        _write_yaml(run_dir / "resolved_project_config.yaml", resolved_config_payload)
    _write_json(run_dir / "summary.json", summary)
    if metrics_payload is not None:
        _write_json(run_dir / "metrics.json", metrics_payload)


def _research_project_config(
    *,
    timeframe: str = "1d",
    horizon: int = 5,
    feature_names: list[str] | None = None,
    model_family: str = "voting",
    trials: int = 50,
    bps: int = 5,
) -> dict:
    return {
        "project": {"name": "research_lab", "profile": "research"},
        "profiles": {
            "research": {"mode": "research"},
            "paper": {"mode": "paper"},
            "live": {"mode": "live"},
        },
        "data": {
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "timeframe": timeframe,
            "test_start": "2020-09-01",
            "test_end": "2020-12-31",
        },
        "features": {
            "definitions": [
                {"name": name, "type": "technical", "params": {"period": 14}}
                for name in (feature_names or ["rsi_14"])
            ]
        },
        "research": {
            "enabled": True,
            "labels": {
                "type": "forward_return",
                "horizon": horizon,
                "buy_threshold": 0.01,
                "sell_threshold": -0.01,
            },
            "model": {
                "kind": "classifier",
                "family": model_family,
                "tuning": {"enabled": True, "trials": trials},
            },
            "evaluation": {
                "split": "time_aware",
                "use_configured_test_window": True,
            },
            "backtest": {"costs": {"enabled": True, "bps": bps}},
        },
        "agents": [],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    }


def _agent_project_config(
    *,
    agent_name: str,
    llm_model: str = "gpt-5.3",
    max_position_pct: float = 0.05,
    tools: list[str] | None = None,
    context_features: list[str] | None = None,
) -> dict:
    return {
        "project": {"name": "agent_lab", "profile": "paper"},
        "profiles": {
            "research": {"mode": "research"},
            "paper": {"mode": "paper"},
            "live": {"mode": "live"},
        },
        "data": {
            "symbols": ["AAPL"],
            "start_date": "2022-01-01",
            "end_date": "2024-12-31",
            "timeframe": "1d",
            "test_start": "2024-09-01",
            "test_end": "2024-12-31",
        },
        "features": {
            "definitions": [
                {"name": name, "type": "technical", "params": {"period": 14}}
                for name in (context_features or ["rsi_14"])
            ]
        },
        "research": {
            "enabled": False,
            "labels": {},
            "model": {},
            "evaluation": {},
            "backtest": {},
        },
        "agents": [
            {
                "name": agent_name,
                "kind": "llm",
                "mode": "paper",
                "llm": {
                    "provider": "openai",
                    "model": llm_model,
                    "prompt_file": "prompts/breakout.md",
                },
                "context": {
                    "features": list(context_features or ["rsi_14"]),
                    "positions": True,
                    "risk_state": True,
                },
                "tools": list(tools or ["get_quote", "place_order"]),
                "risk": {"max_position_pct": max_position_pct},
            }
        ],
        "deployment": {"target": "docker-compose", "mode": "paper"},
    }


def _seed_runs(root: Path) -> None:
    _write_run_bundle(
        root,
        relative_dir="research/20260101_000000_research_lab",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "research_lab",
            "project_name": "research_lab",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_research_lab",
            "run_dir": "runs/research/20260101_000000_research_lab",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {
                "AAPL": {"accuracy": 0.70, "f1": 0.50},
                "MSFT": {"accuracy": 0.90, "f1": 0.70},
            },
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.00, "net_pnl": 0.10},
                "MSFT": {"net_sharpe": 2.00, "net_pnl": 0.30},
            },
        },
    )
    _write_run_bundle(
        root,
        relative_dir="agent/backtest/20260101_010000_breakout_gpt",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 11,
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/backtest/20260101_010000_breakout_gpt",
            "run_dir": "runs/agent/backtest/20260101_010000_breakout_gpt",
        },
        metrics_payload={
            "net_sharpe": 1.25,
            "net_pnl": 0.18,
            "net_mdd": -0.07,
        },
    )
    _write_run_bundle(
        root,
        relative_dir="20251231_230000_legacy_agent",
        summary_payload={
            "agent_name": "legacy_agent",
            "mode": "backtest",
            "status": "failed",
            "symbols": ["TSLA"],
            "timestamps": {"started_at": "2025-12-31T23:00:00+00:00"},
            "artifacts": {},
            "warnings": ["legacy"],
        },
    )


def test_runs_list_human_output_includes_mixed_run_types(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(app, ["runs", "list"])

    assert result.exit_code == 0, result.stdout
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("RUN_ID")
    assert "breakout_gpt" in lines[1]
    assert "research_lab" in result.stdout
    assert "legacy_agent" in result.stdout


def test_runs_list_json_output_returns_normalized_records(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(app, ["runs", "list", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [record["name"] for record in payload] == [
        "breakout_gpt",
        "research_lab",
        "legacy_agent",
    ]
    assert payload[0]["run_id"] == "agent/backtest/20260101_010000_breakout_gpt"
    assert payload[-1]["run_type"] == "agent"
    assert "scoreboard" not in payload[0]


def test_runs_list_reports_empty_state(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["runs", "list"])

    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "No runs found."


def test_runs_list_filters_by_type_status_and_limit(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--type",
            "agent",
            "--status",
            "failed",
            "--limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["run_id"] == "20251231_230000_legacy_agent"
    assert payload[0]["run_type"] == "agent"
    assert payload[0]["mode"] == "backtest"
    assert payload[0]["name"] == "legacy_agent"
    assert payload[0]["status"] == "failed"
    assert payload[0]["symbols"] == ["TSLA"]
    assert payload[0]["warnings"] == ["legacy"]
    assert Path(payload[0]["run_dir"]).name == "20251231_230000_legacy_agent"


def test_runs_list_surfaces_live_agent_runs_and_filters_by_live_mode(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _seed_runs(runs_root)
    _write_run_bundle(
        runs_root,
        relative_dir="agent/live/20260101_020000_breakout_live",
        summary_payload={
            "run_type": "agent",
            "mode": "live",
            "name": "breakout_live",
            "agent_name": "breakout_live",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T02:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/live/20260101_020000_breakout_live",
            "run_dir": "runs/agent/live/20260101_020000_breakout_live",
        },
        metrics_payload={
            "status": "available",
            "total_pnl": 120.0,
            "portfolio_value": 100120.0,
            "execution_count": 3,
            "decision_count": 5,
            "risk_status": {"status": "ok"},
        },
    )

    result = runner.invoke(app, ["runs", "list", "--mode", "live", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["run_id"] == "agent/live/20260101_020000_breakout_live"
    assert payload[0]["mode"] == "live"
    assert payload[0]["name"] == "breakout_live"


def test_runs_list_scoreboard_for_research_runs_uses_research_columns(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_research_lab",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "research_lab",
            "project_name": "research_lab",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.8, "f1": 0.6}},
            "backtest_metrics_by_symbol": {"AAPL": {"net_sharpe": 1.2, "net_pnl": 0.3}},
        },
    )

    result = runner.invoke(app, ["runs", "list", "--type", "research", "--scoreboard"])

    assert result.exit_code == 0, result.stdout
    header = result.stdout.strip().splitlines()[0]
    assert "ACC" in header
    assert "F1" in header
    assert "NET_SHARPE" in header
    assert "NET_PNL" in header
    assert "research_lab" in result.stdout


def test_runs_list_scoreboard_for_agent_backtests_uses_backtest_columns(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260101_010000_breakout_gpt",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 11,
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "net_sharpe": 1.25,
            "net_pnl": 0.18,
            "net_mdd": -0.07,
        },
    )

    result = runner.invoke(
        app,
        ["runs", "list", "--type", "agent", "--mode", "backtest", "--scoreboard"],
    )

    assert result.exit_code == 0, result.stdout
    header = result.stdout.strip().splitlines()[0]
    assert "NET_SHARPE" in header
    assert "NET_PNL" in header
    assert "NET_MDD" in header
    assert "DECISIONS" in header
    assert "breakout_gpt" in result.stdout


def test_runs_list_scoreboard_for_streaming_agents_uses_streaming_columns(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="agent/paper/20260101_010000_breakout_gpt",
        summary_payload={
            "run_type": "agent",
            "mode": "paper",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "status": "available",
            "total_pnl": 125.0,
            "portfolio_value": 100125.0,
            "execution_count": 4,
            "decision_count": 7,
            "risk_status": {"status": "ok"},
        },
    )
    _write_run_bundle(
        runs_root,
        relative_dir="agent/live/20260101_020000_breakout_live",
        summary_payload={
            "run_type": "agent",
            "mode": "live",
            "name": "breakout_live",
            "agent_name": "breakout_live",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T02:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "status": "available",
            "total_pnl": 220.0,
            "portfolio_value": 100220.0,
            "execution_count": 6,
            "decision_count": 8,
            "risk_status": {"status": "warning"},
        },
    )

    result = runner.invoke(
        app,
        ["runs", "list", "--type", "agent", "--scoreboard", "--sort-by", "total_pnl"],
    )

    assert result.exit_code == 0, result.stdout
    header = result.stdout.strip().splitlines()[0]
    assert "TOTAL_PNL" in header
    assert "PORTFOLIO" in header
    assert "EXEC" in header
    assert "DECISIONS" in header
    assert "RISK" in header


def test_runs_list_scoreboard_for_mixed_results_uses_safe_superset(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(app, ["runs", "list", "--scoreboard"])

    assert result.exit_code == 0, result.stdout
    header = result.stdout.strip().splitlines()[0]
    assert "PRIMARY" in header
    assert "PNL" in header
    assert "SHARPE" in header
    assert "EXEC" in header
    assert "DECISIONS" in header
    assert "RISK" in header


def test_runs_list_scoreboard_sorts_by_net_sharpe_descending(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260101_000000_low_sharpe",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "low_sharpe",
            "agent_name": "low_sharpe",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 2,
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={"net_sharpe": 0.5, "net_pnl": 0.1, "net_mdd": -0.1},
    )
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260101_010000_high_sharpe",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "high_sharpe",
            "agent_name": "high_sharpe",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 4,
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={"net_sharpe": 2.5, "net_pnl": 0.3, "net_mdd": -0.05},
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--type",
            "agent",
            "--mode",
            "backtest",
            "--scoreboard",
            "--sort-by",
            "net_sharpe",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [record["name"] for record in payload] == ["high_sharpe", "low_sharpe"]


def test_runs_list_scoreboard_sorts_by_total_pnl_descending(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="agent/live/20260101_000000_lower_pnl",
        summary_payload={
            "run_type": "agent",
            "mode": "live",
            "name": "lower_pnl",
            "agent_name": "lower_pnl",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "status": "available",
            "total_pnl": 80.0,
            "portfolio_value": 100080.0,
            "execution_count": 2,
            "decision_count": 3,
            "risk_status": {"status": "ok"},
        },
    )
    _write_run_bundle(
        runs_root,
        relative_dir="agent/live/20260101_010000_higher_pnl",
        summary_payload={
            "run_type": "agent",
            "mode": "live",
            "name": "higher_pnl",
            "agent_name": "higher_pnl",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
        },
        metrics_payload={
            "status": "available",
            "total_pnl": 180.0,
            "portfolio_value": 100180.0,
            "execution_count": 6,
            "decision_count": 5,
            "risk_status": {"status": "warning"},
        },
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--type",
            "agent",
            "--mode",
            "live",
            "--scoreboard",
            "--sort-by",
            "total_pnl",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [record["name"] for record in payload] == ["higher_pnl", "lower_pnl"]


def test_runs_list_scoreboard_sorts_by_started_at_when_requested(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(
        app,
        ["runs", "list", "--scoreboard", "--sort-by", "started_at", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [record["name"] for record in payload] == [
        "breakout_gpt",
        "research_lab",
        "legacy_agent",
    ]


def test_runs_list_scoreboard_json_includes_additive_scoreboard_payload(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _seed_runs(Path("runs"))

    result = runner.invoke(app, ["runs", "list", "--scoreboard", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "scoreboard" in payload[0]
    assert payload[0]["scoreboard"]["net_sharpe"] == 1.25
    assert "scoreboard" in payload[1]
    assert payload[1]["scoreboard"]["accuracy"] == 0.8


def test_runs_list_ignores_batch_directory_without_summary(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _seed_runs(runs_root)

    batch_root = (
        runs_root / "agent" / "batches" / "20260101_000000_multi_agent_backtest"
    )
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "results.json").write_text("{}", encoding="utf-8")
    (batch_root / "scoreboard.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["runs", "list", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert all("batches" not in record["run_dir"] for record in payload)


def test_runs_list_compare_renders_research_metrics_and_config_differences(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_alpha",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "alpha",
            "project_name": "alpha",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_alpha",
            "run_dir": "runs/research/20260101_000000_alpha",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
            },
        },
        resolved_config_payload=_research_project_config(
            timeframe="1d",
            horizon=5,
            feature_names=["rsi_14"],
        ),
    )
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260102_000000_beta",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "beta",
            "project_name": "beta",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-02T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260102_000000_beta",
            "run_dir": "runs/research/20260102_000000_beta",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.82, "f1": 0.61}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.40, "net_pnl": 0.22}
            },
        },
        resolved_config_payload=_research_project_config(
            timeframe="1h",
            horizon=10,
            feature_names=["rsi_14", "macd_fast"],
        ),
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "research/20260101_000000_alpha",
            "--compare",
            "research/20260102_000000_beta",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Run comparison: research" in result.stdout
    assert "Metrics:" in result.stdout
    assert "ACC" in result.stdout
    assert "NET_SHARPE" in result.stdout
    assert "Config differences:" in result.stdout
    assert "data.timeframe" in result.stdout
    assert "labels.horizon" in result.stdout
    assert "features.definitions" in result.stdout
    assert "Artifacts:" in result.stdout
    assert "resolved_project_config.yaml" in result.stdout


def test_runs_list_compare_renders_agent_backtest_differences(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260101_010000_breakout_gpt",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 11,
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/backtest/20260101_010000_breakout_gpt",
            "run_dir": "runs/agent/backtest/20260101_010000_breakout_gpt",
        },
        metrics_payload={
            "net_sharpe": 1.25,
            "net_pnl": 0.18,
            "net_mdd": -0.07,
        },
        resolved_config_payload=_agent_project_config(
            agent_name="breakout_gpt",
            llm_model="gpt-5.3",
            max_position_pct=0.05,
            tools=["get_quote", "place_order"],
            context_features=["rsi_14"],
        ),
    )
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260102_010000_breakout_gpt_alt",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt_alt",
            "agent_name": "breakout_gpt_alt",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 9,
            "timestamps": {"started_at": "2026-01-02T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/backtest/20260102_010000_breakout_gpt_alt",
            "run_dir": "runs/agent/backtest/20260102_010000_breakout_gpt_alt",
        },
        metrics_payload={
            "net_sharpe": 1.55,
            "net_pnl": 0.26,
            "net_mdd": -0.05,
        },
        resolved_config_payload=_agent_project_config(
            agent_name="breakout_gpt_alt",
            llm_model="gpt-5.4",
            max_position_pct=0.08,
            tools=["get_quote", "get_position", "place_order"],
            context_features=["rsi_14", "macd_fast"],
        ),
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "agent/backtest/20260101_010000_breakout_gpt",
            "--compare",
            "agent/backtest/20260102_010000_breakout_gpt_alt",
            "--sort-by",
            "net_sharpe",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Run comparison: agent/backtest" in result.stdout
    assert "NET_MDD" in result.stdout
    assert "agent.llm.model" in result.stdout
    assert "agent.risk.max_position_pct" in result.stdout
    assert "agent.tools" in result.stdout


def test_runs_list_compare_json_returns_stable_shape(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_alpha",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "alpha",
            "project_name": "alpha",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_alpha",
            "run_dir": "runs/research/20260101_000000_alpha",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
            },
        },
        resolved_config_payload=_research_project_config(),
    )
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260102_000000_beta",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "beta",
            "project_name": "beta",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-02T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260102_000000_beta",
            "run_dir": "runs/research/20260102_000000_beta",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.82, "f1": 0.61}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.40, "net_pnl": 0.22}
            },
        },
        resolved_config_payload=_research_project_config(timeframe="1h", horizon=10),
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "research/20260101_000000_alpha",
            "--compare",
            "research/20260102_000000_beta",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["kind"] == "run_comparison"
    assert payload["run_family"] == "research"
    assert payload["metric_columns"] == ["accuracy", "f1", "net_sharpe", "net_pnl"]
    assert len(payload["runs"]) == 2
    assert len(payload["rows"]) == 2
    assert "config_differences" in payload
    assert "warnings" in payload
    assert payload["rows"][0]["metrics"]["accuracy"] is not None


def test_runs_list_compare_rejects_mixed_run_families(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_alpha",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "alpha",
            "project_name": "alpha",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_alpha",
            "run_dir": "runs/research/20260101_000000_alpha",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
            },
        },
        resolved_config_payload=_research_project_config(),
    )
    _write_run_bundle(
        runs_root,
        relative_dir="agent/backtest/20260101_010000_breakout_gpt",
        summary_payload={
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "decision_count": 11,
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/backtest/20260101_010000_breakout_gpt",
            "run_dir": "runs/agent/backtest/20260101_010000_breakout_gpt",
        },
        metrics_payload={
            "net_sharpe": 1.25,
            "net_pnl": 0.18,
            "net_mdd": -0.07,
        },
        resolved_config_payload=_agent_project_config(agent_name="breakout_gpt"),
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "research/20260101_000000_alpha",
            "--compare",
            "agent/backtest/20260101_010000_breakout_gpt",
        ],
    )

    assert result.exit_code == 1
    combined = result.output
    assert "same family only" in combined
    assert "runs list --scoreboard" in combined


def test_runs_list_compare_rejects_too_few_runs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_alpha",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "alpha",
            "project_name": "alpha",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_alpha",
            "run_dir": "runs/research/20260101_000000_alpha",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
            },
        },
        resolved_config_payload=_research_project_config(),
    )

    result = runner.invoke(
        app,
        ["runs", "list", "--compare", "research/20260101_000000_alpha"],
    )

    assert result.exit_code == 1
    combined = result.output
    assert "at least two explicit --compare values" in combined


def test_runs_list_compare_rejects_unknown_run_id(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _write_run_bundle(
        runs_root,
        relative_dir="research/20260101_000000_alpha",
        summary_payload={
            "run_type": "research",
            "mode": "research",
            "name": "alpha",
            "project_name": "alpha",
            "status": "success",
            "symbols": ["AAPL", "MSFT"],
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "research/20260101_000000_alpha",
            "run_dir": "runs/research/20260101_000000_alpha",
        },
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
            },
        },
        resolved_config_payload=_research_project_config(),
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "research/20260101_000000_alpha",
            "--compare",
            "research/does_not_exist",
        ],
    )

    assert result.exit_code == 1
    combined = result.output
    assert "Run not found for compare" in combined


def test_runs_list_compare_rejects_more_than_four_runs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    for idx in range(1, 6):
        run_id = f"research/2026010{idx}_000000_run_{idx}"
        _write_run_bundle(
            runs_root,
            relative_dir=run_id,
            summary_payload={
                "run_type": "research",
                "mode": "research",
                "name": f"run_{idx}",
                "project_name": f"run_{idx}",
                "status": "success",
                "symbols": ["AAPL", "MSFT"],
                "timestamps": {"started_at": f"2026-01-0{idx}T00:00:00+00:00"},
                "artifacts": {},
                "warnings": [],
                "run_id": run_id,
                "run_dir": f"runs/{run_id}",
            },
            metrics_payload={
                "status": "available",
                "research_metrics_by_symbol": {
                    "AAPL": {"accuracy": 0.70 + idx / 100.0, "f1": 0.50}
                },
                "backtest_metrics_by_symbol": {
                    "AAPL": {"net_sharpe": 1.0 + idx / 10.0, "net_pnl": 0.10}
                },
            },
            resolved_config_payload=_research_project_config(horizon=idx),
        )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--compare",
            "research/20260101_000000_run_1",
            "--compare",
            "research/20260102_000000_run_2",
            "--compare",
            "research/20260103_000000_run_3",
            "--compare",
            "research/20260104_000000_run_4",
            "--compare",
            "research/20260105_000000_run_5",
        ],
    )

    assert result.exit_code == 1
    combined = result.output
    assert "at most four explicit --compare values" in combined


def test_runs_list_compare_rejects_filter_flags(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    for idx in range(1, 3):
        run_id = f"research/2026010{idx}_000000_run_{idx}"
        _write_run_bundle(
            runs_root,
            relative_dir=run_id,
            summary_payload={
                "run_type": "research",
                "mode": "research",
                "name": f"run_{idx}",
                "project_name": f"run_{idx}",
                "status": "success",
                "symbols": ["AAPL", "MSFT"],
                "timestamps": {"started_at": f"2026-01-0{idx}T00:00:00+00:00"},
                "artifacts": {},
                "warnings": [],
                "run_id": run_id,
                "run_dir": f"runs/{run_id}",
            },
            metrics_payload={
                "status": "available",
                "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75, "f1": 0.55}},
                "backtest_metrics_by_symbol": {
                    "AAPL": {"net_sharpe": 1.10, "net_pnl": 0.15}
                },
            },
            resolved_config_payload=_research_project_config(horizon=idx),
        )

    result = runner.invoke(
        app,
        [
            "runs",
            "list",
            "--type",
            "research",
            "--compare",
            "research/20260101_000000_run_1",
            "--compare",
            "research/20260102_000000_run_2",
        ],
    )

    assert result.exit_code != 0
    combined = result.output
    assert "Compare mode does not support --type" in combined
