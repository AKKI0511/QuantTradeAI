import json
from pathlib import Path

import pytest

from quanttradeai.utils.run_scoreboard import load_scoreboard_record


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_bundle(
    root: Path,
    *,
    run_type: str,
    mode: str,
    metrics_payload: dict | None = None,
    summary_payload: dict | None = None,
) -> dict:
    run_dir = root / run_type / mode / "20260101_000000_demo"
    _write_json(
        run_dir / "summary.json",
        {
            "run_type": run_type,
            "mode": mode,
            "name": "demo",
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            **(summary_payload or {}),
        },
    )
    if metrics_payload is not None:
        _write_json(run_dir / "metrics.json", metrics_payload)

    return {
        "run_id": f"{run_type}/{mode}/20260101_000000_demo",
        "run_type": run_type,
        "mode": mode,
        "name": "demo",
        "status": "success",
        "symbols": ["AAPL"],
        "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
        "artifacts": {},
        "warnings": [],
        "run_dir": str(run_dir),
    }


def test_load_scoreboard_record_for_research_with_research_and_backtest_metrics(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="research",
        mode="research",
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {
                "AAPL": {"accuracy": 0.70, "f1": 0.50},
                "MSFT": {"accuracy": 0.90, "f1": 0.70},
            },
            "backtest_metrics_by_symbol": {
                "AAPL": {"net_sharpe": 1.0, "net_pnl": 0.10},
                "MSFT": {"net_sharpe": 2.0, "net_pnl": 0.30},
            },
        },
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["status"] == "available"
    assert scoreboard["accuracy"] == 0.80
    assert scoreboard["f1"] == 0.60
    assert scoreboard["net_sharpe"] == 1.50
    assert scoreboard["net_pnl"] == 0.20
    assert scoreboard["primary_metric_name"] == "net_sharpe"
    assert scoreboard["primary_metric"] == 1.50


def test_load_scoreboard_record_for_research_with_research_metrics_only(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="research",
        mode="research",
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {
                "AAPL": {"accuracy": 0.75, "f1": 0.55},
                "MSFT": {"accuracy": 0.85, "f1": 0.65},
            },
            "backtest_metrics_by_symbol": {},
        },
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["accuracy"] == 0.80
    assert scoreboard["f1"] == pytest.approx(0.60)
    assert scoreboard["net_sharpe"] is None
    assert scoreboard["primary_metric_name"] == "accuracy"
    assert scoreboard["primary_metric"] == 0.80


def test_load_scoreboard_record_for_agent_backtest_uses_aggregate_metrics(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="agent",
        mode="backtest",
        metrics_payload={
            "net_sharpe": 1.25,
            "net_pnl": 0.18,
            "net_mdd": -0.07,
        },
        summary_payload={"decision_count": 12},
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["net_sharpe"] == 1.25
    assert scoreboard["net_pnl"] == 0.18
    assert scoreboard["net_mdd"] == -0.07
    assert scoreboard["decision_count"] == 12
    assert scoreboard["primary_metric_name"] == "net_sharpe"
    assert scoreboard["primary_metric"] == 1.25


def test_load_scoreboard_record_for_agent_backtest_preserves_zero_decision_count(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="agent",
        mode="backtest",
        metrics_payload={
            "net_sharpe": 0.5,
            "decision_count": None,
        },
        summary_payload={"decision_count": 0},
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["decision_count"] == 0


def test_load_scoreboard_record_for_agent_paper_or_live_exposes_streaming_metrics(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="agent",
        mode="paper",
        metrics_payload={
            "status": "available",
            "total_pnl": 150.5,
            "portfolio_value": 100150.5,
            "execution_count": 4,
            "risk_status": {"status": "ok", "halted": False},
        },
        summary_payload={"decision_count": 9},
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["total_pnl"] == 150.5
    assert scoreboard["portfolio_value"] == 100150.5
    assert scoreboard["execution_count"] == 4
    assert scoreboard["decision_count"] == 9
    assert scoreboard["risk_status"] == "ok"
    assert scoreboard["primary_metric_name"] == "total_pnl"
    assert scoreboard["primary_metric"] == 150.5


def test_load_scoreboard_record_handles_missing_metrics_file(tmp_path: Path):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="agent",
        mode="live",
        metrics_payload=None,
    )

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["status"] == "missing"
    assert "Missing JSON artifact" in scoreboard["error"]
    assert scoreboard["primary_metric"] is None


def test_load_scoreboard_record_handles_malformed_metrics_file(tmp_path: Path):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="agent",
        mode="live",
        metrics_payload={"status": "placeholder"},
    )
    metrics_path = Path(record["run_dir"]) / "metrics.json"
    metrics_path.write_text("{not-json", encoding="utf-8")

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["status"] == "invalid"
    assert "Invalid JSON artifact" in scoreboard["error"]
    assert scoreboard["primary_metric"] is None


def test_load_scoreboard_record_resolves_relative_metrics_artifact_from_run_dir(
    tmp_path: Path,
):
    record = _write_run_bundle(
        tmp_path / "runs",
        run_type="research",
        mode="research",
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.9}},
        },
    )
    record["artifacts"] = {"metrics": "metrics.json"}

    scoreboard = load_scoreboard_record(record)

    assert scoreboard["status"] == "available"
    assert scoreboard["accuracy"] == 0.9
    assert scoreboard["metrics_path"] == str(Path(record["run_dir"]) / "metrics.json")
