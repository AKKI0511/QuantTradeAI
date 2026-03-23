import json
from pathlib import Path

from quanttradeai.utils.run_records import RunFilters, discover_runs, filter_runs, normalize_run_summary


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_normalize_run_summary_for_research_run(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "research" / "20260101_000000_research_lab"
    summary = {
        "project_name": "research_lab",
        "status": "success",
        "symbols": ["AAPL", "MSFT"],
        "warnings": ["fallback split"],
        "artifacts": {"metrics": "metrics.json"},
        "timestamps": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:05:00+00:00",
        },
    }

    record = normalize_run_summary(summary, run_dir=run_dir, runs_root=runs_root)

    assert record == {
        "run_id": "research/20260101_000000_research_lab",
        "run_type": "research",
        "mode": "research",
        "name": "research_lab",
        "status": "success",
        "timestamps": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:05:00+00:00",
        },
        "symbols": ["AAPL", "MSFT"],
        "warnings": ["fallback split"],
        "artifacts": {"metrics": "metrics.json"},
        "run_dir": str(run_dir),
    }


def test_normalize_run_summary_detects_legacy_agent_run(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "20260101_000000_breakout_gpt"
    summary = {
        "agent_name": "breakout_gpt",
        "mode": "backtest",
        "status": "failed",
        "timestamps": {
            "started_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
        },
        "artifacts": {},
    }

    record = normalize_run_summary(summary, run_dir=run_dir, runs_root=runs_root)

    assert record["run_id"] == "20260101_000000_breakout_gpt"
    assert record["run_type"] == "agent"
    assert record["mode"] == "backtest"
    assert record["name"] == "breakout_gpt"


def test_discover_runs_skips_validation_summaries_and_sorts_newest_first(tmp_path: Path):
    runs_root = tmp_path / "runs"

    research_run = runs_root / "research" / "20260101_000000_research_lab" / "summary.json"
    _write_summary(
        research_run,
        {
            "project_name": "research_lab",
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
        },
    )
    _write_summary(
        research_run.parent / "validation" / "summary.json",
        {
            "project": {"name": "validation-only"},
        },
    )

    agent_run = runs_root / "agent" / "backtest" / "20260101_010000_breakout_gpt" / "summary.json"
    _write_summary(
        agent_run,
        {
            "agent_name": "breakout_gpt",
            "run_type": "agent",
            "mode": "backtest",
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
        },
    )

    records = discover_runs(runs_root)

    assert [record["name"] for record in records] == ["breakout_gpt", "research_lab"]


def test_filter_runs_applies_filters_and_limit():
    records = [
        {
            "run_id": "agent/backtest/latest",
            "run_type": "agent",
            "mode": "backtest",
            "name": "latest",
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T02:00:00+00:00"},
            "symbols": [],
            "warnings": [],
            "artifacts": {},
            "run_dir": "runs/agent/backtest/latest",
        },
        {
            "run_id": "agent/backtest/older",
            "run_type": "agent",
            "mode": "backtest",
            "name": "older",
            "status": "failed",
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "symbols": [],
            "warnings": [],
            "artifacts": {},
            "run_dir": "runs/agent/backtest/older",
        },
        {
            "run_id": "research/20260101",
            "run_type": "research",
            "mode": "research",
            "name": "research_lab",
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": [],
            "warnings": [],
            "artifacts": {},
            "run_dir": "runs/research/20260101",
        },
    ]

    filtered = filter_runs(
        records,
        RunFilters(run_type="agent", mode="backtest", status="all", limit=1),
    )

    assert len(filtered) == 1
    assert filtered[0]["name"] == "latest"
