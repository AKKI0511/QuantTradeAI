import json
from pathlib import Path

from typer.testing import CliRunner

from quanttradeai.cli import app


runner = CliRunner()


def _write_run_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_runs(root: Path) -> None:
    _write_run_summary(
        root / "research" / "20260101_000000_research_lab" / "summary.json",
        {
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
    )
    _write_run_summary(
        root / "agent" / "backtest" / "20260101_010000_breakout_gpt" / "summary.json",
        {
            "run_type": "agent",
            "mode": "backtest",
            "name": "breakout_gpt",
            "agent_name": "breakout_gpt",
            "status": "success",
            "symbols": ["AAPL"],
            "timestamps": {"started_at": "2026-01-01T01:00:00+00:00"},
            "artifacts": {},
            "warnings": [],
            "run_id": "agent/backtest/20260101_010000_breakout_gpt",
            "run_dir": "runs/agent/backtest/20260101_010000_breakout_gpt",
        },
    )
    _write_run_summary(
        root / "20251231_230000_legacy_agent" / "summary.json",
        {
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
    _write_run_summary(
        runs_root / "agent" / "live" / "20260101_020000_breakout_live" / "summary.json",
        {
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
    )

    result = runner.invoke(app, ["runs", "list", "--mode", "live", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["run_id"] == "agent/live/20260101_020000_breakout_live"
    assert payload[0]["mode"] == "live"
    assert payload[0]["name"] == "breakout_live"
