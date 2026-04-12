import json
from pathlib import Path

from typer.testing import CliRunner

from quanttradeai.cli import app


runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_bundle(
    root: Path,
    *,
    relative_dir: str,
    summary_payload: dict,
    metrics_payload: dict | None = None,
) -> None:
    run_dir = root / Path(relative_dir)
    _write_json(run_dir / "summary.json", summary_payload)
    if metrics_payload is not None:
        _write_json(run_dir / "metrics.json", metrics_payload)


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


def test_runs_list_ignores_batch_artifacts_without_child_run_summaries(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    runs_root = Path("runs")
    _seed_runs(runs_root)

    batch_root = runs_root / "agent" / "batches" / "20260101_000000_multi_agent_backtest"
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "batch_manifest.json").write_text("{}", encoding="utf-8")
    (batch_root / "results.json").write_text("{}", encoding="utf-8")
    (batch_root / "scoreboard.json").write_text("{}", encoding="utf-8")
    (batch_root / "scoreboard.txt").write_text("placeholder", encoding="utf-8")

    result = runner.invoke(app, ["runs", "list", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert all("batches" not in record["run_dir"] for record in payload)
