import json
from pathlib import Path

import yaml

from quanttradeai.utils.run_brief import (
    RUN_BRIEF_KIND,
    RUN_BRIEF_SCHEMA_VERSION,
    build_run_brief,
    render_run_brief_markdown,
    write_run_brief_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "project.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project:\n  name: test\n", encoding="utf-8")
    return config_path


def _summary(
    run_dir: Path,
    *,
    run_id: str,
    run_type: str,
    mode: str,
    name: str,
    status: str = "success",
    extra: dict | None = None,
) -> dict:
    payload = {
        "run_id": run_id,
        "run_type": run_type,
        "mode": mode,
        "name": name,
        "status": status,
        "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
        "symbols": ["AAPL"],
        "warnings": [],
        "artifacts": {},
        "run_dir": str(run_dir),
    }
    payload.update(extra or {})
    return payload


def test_research_success_brief_recommends_ranking_and_promotion(tmp_path: Path):
    run_dir = tmp_path / "runs" / "research" / "20260101_research_lab"
    _write_json(
        run_dir / "metrics.json",
        {
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.8, "f1": 0.6}},
            "backtest_metrics_by_symbol": {"AAPL": {"net_sharpe": 1.4, "net_pnl": 0.2}},
        },
    )
    summary = _summary(
        run_dir,
        run_id="research/20260101_research_lab",
        run_type="research",
        mode="research",
        name="research_lab",
        extra={"project_name": "research_lab"},
    )

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir,
        project_config_path=_config_path(tmp_path),
    )

    assert brief["kind"] == RUN_BRIEF_KIND
    assert brief["schema_version"] == RUN_BRIEF_SCHEMA_VERSION
    assert brief["recommended_next_action"]["action"] == "promote_research_run"
    assert "rank_research_runs" in brief["commands"]
    assert brief["commands"]["promote_this_run"].startswith(
        "quanttradeai promote --run research/20260101_research_lab -c "
    )
    assert brief["scoreboard"]["primary_metric_name"] == "net_sharpe"
    assert brief["scoreboard"]["primary_metric"] == 1.4


def test_agent_backtest_success_brief_recommends_paper_promotion(tmp_path: Path):
    run_dir = tmp_path / "runs" / "agent" / "backtest" / "20260101_breakout_gpt"
    _write_json(
        run_dir / "metrics.json",
        {"net_sharpe": 1.2, "net_pnl": 0.1, "net_mdd": -0.05},
    )
    summary = _summary(
        run_dir,
        run_id="agent/backtest/20260101_breakout_gpt",
        run_type="agent",
        mode="backtest",
        name="breakout_gpt",
        extra={"agent_name": "breakout_gpt", "agent_kind": "llm"},
    )

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir,
        project_config_path=_config_path(tmp_path),
    )

    assert brief["recommended_next_action"]["action"] == "promote_to_paper"
    assert brief["commands"]["promote_to_paper"].startswith(
        "quanttradeai promote --run agent/backtest/20260101_breakout_gpt -c "
    )
    assert brief["scoreboard"]["net_sharpe"] == 1.2


def test_agent_paper_success_brief_recommends_live_promotion_and_deploy(
    tmp_path: Path,
):
    run_dir = tmp_path / "runs" / "agent" / "paper" / "20260101_breakout_gpt"
    resolved_config_path = run_dir / "resolved_project_config.yaml"
    _write_json(
        run_dir / "metrics.json",
        {
            "status": "available",
            "total_pnl": 120.0,
            "portfolio_value": 100120.0,
            "execution_count": 3,
            "decision_count": 5,
        },
    )
    _write_yaml(resolved_config_path, {"deployment": {"target": "render"}})
    summary = _summary(
        run_dir,
        run_id="agent/paper/20260101_breakout_gpt",
        run_type="agent",
        mode="paper",
        name="breakout_gpt",
        extra={
            "agent_name": "breakout_gpt",
            "agent_kind": "llm",
            "artifacts": {"resolved_project_config": str(resolved_config_path)},
        },
    )

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir,
        project_config_path=_config_path(tmp_path),
    )

    assert brief["recommended_next_action"]["action"] == "promote_to_live"
    assert (
        "--to live --acknowledge-live breakout_gpt"
        in brief["commands"]["promote_to_live"]
    )
    assert brief["commands"]["deploy_agent"].endswith("--target render")
    assert brief["scoreboard"]["total_pnl"] == 120.0


def test_agent_live_success_brief_recommends_live_ranking(tmp_path: Path):
    run_dir = tmp_path / "runs" / "agent" / "live" / "20260101_breakout_gpt"
    _write_json(
        run_dir / "metrics.json",
        {
            "status": "available",
            "total_pnl": 80.0,
            "portfolio_value": 100080.0,
            "execution_count": 2,
            "decision_count": 4,
            "risk_status": {"status": "ok"},
        },
    )
    summary = _summary(
        run_dir,
        run_id="agent/live/20260101_breakout_gpt",
        run_type="agent",
        mode="live",
        name="breakout_gpt",
        extra={"agent_name": "breakout_gpt", "agent_kind": "rule"},
    )

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir,
        project_config_path=_config_path(tmp_path),
    )

    assert brief["recommended_next_action"]["action"] == "inspect_live_run"
    assert brief["commands"] == {
        "rank_live_runs": (
            "quanttradeai runs list --type agent --mode live --scoreboard --sort-by total_pnl"
        )
    }
    assert brief["scoreboard"]["risk_status"] == "ok"


def test_failed_run_brief_uses_inspect_failure_without_commands(tmp_path: Path):
    run_dir = tmp_path / "runs" / "agent" / "backtest" / "20260101_failed"
    summary = _summary(
        run_dir,
        run_id="agent/backtest/20260101_failed",
        run_type="agent",
        mode="backtest",
        name="failed_agent",
        status="failed",
        extra={
            "agent_name": "failed_agent",
            "error": "missing prompt",
            "warnings": ["config warning"],
        },
    )

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir,
        project_config_path=_config_path(tmp_path),
    )

    assert brief["recommended_next_action"]["action"] == "inspect_failure"
    assert brief["commands"] == {}
    assert brief["error"] == "missing prompt"
    assert brief["warnings"] == ["config warning"]
    assert brief["scoreboard"]["status"] == "missing"


def test_write_run_brief_artifacts_updates_summary_and_markdown(tmp_path: Path):
    run_dir = tmp_path / "runs" / "agent" / "backtest" / "20260101_breakout_gpt"
    _write_json(
        run_dir / "metrics.json",
        {"net_sharpe": 1.2, "net_pnl": 0.1, "net_mdd": -0.05},
    )
    summary = _summary(
        run_dir,
        run_id="agent/backtest/20260101_breakout_gpt",
        run_type="agent",
        mode="backtest",
        name="breakout_gpt",
        extra={"agent_name": "breakout_gpt", "artifacts": {"metrics": "metrics.json"}},
    )

    artifacts = write_run_brief_artifacts(summary, run_dir, _config_path(tmp_path))

    assert Path(artifacts["run_brief_json"]).is_file()
    assert Path(artifacts["run_brief_md"]).is_file()
    assert summary["artifacts"]["run_brief_json"] == artifacts["run_brief_json"]
    assert summary["artifacts"]["run_brief_md"] == artifacts["run_brief_md"]

    brief = json.loads(Path(artifacts["run_brief_json"]).read_text("utf-8"))
    rendered = render_run_brief_markdown(brief)
    assert "## Recommendation" in rendered
    assert "promote_to_paper" in rendered
    assert "## Commands" in rendered
    assert "## Artifacts" in rendered


def test_write_run_brief_artifacts_keeps_summary_backed_decision_count(
    tmp_path: Path,
):
    run_dir = tmp_path / "runs" / "agent" / "backtest" / "20260101_breakout_gpt"
    _write_json(
        run_dir / "metrics.json",
        {"net_sharpe": 1.2, "net_pnl": 0.1, "net_mdd": -0.05},
    )
    summary = _summary(
        run_dir,
        run_id="agent/backtest/20260101_breakout_gpt",
        run_type="agent",
        mode="backtest",
        name="breakout_gpt",
        extra={
            "agent_name": "breakout_gpt",
            "decision_count": 17,
            "artifacts": {"metrics": str(run_dir / "metrics.json")},
        },
    )

    artifacts = write_run_brief_artifacts(summary, run_dir, _config_path(tmp_path))

    brief = json.loads(Path(artifacts["run_brief_json"]).read_text("utf-8"))
    persisted_summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert brief["scoreboard"]["decision_count"] == 17
    assert persisted_summary["decision_count"] == 17
    assert (
        persisted_summary["artifacts"]["run_brief_json"] == artifacts["run_brief_json"]
    )
