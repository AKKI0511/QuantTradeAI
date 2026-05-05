from quanttradeai.utils.run_result import attach_run_result, compact_cli_result


def _summary(*, run_type: str, mode: str, status: str = "success") -> dict:
    run_id = f"{run_type}/{mode}/demo" if run_type != "research" else "research/demo"
    return {
        "run_id": run_id,
        "run_type": run_type,
        "mode": mode,
        "name": "demo_agent" if run_type == "agent" else "demo_project",
        "agent_name": "demo_agent" if run_type == "agent" else None,
        "status": status,
        "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
        "symbols": ["AAPL"],
        "warnings": [],
        "artifacts": {"metrics": "runs/demo/metrics.json"},
        "run_dir": "runs/demo",
    }


def _entry(
    name: str,
    run_id: str,
    *,
    status: str = "success",
    score: float | None = None,
    error: str | None = None,
) -> dict:
    return {
        "agent_name": name,
        "base_agent_name": "rsi_reversion",
        "status": status,
        "warnings": ["child warning"] if status == "failed" else [],
        "error": error,
        "parameters": {"rule.buy_below": 25.0},
        "artifacts": {"summary": f"runs/{run_id}/summary.json"},
        "run_id": run_id,
        "run_dir": f"runs/{run_id}",
        "stdout_log": f"logs/{name}.stdout.log",
        "stderr_log": f"logs/{name}.stderr.log",
        "scoreboard": {"net_sharpe": score},
        "promote_command": f"quanttradeai promote --run {run_id} -c config/project.yaml",
    }


def test_research_success_recommends_model_promotion():
    summary = _summary(run_type="research", mode="research")
    attach_run_result(
        summary,
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75}},
            "backtest_metrics_by_symbol": {"AAPL": {"net_sharpe": 1.2}},
        },
    )

    assert summary["run_result"]["next_action"]["action"] == "promote_research_model"
    assert summary["run_result"]["commands"]["promote_model"] == (
        "quanttradeai promote --run research/demo -c config/project.yaml"
    )
    assert summary["run_result"]["key_metrics"]["accuracy"] == 0.75


def test_agent_backtest_success_recommends_paper_promotion():
    summary = _summary(run_type="agent", mode="backtest")
    attach_run_result(
        summary,
        metrics_payload={"net_sharpe": 1.1, "net_pnl": 0.2, "net_mdd": -0.05},
    )

    assert summary["run_result"]["next_action"]["action"] == "promote_agent_to_paper"
    assert summary["run_result"]["commands"]["promote_to_paper"] == (
        "quanttradeai promote --run agent/backtest/demo -c config/project.yaml"
    )
    assert summary["run_result"]["commands"]["run_paper"] == (
        "quanttradeai agent run --agent demo_agent -c config/project.yaml --mode paper"
    )


def test_agent_paper_success_recommends_live_promotion():
    summary = _summary(run_type="agent", mode="paper")
    attach_run_result(
        summary,
        metrics_payload={"total_pnl": 42.0, "portfolio_value": 100042.0},
    )

    assert summary["run_result"]["next_action"]["action"] == "promote_agent_to_live"
    assert summary["run_result"]["commands"]["promote_to_live"] == (
        "quanttradeai promote --run agent/paper/demo -c config/project.yaml "
        "--to live --acknowledge-live demo_agent"
    )


def test_agent_live_success_recommends_inspection():
    summary = _summary(run_type="agent", mode="live")
    attach_run_result(summary, metrics_payload={"total_pnl": 12.0})

    assert summary["run_result"]["next_action"]["action"] == "inspect_live_run"
    assert (
        "runs list --type agent --mode live"
        in summary["run_result"]["next_action"]["command"]
    )


def test_failed_run_recommends_failure_inspection():
    summary = _summary(run_type="agent", mode="backtest", status="failed")
    summary["error"] = "model failed"
    attach_run_result(summary)

    assert summary["run_result"]["next_action"]["action"] == "inspect_failure"
    assert summary["run_result"]["failure"]["error"] == "model failed"


def test_batch_sweep_winner_failures_and_compact_output_are_agent_ready():
    batch_summary = {
        "run_id": "agent/batches/demo_batch",
        "run_type": "batch",
        "mode": "backtest",
        "name": "demo_batch",
        "batch_type": "sweep",
        "project_name": "lab",
        "status": "success",
        "agent_count": 3,
        "success_count": 2,
        "failure_count": 1,
        "warnings": ["batch warning"],
        "artifacts": {
            "summary": "runs/agent/batches/demo_batch/summary.json",
            "results": "runs/agent/batches/demo_batch/results.json",
            "scoreboard_json": "runs/agent/batches/demo_batch/scoreboard.json",
        },
        "run_dir": "runs/agent/batches/demo_batch",
    }
    results = [
        _entry("low", "agent/backtest/low", score=0.5),
        _entry("high", "agent/backtest/high", score=2.5),
        _entry(
            "failed",
            "agent/backtest/failed",
            status="failed",
            error="bad config",
        ),
    ]

    attach_run_result(
        batch_summary,
        batch_results=results,
        scoreboard_order=["agent/backtest/high", "agent/backtest/low"],
        scoreboard_sort_by="net_sharpe",
    )

    run_result = batch_summary["run_result"]
    assert run_result["winner"]["agent_name"] == "high"
    assert run_result["top_candidates"][0]["score"] == 2.5
    assert run_result["failures"][0]["error"] == "bad config"
    assert run_result["next_action"]["action"] == "materialize_sweep_winner"
    assert run_result["commands"]["run_promoted_paper_agent"] == (
        "quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode paper"
    )

    compact = compact_cli_result(batch_summary)
    assert compact["run_id"] == "agent/batches/demo_batch"
    assert compact["key_metrics"]["winner_agent"] == "high"
    assert compact["important_artifacts"]["results"].endswith("results.json")
