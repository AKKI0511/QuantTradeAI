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


def test_research_success_summarizes_metrics_without_prescriptive_fields():
    summary = _summary(run_type="research", mode="research")
    attach_run_result(
        summary,
        metrics_payload={
            "status": "available",
            "research_metrics_by_symbol": {"AAPL": {"accuracy": 0.75}},
            "backtest_metrics_by_symbol": {"AAPL": {"net_sharpe": 1.2}},
        },
    )

    run_result = summary["run_result"]
    assert run_result["schema_version"] == 2
    assert run_result["metrics"]["accuracy"] == 0.75
    assert run_result["metrics"]["net_sharpe"] == 1.2
    assert "next_action" not in run_result
    assert "commands" not in run_result
    assert "important_artifacts" not in run_result
    assert "warnings" not in run_result


def test_agent_backtest_success_summarizes_result_metrics():
    summary = _summary(run_type="agent", mode="backtest")
    attach_run_result(
        summary,
        metrics_payload={"net_sharpe": 1.1, "net_pnl": 0.2, "net_mdd": -0.05},
    )

    assert summary["run_result"]["metrics"] == {
        "metrics_status": "available",
        "net_sharpe": 1.1,
        "net_pnl": 0.2,
        "net_mdd": -0.05,
    }


def test_agent_paper_success_summarizes_execution_metrics():
    summary = _summary(run_type="agent", mode="paper")
    attach_run_result(
        summary,
        metrics_payload={"total_pnl": 42.0, "portfolio_value": 100042.0},
    )

    assert summary["run_result"]["metrics"] == {
        "metrics_status": "available",
        "total_pnl": 42.0,
        "portfolio_value": 100042.0,
    }


def test_agent_live_success_summarizes_execution_metrics():
    summary = _summary(run_type="agent", mode="live")
    attach_run_result(
        summary,
        metrics_payload={"total_pnl": 12.0, "risk_status": "ok"},
    )

    assert summary["run_result"]["metrics"] == {
        "metrics_status": "available",
        "total_pnl": 12.0,
        "risk_status": "ok",
    }


def test_failed_run_preserves_failure_at_summary_level_only():
    summary = _summary(run_type="agent", mode="backtest", status="failed")
    summary["error"] = "model failed"
    attach_run_result(summary)

    run_result = summary["run_result"]
    assert run_result["metrics"]["metrics_status"] == "missing"
    assert "failure" not in run_result
    compact = compact_cli_result(summary)
    assert compact["error"] == "model failed"


def test_batch_sweep_winner_failures_and_compact_output_are_sparse():
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
        "scoreboard_sort_by": "net_sharpe",
        "warnings": ["batch warning"],
        "artifacts": {
            "summary": "runs/agent/batches/demo_batch/summary.json",
            "results": "runs/agent/batches/demo_batch/results.json",
            "scoreboard_json": "runs/agent/batches/demo_batch/scoreboard.json",
        },
        "run_dir": "runs/agent/batches/demo_batch",
        "sweep": {"name": "grid", "base_agent_name": "rsi_reversion"},
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

    batch = batch_summary["run_result"]["batch"]
    assert batch["winner"]["agent_name"] == "high"
    assert batch["top_candidates"][0]["agent_name"] == "low"
    assert batch["top_candidates"][0]["score"] == 0.5
    assert batch["failures"][0]["error"] == "bad config"
    assert "artifacts" not in batch["winner"]
    assert "promote_command" not in batch["winner"]
    assert "score_metric" not in batch["winner"]
    assert "warnings" not in batch["failures"][0]

    compact = compact_cli_result(batch_summary)
    assert compact == {
        "run_id": "agent/batches/demo_batch",
        "status": "success",
        "run_dir": "runs/agent/batches/demo_batch",
        "run_type": "batch",
        "mode": "backtest",
        "name": "demo_batch",
        "metrics": {"scoreboard_sort_by": "net_sharpe"},
        "winner": {
            "agent_name": "high",
            "run_id": "agent/backtest/high",
            "parameters": {"rule.buy_below": 25.0},
            "score": 2.5,
        },
        "batch_type": "sweep",
        "agent_count": 3,
        "success_count": 2,
        "failure_count": 1,
        "sweep": {"name": "grid", "base_agent_name": "rsi_reversion"},
    }
