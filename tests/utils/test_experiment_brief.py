from quanttradeai.utils.experiment_brief import (
    build_experiment_brief,
    render_experiment_brief_markdown,
)


def _entry(
    name: str,
    run_id: str,
    *,
    status: str = "success",
    score: float | None = None,
    mode: str = "backtest",
    base_agent_name: str | None = None,
    parameters: dict | None = None,
    promote_command: str | None = None,
    error: str | None = None,
    warnings: list[str] | None = None,
) -> dict:
    entry = {
        "agent_name": name,
        "base_agent_name": base_agent_name,
        "agent_kind": "rule",
        "configured_mode": mode,
        "status": status,
        "warnings": list(warnings or []),
        "error": error,
        "parameters": parameters,
        "artifacts": {"summary": f"runs/{run_id}/summary.json"},
        "run_id": run_id,
        "run_dir": f"runs/{run_id}",
        "stdout_log": f"logs/{name}.stdout.log",
        "stderr_log": f"logs/{name}.stderr.log",
        "scoreboard": {"net_sharpe": score, "total_pnl": score},
    }
    if promote_command:
        entry["promote_command"] = promote_command
    return entry


def _batch(*, batch_type: str, mode: str, status: str = "success") -> dict:
    return {
        "run_id": "agent/batches/20260101_000000_lab_backtest",
        "batch_type": batch_type,
        "project_name": "lab",
        "mode": mode,
        "status": status,
        "agent_count": 2,
        "success_count": 2 if status == "success" else 1,
        "failure_count": 0 if status == "success" else 1,
        "run_dir": "runs/agent/batches/20260101_000000_lab_backtest",
    }


def test_sweep_brief_chooses_top_ranked_success_and_uses_promote_command():
    winner_command = (
        "quanttradeai promote --run agent/backtest/high -c C:/repo/config/project.yaml"
    )
    results = [
        _entry(
            "low",
            "agent/backtest/low",
            score=0.5,
            base_agent_name="rsi_reversion",
            parameters={"rule.buy_below": 25.0},
            promote_command="quanttradeai promote --run agent/backtest/low -c C:/repo/config/project.yaml",
        ),
        _entry(
            "high",
            "agent/backtest/high",
            score=2.5,
            base_agent_name="rsi_reversion",
            parameters={"rule.buy_below": 30.0},
            promote_command=winner_command,
        ),
    ]

    brief = build_experiment_brief(
        batch=_batch(batch_type="sweep", mode="backtest"),
        results=results,
        scoreboard_order=["agent/backtest/high", "agent/backtest/low"],
        scoreboard_sort_by="net_sharpe",
        project_config_path="C:/repo/config/project.yaml",
        artifacts={"results": "results.json"},
    )

    assert brief["winner"]["agent_name"] == "high"
    assert brief["winner"]["parameters"] == {"rule.buy_below": 30.0}
    assert brief["winner"]["artifacts"] == {
        "summary": "runs/agent/backtest/high/summary.json"
    }
    assert brief["recommended_next_action"]["action"] == "materialize_sweep_winner"
    assert brief["commands"]["promote_winner"] == winner_command
    assert (
        brief["commands"]["run_promoted_paper_agent"]
        == "quanttradeai agent run --agent rsi_reversion -c C:/repo/config/project.yaml --mode paper"
    )
    assert "compare_top_runs" in brief["commands"]


def test_all_agent_backtest_brief_creates_paper_promotion_command():
    results = [
        _entry("agent_a", "agent/backtest/a", score=1.0),
        _entry("agent_b", "agent/backtest/b", score=1.5),
    ]

    brief = build_experiment_brief(
        batch=_batch(batch_type="all_agents", mode="backtest"),
        results=results,
        scoreboard_order=["agent/backtest/b", "agent/backtest/a"],
        scoreboard_sort_by="net_sharpe",
        project_config_path="config/project.yaml",
        artifacts={},
    )

    assert brief["winner"]["agent_name"] == "agent_b"
    assert brief["recommended_next_action"]["action"] == "promote_winner_to_paper"
    assert (
        brief["commands"]["promote_winner"]
        == "quanttradeai promote --run agent/backtest/b -c config/project.yaml"
    )


def test_paper_batch_brief_creates_live_promotion_command_with_acknowledgement():
    results = [
        _entry("paper_a", "agent/paper/a", score=10.0, mode="paper"),
        _entry("paper_b", "agent/paper/b", score=20.0, mode="paper"),
    ]

    brief = build_experiment_brief(
        batch=_batch(batch_type="all_agents", mode="paper"),
        results=results,
        scoreboard_order=["agent/paper/b", "agent/paper/a"],
        scoreboard_sort_by="total_pnl",
        project_config_path="config/project.yaml",
        artifacts={},
    )

    assert brief["winner"]["agent_name"] == "paper_b"
    assert brief["recommended_next_action"]["action"] == "promote_winner_to_live"
    assert brief["commands"]["promote_winner_to_live"] == (
        "quanttradeai promote --run agent/paper/b -c config/project.yaml "
        "--to live --acknowledge-live paper_b"
    )


def test_partial_failure_brief_keeps_failure_logs_and_picks_best_success():
    results = [
        _entry("winner", "agent/backtest/winner", score=1.0),
        _entry(
            "failed",
            "agent/backtest/failed",
            status="failed",
            error="model failure",
            warnings=["child warning"],
        ),
    ]

    brief = build_experiment_brief(
        batch=_batch(batch_type="all_agents", mode="backtest", status="failed"),
        results=results,
        scoreboard_order=["agent/backtest/winner", "agent/backtest/failed"],
        scoreboard_sort_by="net_sharpe",
        project_config_path="config/project.yaml",
        artifacts={},
        warnings=["batch warning"],
    )

    assert brief["winner"]["agent_name"] == "winner"
    assert brief["failures"] == [
        {
            "agent_name": "failed",
            "base_agent_name": None,
            "run_id": "agent/backtest/failed",
            "run_dir": "runs/agent/backtest/failed",
            "status": "failed",
            "error": "model failure",
            "warnings": ["child warning"],
            "parameters": None,
            "artifacts": {"summary": "runs/agent/backtest/failed/summary.json"},
            "stdout_log": "logs/failed.stdout.log",
            "stderr_log": "logs/failed.stderr.log",
        }
    ]
    assert brief["warnings"] == ["batch warning", "child warning"]


def test_all_failures_brief_recommends_inspecting_failures():
    results = [
        _entry(
            "failed_a",
            "agent/backtest/failed_a",
            status="failed",
            error="first failure",
        ),
        _entry(
            "failed_b",
            "agent/backtest/failed_b",
            status="failed",
            error="second failure",
        ),
    ]

    brief = build_experiment_brief(
        batch=_batch(batch_type="all_agents", mode="backtest", status="failed"),
        results=results,
        scoreboard_order=["agent/backtest/failed_a", "agent/backtest/failed_b"],
        scoreboard_sort_by="net_sharpe",
        project_config_path="config/project.yaml",
        artifacts={},
    )

    assert brief["winner"] is None
    assert brief["recommended_next_action"] == {
        "action": "inspect_failures",
        "reason": "No successful child runs were available to promote or compare.",
        "command": None,
        "follow_up_command": None,
    }
    assert brief["commands"] == {}
    assert len(brief["failures"]) == 2


def test_render_experiment_brief_markdown_mirrors_core_sections():
    brief = build_experiment_brief(
        batch=_batch(batch_type="all_agents", mode="backtest"),
        results=[_entry("winner", "agent/backtest/winner", score=1.0)],
        scoreboard_order=["agent/backtest/winner"],
        scoreboard_sort_by="net_sharpe",
        project_config_path="config/project.yaml",
        artifacts={"experiment_brief_json": "experiment_brief.json"},
    )

    rendered = render_experiment_brief_markdown(brief)

    assert "# Experiment Brief" in rendered
    assert "## Recommendation" in rendered
    assert "promote_winner_to_paper" in rendered
    assert "experiment_brief_json" in rendered
