"""Agent-ready run result contracts for CLI output and run summaries."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

IMPORTANT_ARTIFACT_KEYS = (
    "summary",
    "metrics",
    "results",
    "scoreboard_json",
    "resolved_project_config",
    "runtime_model_config",
    "runtime_features_config",
    "runtime_backtest_config",
    "runtime_streaming_config",
    "runtime_risk_config",
    "runtime_position_manager_config",
    "backtest_summary",
    "equity_curve",
    "ledger",
    "decisions",
    "executions",
    "prompt_samples",
    "replay_manifest",
    "broker_account_start",
    "broker_account_end",
    "broker_positions_start",
    "broker_positions_end",
)


def attach_run_result(
    summary: dict[str, Any],
    *,
    project_config_path: str | Path = "config/project.yaml",
    metrics_payload: dict[str, Any] | None = None,
    batch_results: list[dict[str, Any]] | None = None,
    scoreboard_order: list[str] | None = None,
    scoreboard_sort_by: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Attach the shared agent-ready contract to a run summary payload."""

    summary["run_result"] = build_run_result(
        summary,
        project_config_path=project_config_path,
        metrics_payload=metrics_payload,
        batch_results=batch_results,
        scoreboard_order=scoreboard_order,
        scoreboard_sort_by=scoreboard_sort_by,
        top_n=top_n,
    )
    return summary


def build_run_result(
    summary: dict[str, Any],
    *,
    project_config_path: str | Path = "config/project.yaml",
    metrics_payload: dict[str, Any] | None = None,
    batch_results: list[dict[str, Any]] | None = None,
    scoreboard_order: list[str] | None = None,
    scoreboard_sort_by: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build the shared result payload used by summary.json and compact CLI JSON."""

    run_type = str(summary.get("run_type") or "")
    mode = str(summary.get("mode") or "")
    status = str(summary.get("status") or "unknown")
    warnings = _unique_strings(list(summary.get("warnings") or []))

    if run_type == "batch":
        run_result = _build_batch_run_result(
            summary,
            project_config_path=project_config_path,
            batch_results=batch_results or [],
            scoreboard_order=scoreboard_order or [],
            scoreboard_sort_by=scoreboard_sort_by
            or str(summary.get("scoreboard_sort_by") or ""),
            top_n=top_n,
        )
    else:
        key_metrics = _build_key_metrics(summary, metrics_payload=metrics_payload)
        commands = _build_single_run_commands(
            summary,
            project_config_path=project_config_path,
        )
        next_action = _build_single_next_action(
            summary,
            commands=commands,
        )
        run_result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": summary.get("run_id"),
            "run_type": run_type,
            "mode": mode,
            "status": status,
            "key_metrics": key_metrics,
            "next_action": next_action,
            "commands": commands,
            "important_artifacts": _important_artifacts(summary),
            "warnings": warnings,
        }

    if status != "success":
        run_result["failure"] = _failure_payload(summary)
    return _json_safe(run_result)


def compact_cli_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the compact JSON result printed by completion-oriented commands."""

    run_result = summary.get("run_result")
    if not isinstance(run_result, dict):
        run_result = build_run_result(summary)

    payload: dict[str, Any] = {
        "run_id": summary.get("run_id") or run_result.get("run_id"),
        "status": summary.get("status") or run_result.get("status"),
        "run_dir": summary.get("run_dir"),
        "run_type": summary.get("run_type") or run_result.get("run_type"),
        "mode": summary.get("mode") or run_result.get("mode"),
        "name": summary.get("name"),
        "key_metrics": dict(run_result.get("key_metrics") or {}),
        "next_action": dict(run_result.get("next_action") or {}),
        "commands": dict(run_result.get("commands") or {}),
        "important_artifacts": dict(run_result.get("important_artifacts") or {}),
        "warnings": list(run_result.get("warnings") or []),
    }

    for key in (
        "batch_type",
        "project_name",
        "agent_count",
        "success_count",
        "failure_count",
        "agent_kind",
        "paper_source",
        "execution_backend",
        "broker_provider",
        "decision_count",
        "execution_count",
    ):
        if key in summary:
            payload[key] = summary[key]
    if "sweep" in summary:
        payload["sweep"] = summary["sweep"]
    if "failure" in run_result:
        payload["failure"] = run_result["failure"]

    return _drop_none(_json_safe(payload))


def _build_batch_run_result(
    summary: dict[str, Any],
    *,
    project_config_path: str | Path,
    batch_results: list[dict[str, Any]],
    scoreboard_order: list[str],
    scoreboard_sort_by: str,
    top_n: int,
) -> dict[str, Any]:
    status = str(summary.get("status") or "unknown")
    ranked_successes = _rank_successful_results(
        batch_results,
        scoreboard_order=scoreboard_order,
    )
    winner = (
        _candidate_payload(
            ranked_successes[0],
            scoreboard_sort_by=scoreboard_sort_by,
        )
        if ranked_successes
        else None
    )
    top_candidates = [
        _candidate_payload(entry, scoreboard_sort_by=scoreboard_sort_by)
        for entry in ranked_successes[: max(top_n, 1)]
    ]
    failures = [
        _failure_entry_payload(entry)
        for entry in batch_results
        if entry.get("status") != "success"
    ]
    warnings = _unique_strings(
        list(summary.get("warnings") or [])
        + [
            warning
            for entry in batch_results
            for warning in list(entry.get("warnings") or [])
        ]
    )
    commands = _build_batch_commands(
        summary,
        winner=winner,
        project_config_path=project_config_path,
        scoreboard_sort_by=scoreboard_sort_by,
    )
    next_action = _build_batch_next_action(
        summary,
        winner=winner,
        failures=failures,
        commands=commands,
    )
    key_metrics = {
        "agent_count": summary.get("agent_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "scoreboard_sort_by": scoreboard_sort_by or None,
    }
    if winner is not None:
        key_metrics["winner_agent"] = winner.get("agent_name")
        key_metrics["winner_run_id"] = winner.get("run_id")
        if winner.get("score") is not None:
            key_metrics["winner_score"] = winner.get("score")
            key_metrics["winner_score_metric"] = scoreboard_sort_by

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": summary.get("run_id"),
        "run_type": summary.get("run_type"),
        "mode": summary.get("mode"),
        "status": status,
        "batch_type": summary.get("batch_type"),
        "key_metrics": _drop_none(key_metrics),
        "next_action": next_action,
        "commands": commands,
        "important_artifacts": _important_artifacts(summary),
        "warnings": warnings,
        "winner": winner,
        "top_candidates": top_candidates,
        "failures": failures,
    }


def _build_single_run_commands(
    summary: dict[str, Any],
    *,
    project_config_path: str | Path,
) -> dict[str, str]:
    if summary.get("status") != "success":
        return {}

    run_id = summary.get("run_id")
    if not run_id:
        return {}

    config = _config_path(project_config_path)
    run_type = str(summary.get("run_type") or "")
    mode = str(summary.get("mode") or "")
    agent_name = _agent_name(summary)
    commands: dict[str, str] = {}

    if run_type == "research":
        commands["promote_model"] = f"quanttradeai promote --run {run_id} -c {config}"
        commands["compare_research_runs"] = (
            "quanttradeai runs list --type research --scoreboard --sort-by net_sharpe"
        )
    elif run_type == "agent" and mode == "backtest":
        commands["promote_to_paper"] = (
            f"quanttradeai promote --run {run_id} -c {config}"
        )
        if agent_name:
            commands["run_paper"] = (
                f"quanttradeai agent run --agent {agent_name} -c {config} --mode paper"
            )
    elif run_type == "agent" and mode == "paper":
        if agent_name:
            commands["promote_to_live"] = (
                f"quanttradeai promote --run {run_id} -c {config} "
                f"--to live --acknowledge-live {agent_name}"
            )
            commands["run_live"] = (
                f"quanttradeai agent run --agent {agent_name} -c {config} --mode live"
            )
    elif run_type == "agent" and mode == "live":
        commands["inspect_live_runs"] = (
            "quanttradeai runs list --type agent --mode live "
            "--scoreboard --sort-by total_pnl"
        )
    return commands


def _build_single_next_action(
    summary: dict[str, Any],
    *,
    commands: dict[str, str],
) -> dict[str, Any]:
    status = str(summary.get("status") or "")
    run_type = str(summary.get("run_type") or "")
    mode = str(summary.get("mode") or "")

    if status != "success":
        return {
            "action": "inspect_failure",
            "reason": "Run failed before a promotable result was produced.",
            "command": None,
            "follow_up_command": None,
        }

    if run_type == "research":
        return {
            "action": "promote_research_model",
            "reason": "Research run completed and produced a candidate model result.",
            "command": commands.get("promote_model"),
            "follow_up_command": commands.get("compare_research_runs"),
        }
    if run_type == "agent" and mode == "backtest":
        return {
            "action": "promote_agent_to_paper",
            "reason": "Agent backtest completed and can be promoted to paper mode.",
            "command": commands.get("promote_to_paper"),
            "follow_up_command": commands.get("run_paper"),
        }
    if run_type == "agent" and mode == "paper":
        return {
            "action": "promote_agent_to_live",
            "reason": "Paper run completed and can be promoted to live mode.",
            "command": commands.get("promote_to_live"),
            "follow_up_command": commands.get("run_live"),
        }
    if run_type == "agent" and mode == "live":
        return {
            "action": "inspect_live_run",
            "reason": "Live run completed; inspect metrics and execution logs before changing exposure.",
            "command": commands.get("inspect_live_runs"),
            "follow_up_command": None,
        }

    return {
        "action": "inspect_run",
        "reason": "Run completed.",
        "command": "quanttradeai runs list --json",
        "follow_up_command": None,
    }


def _build_batch_commands(
    summary: dict[str, Any],
    *,
    winner: dict[str, Any] | None,
    project_config_path: str | Path,
    scoreboard_sort_by: str,
) -> dict[str, str]:
    mode = str(summary.get("mode") or "")
    batch_type = str(summary.get("batch_type") or "")
    config = _config_path(project_config_path)
    commands: dict[str, str] = {
        "inspect_batch": "quanttradeai runs list --type batch --json",
    }
    if mode:
        commands["compare_top_runs"] = (
            f"quanttradeai runs list --type agent --mode {mode} "
            f"--scoreboard --sort-by {scoreboard_sort_by or 'started_at'}"
        )
    if winner is None:
        commands["inspect_failures"] = (
            "quanttradeai runs list --type batch --status failed --json"
        )
        return commands

    run_id = winner.get("run_id")
    agent_name = winner.get("base_agent_name") or winner.get("agent_name")
    if mode == "backtest" and run_id:
        commands["promote_winner"] = str(
            winner.get("promote_command")
            or f"quanttradeai promote --run {run_id} -c {config}"
        )
        if agent_name:
            commands["run_promoted_paper_agent"] = (
                f"quanttradeai agent run --agent {agent_name} -c {config} --mode paper"
            )
    elif mode == "paper" and run_id and winner.get("agent_name"):
        commands["promote_winner_to_live"] = (
            f"quanttradeai promote --run {run_id} -c {config} "
            f"--to live --acknowledge-live {winner['agent_name']}"
        )
    elif mode == "live":
        commands["inspect_live_batch"] = (
            "quanttradeai runs list --type agent --mode live "
            "--scoreboard --sort-by total_pnl"
        )
    if batch_type == "sweep" and "promote_winner" in commands:
        commands["materialize_sweep_winner"] = commands["promote_winner"]
    return commands


def _build_batch_next_action(
    summary: dict[str, Any],
    *,
    winner: dict[str, Any] | None,
    failures: list[dict[str, Any]],
    commands: dict[str, str],
) -> dict[str, Any]:
    status = str(summary.get("status") or "")
    mode = str(summary.get("mode") or "")
    batch_type = str(summary.get("batch_type") or "")

    if status != "success":
        return {
            "action": "inspect_failures",
            "reason": "One or more child runs failed; inspect failed child logs before promotion.",
            "command": commands.get("inspect_failures")
            or "quanttradeai runs list --type batch --status failed --json",
            "follow_up_command": commands.get("compare_top_runs"),
        }
    if winner is None:
        return {
            "action": "inspect_failures",
            "reason": "No successful child runs were available to promote or compare.",
            "command": commands.get("inspect_failures"),
            "follow_up_command": None,
        }
    if batch_type == "sweep" and mode == "backtest":
        return {
            "action": "materialize_sweep_winner",
            "reason": "Sweep completed; materialize the top-ranked variant into the base agent.",
            "command": commands.get("promote_winner"),
            "follow_up_command": commands.get("run_promoted_paper_agent"),
        }
    if mode == "backtest":
        return {
            "action": "promote_winner_to_paper",
            "reason": "Batch backtest completed; promote the top-ranked agent to paper mode.",
            "command": commands.get("promote_winner"),
            "follow_up_command": commands.get("run_promoted_paper_agent"),
        }
    if mode == "paper":
        return {
            "action": "promote_winner_to_live",
            "reason": "Paper batch completed; promote the top-ranked paper run to live mode.",
            "command": commands.get("promote_winner_to_live"),
            "follow_up_command": None,
        }
    if mode == "live":
        return {
            "action": "inspect_live_batch",
            "reason": "Live batch completed; inspect metrics, decisions, executions, and logs.",
            "command": commands.get("inspect_live_batch"),
            "follow_up_command": None,
        }
    return {
        "action": "inspect_batch",
        "reason": "Batch completed.",
        "command": commands.get("inspect_batch"),
        "follow_up_command": None,
    }


def _build_key_metrics(
    summary: dict[str, Any],
    *,
    metrics_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    run_type = str(summary.get("run_type") or "")
    mode = str(summary.get("mode") or "")
    metrics = metrics_payload if isinstance(metrics_payload, dict) else None
    metrics = metrics or (
        summary.get("metrics") if isinstance(summary.get("metrics"), dict) else None
    )

    if run_type == "research":
        metrics = metrics or {}
        research_metrics = metrics.get("research_metrics_by_symbol") or {}
        backtest_metrics = metrics.get("backtest_metrics_by_symbol") or {}
        payload = {
            "metrics_status": metrics.get("status") or _metrics_status(metrics),
            "symbol_count": len(summary.get("symbols") or []),
            "accuracy": _mean_metric(research_metrics, "accuracy"),
            "f1": _mean_metric(research_metrics, "f1"),
            "net_sharpe": _mean_metric(backtest_metrics, "net_sharpe"),
            "net_pnl": _mean_metric(backtest_metrics, "net_pnl"),
            "net_mdd": _mean_metric(backtest_metrics, "net_mdd"),
        }
        return _drop_none(payload)

    if run_type == "agent" and mode == "backtest":
        aggregate = metrics or _aggregate_metrics_by_symbol(
            summary.get("metrics_by_symbol")
        )
        payload = {
            "metrics_status": _metrics_status(aggregate),
            "net_sharpe": _coerce_float(
                aggregate.get("net_sharpe") if aggregate else None
            ),
            "net_pnl": _coerce_float(aggregate.get("net_pnl") if aggregate else None),
            "net_mdd": _coerce_float(aggregate.get("net_mdd") if aggregate else None),
            "decision_count": _coerce_int(summary.get("decision_count")),
        }
        return _drop_none(payload)

    if run_type == "agent" and mode in {"paper", "live"}:
        metrics = metrics or {}
        risk_status = metrics.get("risk_status")
        if isinstance(risk_status, dict):
            risk_status = risk_status.get("status")
        payload = {
            "metrics_status": metrics.get("status") or _metrics_status(metrics),
            "total_pnl": _coerce_float(metrics.get("total_pnl")),
            "portfolio_value": _coerce_float(metrics.get("portfolio_value")),
            "execution_count": _coerce_int(
                metrics.get("execution_count") or summary.get("execution_count")
            ),
            "decision_count": _coerce_int(
                metrics.get("decision_count") or summary.get("decision_count")
            ),
            "risk_status": risk_status,
        }
        return _drop_none(payload)

    return _drop_none({"metrics_status": _metrics_status(metrics or {})})


def _important_artifacts(summary: dict[str, Any]) -> dict[str, str]:
    artifacts = dict(summary.get("artifacts") or {})
    run_dir = summary.get("run_dir")
    if run_dir and "summary" not in artifacts:
        artifacts["summary"] = str(Path(str(run_dir)) / "summary.json")

    important = {
        key: str(artifacts[key])
        for key in IMPORTANT_ARTIFACT_KEYS
        if key in artifacts and artifacts[key]
    }
    return important


def _rank_successful_results(
    results: list[dict[str, Any]],
    *,
    scoreboard_order: list[str],
) -> list[dict[str, Any]]:
    successes = [entry for entry in results if entry.get("status") == "success"]
    order = {run_id: index for index, run_id in enumerate(scoreboard_order)}
    return sorted(
        successes,
        key=lambda entry: order.get(str(entry.get("run_id") or ""), len(order)),
    )


def _candidate_payload(
    entry: dict[str, Any],
    *,
    scoreboard_sort_by: str,
) -> dict[str, Any]:
    scoreboard = dict(entry.get("scoreboard") or {})
    score = scoreboard.get(scoreboard_sort_by)
    return _drop_none(
        {
            "agent_name": entry.get("agent_name"),
            "base_agent_name": entry.get("base_agent_name"),
            "run_id": entry.get("run_id"),
            "run_dir": entry.get("run_dir"),
            "status": entry.get("status"),
            "parameters": entry.get("parameters"),
            "score_metric": scoreboard_sort_by or None,
            "score": score,
            "artifacts": _important_child_artifacts(entry),
            "promote_command": entry.get("promote_command"),
            "execution_backend": entry.get("execution_backend"),
            "broker_provider": entry.get("broker_provider"),
        }
    )


def _failure_entry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "agent_name": entry.get("agent_name"),
            "base_agent_name": entry.get("base_agent_name"),
            "run_id": entry.get("run_id"),
            "run_dir": entry.get("run_dir"),
            "status": entry.get("status"),
            "error": entry.get("error"),
            "warnings": list(entry.get("warnings") or []),
            "parameters": entry.get("parameters"),
            "artifacts": _important_child_artifacts(entry),
            "stdout_log": entry.get("stdout_log"),
            "stderr_log": entry.get("stderr_log"),
        }
    )


def _important_child_artifacts(entry: dict[str, Any]) -> dict[str, str]:
    artifacts = dict(entry.get("artifacts") or {})
    run_dir = entry.get("run_dir")
    if run_dir and "summary" not in artifacts:
        artifacts["summary"] = str(Path(str(run_dir)) / "summary.json")
    return {
        key: str(value)
        for key, value in artifacts.items()
        if key in IMPORTANT_ARTIFACT_KEYS and value
    }


def _failure_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "error": summary.get("error"),
            "run_id": summary.get("run_id"),
            "run_dir": summary.get("run_dir"),
            "warnings": list(summary.get("warnings") or []),
        }
    )


def _aggregate_metrics_by_symbol(metrics_by_symbol: Any) -> dict[str, Any]:
    if not isinstance(metrics_by_symbol, dict) or not metrics_by_symbol:
        return {}
    if isinstance(metrics_by_symbol.get("portfolio"), dict):
        return dict(metrics_by_symbol["portfolio"])
    first_metrics = next(
        (
            metrics
            for metrics in metrics_by_symbol.values()
            if isinstance(metrics, dict)
        ),
        {},
    )
    return dict(first_metrics)


def _mean_metric(metrics_by_symbol: Any, metric_name: str) -> float | None:
    if not isinstance(metrics_by_symbol, dict):
        return None
    values = []
    for metrics in metrics_by_symbol.values():
        if not isinstance(metrics, dict):
            continue
        value = _coerce_float(metrics.get(metric_name))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _metrics_status(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "missing"
    if metrics.get("status"):
        return str(metrics["status"])
    return "available"


def _agent_name(summary: dict[str, Any]) -> str | None:
    for key in ("agent_name", "name"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _config_path(project_config_path: str | Path) -> str:
    return Path(str(project_config_path)).as_posix()


def _unique_strings(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        unique.append(text)
        seen.add(text)
    return unique


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
