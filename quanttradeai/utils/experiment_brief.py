"""Deterministic experiment briefs for agent batch runs."""

from __future__ import annotations

import json
from typing import Any


EXPERIMENT_BRIEF_KIND = "quanttradeai.experiment_brief"
EXPERIMENT_BRIEF_SCHEMA_VERSION = 1


def _unique_strings(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        unique.append(text)
        seen.add(text)
    return unique


def _scoreboard_metric(scoreboard: dict[str, Any], sort_by: str) -> Any:
    if sort_by in scoreboard:
        return scoreboard.get(sort_by)
    return scoreboard.get("primary_metric")


def _entry_candidate(
    entry: dict[str, Any],
    *,
    rank: int,
    scoreboard_sort_by: str,
) -> dict[str, Any]:
    scoreboard = dict(entry.get("scoreboard") or {})
    payload = {
        "rank": rank,
        "agent_name": entry.get("agent_name"),
        "base_agent_name": entry.get("base_agent_name"),
        "run_id": entry.get("run_id"),
        "run_dir": entry.get("run_dir"),
        "status": entry.get("status"),
        "parameters": entry.get("parameters"),
        "scoreboard_sort_by": scoreboard_sort_by,
        "score": _scoreboard_metric(scoreboard, scoreboard_sort_by),
        "scoreboard": scoreboard,
        "artifacts": dict(entry.get("artifacts") or {}),
        "stdout_log": entry.get("stdout_log"),
        "stderr_log": entry.get("stderr_log"),
    }
    if entry.get("variant_project_config"):
        payload["variant_project_config"] = entry.get("variant_project_config")
    if entry.get("promote_command"):
        payload["promote_command"] = entry.get("promote_command")
    return payload


def _ordered_successful_entries(
    *,
    results: list[dict[str, Any]],
    scoreboard_order: list[str],
) -> list[dict[str, Any]]:
    by_run_id = {
        str(entry.get("run_id") or ""): entry
        for entry in results
        if entry.get("run_id")
    }
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for run_id in scoreboard_order:
        entry = by_run_id.get(str(run_id))
        if not entry or entry.get("status") != "success":
            continue
        ordered.append(entry)
        seen.add(id(entry))

    for entry in results:
        if id(entry) in seen or entry.get("status") != "success":
            continue
        ordered.append(entry)
    return ordered


def _failure_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_name": entry.get("agent_name"),
        "base_agent_name": entry.get("base_agent_name"),
        "run_id": entry.get("run_id"),
        "run_dir": entry.get("run_dir"),
        "status": entry.get("status"),
        "error": entry.get("error"),
        "warnings": list(entry.get("warnings") or []),
        "parameters": entry.get("parameters"),
        "artifacts": dict(entry.get("artifacts") or {}),
        "stdout_log": entry.get("stdout_log"),
        "stderr_log": entry.get("stderr_log"),
    }


def _compare_command(
    *,
    successful_entries: list[dict[str, Any]],
    scoreboard_sort_by: str,
) -> str | None:
    run_ids = [
        str(entry.get("run_id") or "").strip()
        for entry in successful_entries[:2]
        if entry.get("run_id")
    ]
    if len(run_ids) < 2:
        return None
    return (
        "quanttradeai runs list "
        f"--compare {run_ids[0]} --compare {run_ids[1]} "
        f"--sort-by {scoreboard_sort_by}"
    )


def _next_action_for_winner(
    *,
    batch_type: str,
    mode: str,
    winner: dict[str, Any] | None,
    project_config_path: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if winner is None:
        return (
            {
                "action": "inspect_failures",
                "reason": "No successful child runs were available to promote or compare.",
                "command": None,
                "follow_up_command": None,
            },
            {},
        )

    run_id = str(winner.get("run_id") or "")
    agent_name = str(winner.get("agent_name") or "")
    base_agent_name = str(winner.get("base_agent_name") or agent_name)
    commands: dict[str, str] = {}

    if batch_type == "sweep" and mode == "backtest":
        promote_command = str(winner.get("promote_command") or "").strip()
        if not promote_command:
            promote_command = (
                "quanttradeai promote " f"--run {run_id} -c {project_config_path}"
            )
        paper_command = (
            "quanttradeai agent run "
            f"--agent {base_agent_name} -c {project_config_path} --mode paper"
        )
        commands["promote_winner"] = promote_command
        commands["run_promoted_paper_agent"] = paper_command
        return (
            {
                "action": "materialize_sweep_winner",
                "reason": "Promote the best sweep child into the base agent, then run the promoted paper agent.",
                "command": promote_command,
                "follow_up_command": paper_command,
            },
            commands,
        )

    if mode == "backtest":
        promote_command = (
            "quanttradeai promote " f"--run {run_id} -c {project_config_path}"
        )
        commands["promote_winner"] = promote_command
        return (
            {
                "action": "promote_winner_to_paper",
                "reason": "Promote the best successful backtest run to paper mode.",
                "command": promote_command,
                "follow_up_command": None,
            },
            commands,
        )

    if mode == "paper":
        promote_command = (
            "quanttradeai promote "
            f"--run {run_id} -c {project_config_path} "
            f"--to live --acknowledge-live {agent_name}"
        )
        commands["promote_winner_to_live"] = promote_command
        return (
            {
                "action": "promote_winner_to_live",
                "reason": "Promote the best successful paper run to live mode with explicit acknowledgement.",
                "command": promote_command,
                "follow_up_command": None,
            },
            commands,
        )

    return (
        {
            "action": "inspect_live_batch",
            "reason": "Live batch runs should be inspected through metrics, decisions, executions, and logs.",
            "command": None,
            "follow_up_command": None,
        },
        commands,
    )


def build_experiment_brief(
    *,
    batch: dict[str, Any],
    results: list[dict[str, Any]],
    scoreboard_order: list[str],
    scoreboard_sort_by: str,
    project_config_path: str,
    artifacts: dict[str, Any],
    warnings: list[str] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Build a stable, agent-readable experiment brief payload."""

    batch_type = str(batch.get("batch_type") or "")
    mode = str(batch.get("mode") or "")
    successful_entries = _ordered_successful_entries(
        results=results,
        scoreboard_order=scoreboard_order,
    )
    ranked_candidates = [
        _entry_candidate(
            entry,
            rank=index,
            scoreboard_sort_by=scoreboard_sort_by,
        )
        for index, entry in enumerate(successful_entries[:top_n], start=1)
    ]
    winner = ranked_candidates[0] if ranked_candidates else None
    failures = [
        _failure_entry(entry) for entry in results if entry.get("status") != "success"
    ]

    recommended_next_action, commands = _next_action_for_winner(
        batch_type=batch_type,
        mode=mode,
        winner=winner,
        project_config_path=project_config_path,
    )
    compare = _compare_command(
        successful_entries=successful_entries,
        scoreboard_sort_by=scoreboard_sort_by,
    )
    if compare:
        commands["compare_top_runs"] = compare

    all_warnings: list[Any] = list(warnings or [])
    for entry in results:
        all_warnings.extend(entry.get("warnings") or [])

    return {
        "kind": EXPERIMENT_BRIEF_KIND,
        "schema_version": EXPERIMENT_BRIEF_SCHEMA_VERSION,
        "batch": dict(batch),
        "ranking": {
            "scoreboard_sort_by": scoreboard_sort_by,
            "scoreboard_order": list(scoreboard_order),
            "top_candidates": ranked_candidates,
            "successful_count": len(successful_entries),
        },
        "winner": winner,
        "failures": failures,
        "recommended_next_action": recommended_next_action,
        "commands": commands,
        "artifacts": dict(artifacts),
        "warnings": _unique_strings(all_warnings),
    }


def _render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_experiment_brief_markdown(brief: dict[str, Any]) -> str:
    """Render the experiment brief as compact Markdown."""

    batch = dict(brief.get("batch") or {})
    ranking = dict(brief.get("ranking") or {})
    winner = brief.get("winner")
    action = dict(brief.get("recommended_next_action") or {})
    commands = dict(brief.get("commands") or {})
    artifacts = dict(brief.get("artifacts") or {})
    failures = list(brief.get("failures") or [])
    warnings = list(brief.get("warnings") or [])

    lines = [
        "# Experiment Brief",
        "",
        "## Batch",
        "",
        f"- Run ID: {_render_value(batch.get('run_id'))}",
        f"- Type: {_render_value(batch.get('batch_type'))}",
        f"- Mode: {_render_value(batch.get('mode'))}",
        f"- Status: {_render_value(batch.get('status'))}",
        f"- Project: {_render_value(batch.get('project_name'))}",
        f"- Successes: {_render_value(batch.get('success_count'))}",
        f"- Failures: {_render_value(batch.get('failure_count'))}",
        "",
        "## Recommendation",
        "",
        f"- Action: {_render_value(action.get('action'))}",
        f"- Reason: {_render_value(action.get('reason'))}",
    ]
    if action.get("command"):
        lines.append(f"- Command: `{action['command']}`")
    if action.get("follow_up_command"):
        lines.append(f"- Follow-up: `{action['follow_up_command']}`")

    lines.extend(["", "## Winner", ""])
    if isinstance(winner, dict):
        lines.extend(
            [
                f"- Agent: {_render_value(winner.get('agent_name'))}",
                f"- Run ID: {_render_value(winner.get('run_id'))}",
                f"- Score ({ranking.get('scoreboard_sort_by')}): {_render_value(winner.get('score'))}",
                f"- Parameters: {_render_value(winner.get('parameters'))}",
                f"- Run Dir: {_render_value(winner.get('run_dir'))}",
            ]
        )
    else:
        lines.append("- No successful winner.")

    lines.extend(["", "## Top Candidates", ""])
    candidates = list(ranking.get("top_candidates") or [])
    if not candidates:
        lines.append("- None.")
    else:
        for candidate in candidates:
            lines.append(
                "- "
                f"{candidate.get('rank')}. {candidate.get('agent_name')} "
                f"({candidate.get('run_id')}): "
                f"{ranking.get('scoreboard_sort_by')}={_render_value(candidate.get('score'))}"
            )

    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("- None.")
    else:
        for failure in failures:
            detail = failure.get("error") or "failed"
            lines.append(
                f"- {failure.get('agent_name')} ({failure.get('run_id')}): {detail}"
            )
            if failure.get("stderr_log"):
                lines.append(f"  stderr: {failure['stderr_log']}")

    lines.extend(["", "## Commands", ""])
    if not commands:
        lines.append("- None.")
    else:
        for name, command in commands.items():
            lines.append(f"- {name}: `{command}`")

    lines.extend(["", "## Artifacts", ""])
    for name, path in artifacts.items():
        lines.append(f"- {name}: {_render_value(path)}")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"
