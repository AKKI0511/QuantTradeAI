"""Agent-readable briefs for single QuantTradeAI runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from quanttradeai.utils.run_records import normalize_run_summary
from quanttradeai.utils.run_scoreboard import load_scoreboard_record


RUN_BRIEF_KIND = "quanttradeai.run_brief"
RUN_BRIEF_SCHEMA_VERSION = 1


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


def _config_command_path(project_config_path: str | Path) -> str:
    return Path(project_config_path).resolve().as_posix()


def _summary_artifacts(summary: dict[str, Any]) -> dict[str, Any]:
    return dict(summary.get("artifacts") or {})


def _load_resolved_project_config(summary: dict[str, Any]) -> dict[str, Any]:
    resolved_path = _summary_artifacts(summary).get("resolved_project_config")
    if not resolved_path:
        return {}

    try:
        payload = yaml.safe_load(Path(str(resolved_path)).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scoreboard_for_summary(
    summary: dict[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    record = normalize_run_summary(summary, run_dir=run_dir)
    if record is None:
        return {
            "status": "missing",
            "error": "Run summary could not be normalized for scoreboard loading.",
        }
    return load_scoreboard_record(record)


def _deployment_target(project_config: dict[str, Any]) -> str | None:
    deployment = dict(project_config.get("deployment") or {})
    target = str(deployment.get("target") or "").strip()
    return target or None


def _recommended_next_action(
    *,
    summary: dict[str, Any],
    project_config_path: str,
    resolved_project_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    status = str(summary.get("status") or "").strip().lower()
    run_id = str(summary.get("run_id") or "").strip()
    run_type = str(summary.get("run_type") or "").strip().lower()
    mode = str(summary.get("mode") or "").strip().lower()
    name = str(summary.get("name") or "").strip()
    agent_name = str(summary.get("agent_name") or name).strip()
    config_path = _config_command_path(project_config_path)

    if status != "success":
        return (
            {
                "action": "inspect_failure",
                "reason": "Run failed. Inspect the error, warnings, and artifact paths before retrying.",
                "command": None,
                "follow_up_command": None,
            },
            {},
        )

    if run_type == "research":
        commands = {
            "rank_research_runs": (
                "quanttradeai runs list --type research --scoreboard --sort-by net_sharpe"
            ),
            "promote_this_run": f"quanttradeai promote --run {run_id} -c {config_path}",
        }
        return (
            {
                "action": "promote_research_run",
                "reason": "Promote this successful research artifact if its metrics are acceptable.",
                "command": commands["promote_this_run"],
                "follow_up_command": commands["rank_research_runs"],
            },
            commands,
        )

    if run_type == "agent" and mode == "backtest":
        commands = {
            "rank_agent_backtests": (
                "quanttradeai runs list --type agent --mode backtest --scoreboard --sort-by net_sharpe"
            ),
            "promote_to_paper": f"quanttradeai promote --run {run_id} -c {config_path}",
        }
        return (
            {
                "action": "promote_to_paper",
                "reason": "Promote this successful agent backtest to paper mode if the run is the preferred candidate.",
                "command": commands["promote_to_paper"],
                "follow_up_command": commands["rank_agent_backtests"],
            },
            commands,
        )

    if run_type == "agent" and mode == "paper":
        commands = {
            "rank_paper_runs": (
                "quanttradeai runs list --type agent --mode paper --scoreboard --sort-by total_pnl"
            ),
            "promote_to_live": (
                f"quanttradeai promote --run {run_id} -c {config_path} "
                f"--to live --acknowledge-live {agent_name}"
            ),
        }
        target = _deployment_target(resolved_project_config)
        if target:
            commands["deploy_agent"] = (
                f"quanttradeai deploy --agent {agent_name} -c {config_path} --target {target}"
            )
        return (
            {
                "action": "promote_to_live",
                "reason": "Promote this successful paper run to live mode after reviewing execution and risk artifacts.",
                "command": commands["promote_to_live"],
                "follow_up_command": commands.get("deploy_agent"),
            },
            commands,
        )

    if run_type == "agent" and mode == "live":
        commands = {
            "rank_live_runs": (
                "quanttradeai runs list --type agent --mode live --scoreboard --sort-by total_pnl"
            )
        }
        return (
            {
                "action": "inspect_live_run",
                "reason": "Inspect live metrics, decisions, executions, risk state, and logs before taking operational action.",
                "command": commands["rank_live_runs"],
                "follow_up_command": None,
            },
            commands,
        )

    return (
        {
            "action": "inspect_run",
            "reason": "Inspect the run artifacts and metrics before choosing the next workflow step.",
            "command": None,
            "follow_up_command": None,
        },
        {},
    )


def build_run_brief(
    *,
    summary: dict[str, Any],
    run_dir: str | Path,
    project_config_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic brief for one research or agent run."""

    run_dir_path = Path(run_dir)
    resolved_project_config = _load_resolved_project_config(summary)
    next_action, commands = _recommended_next_action(
        summary=summary,
        project_config_path=str(project_config_path),
        resolved_project_config=resolved_project_config,
    )
    run_payload = {
        "run_id": summary.get("run_id"),
        "run_type": summary.get("run_type"),
        "mode": summary.get("mode"),
        "name": summary.get("name"),
        "status": summary.get("status"),
        "run_dir": str(summary.get("run_dir") or run_dir_path),
        "timestamps": dict(summary.get("timestamps") or {}),
        "symbols": list(summary.get("symbols") or []),
    }
    for key in ("project_name", "project_profile", "agent_name", "agent_kind"):
        if summary.get(key) is not None:
            run_payload[key] = summary.get(key)

    return {
        "kind": RUN_BRIEF_KIND,
        "schema_version": RUN_BRIEF_SCHEMA_VERSION,
        "run": run_payload,
        "scoreboard": _scoreboard_for_summary(summary, run_dir=run_dir_path),
        "recommended_next_action": next_action,
        "commands": commands,
        "artifacts": _summary_artifacts(summary),
        "warnings": _unique_strings(list(summary.get("warnings") or [])),
        "error": summary.get("error"),
    }


def _render_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_run_brief_markdown(brief: dict[str, Any]) -> str:
    """Render a compact Markdown view of a single-run brief."""

    run = dict(brief.get("run") or {})
    scoreboard = dict(brief.get("scoreboard") or {})
    action = dict(brief.get("recommended_next_action") or {})
    commands = dict(brief.get("commands") or {})
    artifacts = dict(brief.get("artifacts") or {})
    warnings = list(brief.get("warnings") or [])

    lines = [
        "# Run Brief",
        "",
        "## Run",
        "",
        f"- Run ID: {_render_value(run.get('run_id'))}",
        f"- Type: {_render_value(run.get('run_type'))}",
        f"- Mode: {_render_value(run.get('mode'))}",
        f"- Name: {_render_value(run.get('name'))}",
        f"- Status: {_render_value(run.get('status'))}",
        f"- Symbols: {_render_value(run.get('symbols'))}",
        f"- Run Dir: {_render_value(run.get('run_dir'))}",
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

    lines.extend(["", "## Scoreboard", ""])
    for key in (
        "primary_metric_name",
        "primary_metric",
        "accuracy",
        "f1",
        "net_sharpe",
        "net_pnl",
        "total_pnl",
        "portfolio_value",
        "execution_count",
        "decision_count",
        "risk_status",
        "status",
        "error",
    ):
        if key in scoreboard and scoreboard.get(key) is not None:
            lines.append(f"- {key}: {_render_value(scoreboard.get(key))}")

    lines.extend(["", "## Commands", ""])
    if commands:
        for name, command in commands.items():
            lines.append(f"- {name}: `{command}`")
    else:
        lines.append("- None.")

    lines.extend(["", "## Artifacts", ""])
    if artifacts:
        for name, path in artifacts.items():
            lines.append(f"- {name}: {_render_value(path)}")
    else:
        lines.append("- None.")

    if brief.get("error"):
        lines.extend(["", "## Error", "", f"- {_render_value(brief.get('error'))}"])

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"


def write_run_brief_artifacts(
    summary: dict[str, Any],
    run_dir: str | Path,
    project_config_path: str | Path,
) -> dict[str, str]:
    """Write run_brief JSON/Markdown artifacts and attach them to summary."""

    run_dir_path = Path(run_dir)
    run_brief_json_path = run_dir_path / "run_brief.json"
    run_brief_md_path = run_dir_path / "run_brief.md"
    artifacts = dict(summary.get("artifacts") or {})
    artifacts["run_brief_json"] = str(run_brief_json_path)
    artifacts["run_brief_md"] = str(run_brief_md_path)
    summary["artifacts"] = artifacts

    brief = build_run_brief(
        summary=summary,
        run_dir=run_dir_path,
        project_config_path=project_config_path,
    )
    run_brief_json_path.write_text(
        json.dumps(brief, indent=2, default=str),
        encoding="utf-8",
    )
    run_brief_md_path.write_text(
        render_run_brief_markdown(brief),
        encoding="utf-8",
    )
    return {
        "run_brief_json": str(run_brief_json_path),
        "run_brief_md": str(run_brief_md_path),
    }
