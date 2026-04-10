"""Run-backed promotion helpers for canonical project workflows."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from quanttradeai.utils.config_schemas import ProjectConfigSchema
from quanttradeai.utils.project_config import extract_canonical_live_risk_config
from quanttradeai.utils.run_records import RUNS_ROOT, discover_runs


def _normalize_run_id(value: str | Path) -> str:
    """Normalize a run id/path for matching records discovered under runs/."""

    normalized = str(value).replace("\\", "/").strip().strip("/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if "runs" in parts:
        parts = parts[parts.index("runs") + 1 :]
    return "/".join(parts)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Run summary not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run summary is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Run summary must contain a JSON object: {path}")
    return payload


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Project config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Project config must contain a mapping at root: {path}")
    return payload


def _find_run_record(run_id: str | Path, *, runs_root: Path | str) -> dict[str, Any]:
    expected = _normalize_run_id(run_id)
    for record in discover_runs(runs_root):
        candidates = {
            _normalize_run_id(str(record.get("run_id") or "")),
            _normalize_run_id(str(record.get("run_dir") or "")),
        }
        if expected in candidates:
            return record

    raise ValueError(f"Run not found: {run_id}")


def _summary_path_for_record(record: dict[str, Any], runs_root: Path | str) -> Path:
    summary_path = Path(str(record.get("run_dir") or "")) / "summary.json"
    if summary_path.is_file():
        return summary_path

    run_id = str(record.get("run_id") or "")
    return (
        Path(runs_root).joinpath(*_normalize_run_id(run_id).split("/")) / "summary.json"
    )


def _validate_promotable_run(
    record: dict[str, Any],
    summary: dict[str, Any],
    *,
    target_mode: str,
) -> tuple[str, str]:
    run_type = str(summary.get("run_type") or record.get("run_type") or "")
    mode = str(summary.get("mode") or record.get("mode") or "")
    status = str(summary.get("status") or record.get("status") or "")

    source_mode_by_target = {
        "paper": "backtest",
        "live": "paper",
    }
    required_source_mode = source_mode_by_target[target_mode]

    if run_type != "agent" or mode != required_source_mode or status != "success":
        raise ValueError(
            f"Only successful agent {required_source_mode} runs can be promoted to {target_mode}."
        )

    agent_name = str(
        summary.get("agent_name") or summary.get("name") or record.get("name") or ""
    ).strip()
    if not agent_name:
        raise ValueError("Promotable agent run is missing an agent name.")

    return agent_name, required_source_mode


def _apply_paper_promotion(
    project_config: dict[str, Any],
    *,
    agent_name: str,
) -> tuple[dict[str, Any], bool]:
    proposed = copy.deepcopy(project_config)
    agents = proposed.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Project config must include an agents list.")

    target_agent: dict[str, Any] | None = None
    for agent in agents:
        if isinstance(agent, dict) and agent.get("name") == agent_name:
            target_agent = agent
            break

    if target_agent is None:
        raise ValueError(f"Agent '{agent_name}' not found in project config.")

    changed = False
    if target_agent.get("mode") != "paper":
        target_agent["mode"] = "paper"
        changed = True

    deployment = proposed.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("Project config must include a deployment mapping.")
    if deployment.get("mode") != "paper":
        deployment["mode"] = "paper"
        changed = True

    streaming = (proposed.get("data") or {}).get("streaming")
    if not isinstance(streaming, dict) or streaming.get("enabled") is not True:
        raise ValueError(
            "data.streaming.enabled must be true before promoting an agent to paper."
        )

    try:
        ProjectConfigSchema(**proposed)
    except ValidationError as exc:
        raise ValueError(f"Promoted project config is invalid: {exc}") from exc

    return proposed, changed


def _apply_live_promotion(
    project_config: dict[str, Any],
    *,
    agent_name: str,
) -> tuple[dict[str, Any], bool]:
    proposed = copy.deepcopy(project_config)
    agents = proposed.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Project config must include an agents list.")

    target_agent: dict[str, Any] | None = None
    for agent in agents:
        if isinstance(agent, dict) and agent.get("name") == agent_name:
            target_agent = agent
            break

    if target_agent is None:
        raise ValueError(f"Agent '{agent_name}' not found in project config.")

    changed = False
    if target_agent.get("mode") != "live":
        target_agent["mode"] = "live"
        changed = True

    if (
        not isinstance((proposed.get("data") or {}).get("streaming"), dict)
        or (proposed.get("data") or {}).get("streaming", {}).get("enabled") is not True
    ):
        raise ValueError(
            "data.streaming.enabled must be true before promoting an agent to live."
        )
    risk_cfg, used_nested_fallback = extract_canonical_live_risk_config(proposed)
    if not risk_cfg:
        raise ValueError("risk is required before promoting an agent to live.")
    if used_nested_fallback:
        proposed["risk"] = risk_cfg
    if not isinstance(proposed.get("position_manager"), dict):
        raise ValueError(
            "position_manager is required before promoting an agent to live."
        )

    try:
        ProjectConfigSchema(**proposed)
    except ValidationError as exc:
        raise ValueError(f"Promoted project config is invalid: {exc}") from exc

    return proposed, changed


def promote_agent_run(
    *,
    run_id: str | Path,
    config_path: str | Path = "config/project.yaml",
    target_mode: str = "paper",
    dry_run: bool = False,
    acknowledge_live: str | None = None,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any]:
    """Promote a successful project-defined agent run to paper or live."""

    target_mode = target_mode.strip().lower()
    if target_mode not in {"paper", "live"}:
        raise ValueError("Only promotion to paper or live is supported.")

    record = _find_run_record(run_id, runs_root=runs_root)
    summary_path = _summary_path_for_record(record, runs_root)
    summary = _load_json(summary_path)
    agent_name, source_mode = _validate_promotable_run(
        record,
        summary,
        target_mode=target_mode,
    )
    if target_mode == "live":
        if acknowledge_live != agent_name:
            raise ValueError(
                f"Promoting to live requires --acknowledge-live {agent_name}"
            )

    project_config_path = Path(config_path)
    project_config = _load_yaml_mapping(project_config_path)
    if target_mode == "paper":
        promoted_config, changed = _apply_paper_promotion(
            project_config,
            agent_name=agent_name,
        )
    else:
        promoted_config, changed = _apply_live_promotion(
            project_config,
            agent_name=agent_name,
        )

    if changed and not dry_run:
        project_config_path.write_text(
            yaml.safe_dump(promoted_config, sort_keys=False),
            encoding="utf-8",
        )

    config_path_display = project_config_path.as_posix()
    next_command = (
        f"quanttradeai agent run --agent {agent_name} "
        f"-c {config_path_display} --mode {target_mode}"
    )
    return {
        "status": "success",
        "source_run_id": str(record.get("run_id") or _normalize_run_id(run_id)),
        "agent_name": agent_name,
        "from_mode": source_mode,
        "to_mode": target_mode,
        "changed": changed,
        "config_path": config_path_display,
        "dry_run": dry_run,
        "next_command": next_command,
    }
