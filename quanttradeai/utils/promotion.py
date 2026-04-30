"""Run-backed promotion helpers for canonical project workflows."""

from __future__ import annotations

import copy
import json
import posixpath
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from quanttradeai.utils.config_schemas import ProjectConfigSchema
from quanttradeai.utils.project_config import extract_canonical_live_risk_config
from quanttradeai.utils.project_paths import infer_project_root, resolve_project_path
from quanttradeai.utils.run_records import RUNS_ROOT, discover_runs
from quanttradeai.utils.sweeps import apply_agent_scalar_overrides


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


def _validate_agent_promotable_run(
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

    sweep_metadata = dict(summary.get("sweep") or {})
    if required_source_mode == "backtest" and sweep_metadata.get("promotable") is False:
        sweep_name = str(sweep_metadata.get("name") or "unnamed_sweep")
        base_agent_name = str(
            sweep_metadata.get("base_agent_name")
            or summary.get("agent_name")
            or record.get("name")
            or ""
        ).strip()
        raise ValueError(
            "Sweep-generated backtest runs cannot be promoted directly. "
            f"Copy the winning parameters from sweep '{sweep_name}' into config/project.yaml, "
            f"rerun agent '{base_agent_name}' normally, and then promote that non-sweep backtest run."
        )

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


def _validate_sweep_apply_run(
    record: dict[str, Any],
    summary: dict[str, Any],
    *,
    target_mode: str,
    acknowledge_live: str | None,
) -> tuple[str, str, dict[str, Any]]:
    if target_mode != "paper":
        raise ValueError("--apply-sweep does not support --to live.")
    if acknowledge_live is not None:
        raise ValueError(
            "--acknowledge-live is only supported when promoting agent paper runs to live."
        )

    run_type = str(summary.get("run_type") or record.get("run_type") or "")
    mode = str(summary.get("mode") or record.get("mode") or "")
    status = str(summary.get("status") or record.get("status") or "")
    if run_type != "agent" or mode != "backtest" or status != "success":
        raise ValueError(
            "Only successful agent backtest sweep child runs can be applied to project config."
        )

    sweep_metadata = summary.get("sweep")
    if not isinstance(sweep_metadata, dict):
        raise ValueError("--apply-sweep requires a sweep-generated agent backtest run.")

    base_agent_name = str(sweep_metadata.get("base_agent_name") or "").strip()
    if not base_agent_name:
        raise ValueError("Sweep child run summary is missing sweep.base_agent_name.")

    parameters = sweep_metadata.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Sweep child run summary is missing sweep.parameters.")

    sweep_name = str(sweep_metadata.get("name") or "").strip() or "unnamed_sweep"
    return sweep_name, base_agent_name, dict(parameters)


def _validate_research_promotable_run(
    record: dict[str, Any],
    summary: dict[str, Any],
    *,
    target_mode: str,
    acknowledge_live: str | None,
) -> tuple[str, str]:
    run_type = str(summary.get("run_type") or record.get("run_type") or "")
    mode = str(summary.get("mode") or record.get("mode") or "")
    status = str(summary.get("status") or record.get("status") or "")

    if run_type != "research" or mode != "research" or status != "success":
        raise ValueError(
            "Only successful research runs can promote models into stable project paths."
        )
    if target_mode != "paper":
        raise ValueError(
            "Research run promotion does not support --to. Omit --to and use the default promote behavior."
        )
    if acknowledge_live is not None:
        raise ValueError(
            "--acknowledge-live is only supported when promoting agent paper runs to live."
        )

    project_name = str(
        summary.get("project_name") or summary.get("name") or record.get("name") or ""
    ).strip()
    if not project_name:
        raise ValueError("Promotable research run is missing a project name.")

    return project_name, "research"


def _apply_sweep_parameters(
    project_config: dict[str, Any],
    *,
    base_agent_name: str,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    proposed = copy.deepcopy(project_config)
    agents = proposed.get("agents")
    if not isinstance(agents, list):
        raise ValueError("Project config must include an agents list.")

    target_index: int | None = None
    target_agent: dict[str, Any] | None = None
    for index, agent in enumerate(agents):
        if isinstance(agent, dict) and agent.get("name") == base_agent_name:
            target_index = index
            target_agent = agent
            break

    if target_agent is None or target_index is None:
        raise ValueError(f"Base agent '{base_agent_name}' not found in project config.")

    updated_agent = apply_agent_scalar_overrides(target_agent, parameters)
    updated_agent["name"] = base_agent_name
    changed = updated_agent != target_agent
    agents[target_index] = updated_agent
    return proposed, changed


def _validate_proposed_project_config(
    *,
    proposed_config: dict[str, Any],
    config_path: Path,
) -> None:
    try:
        ProjectConfigSchema(**proposed_config)
    except ValidationError as exc:
        raise ValueError(f"Proposed project config is invalid: {exc}") from exc

    from quanttradeai.utils.config_validator import validate_project_config

    with tempfile.TemporaryDirectory() as temp_dir:
        validate_project_config(
            config_path=config_path,
            output_dir=temp_dir,
            project_config_override=proposed_config,
            timestamp_subdir=False,
        )


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


def _apply_sweep_child_run(
    *,
    record: dict[str, Any],
    summary: dict[str, Any],
    config_path: Path,
    target_mode: str,
    dry_run: bool,
    acknowledge_live: str | None,
) -> dict[str, Any]:
    sweep_name, base_agent_name, parameters = _validate_sweep_apply_run(
        record,
        summary,
        target_mode=target_mode,
        acknowledge_live=acknowledge_live,
    )
    project_config = _load_yaml_mapping(config_path)
    proposed_config, changed = _apply_sweep_parameters(
        project_config,
        base_agent_name=base_agent_name,
        parameters=parameters,
    )
    _validate_proposed_project_config(
        proposed_config=proposed_config,
        config_path=config_path,
    )

    if changed and not dry_run:
        config_path.write_text(
            yaml.safe_dump(proposed_config, sort_keys=False),
            encoding="utf-8",
        )

    config_path_display = config_path.as_posix()
    next_command = (
        f"quanttradeai agent run --agent {base_agent_name} "
        f"-c {config_path_display} --mode backtest"
    )
    return {
        "status": "success",
        "operation": "apply_sweep",
        "source_run_id": str(
            record.get("run_id") or _normalize_run_id(summary.get("run_id") or "")
        ),
        "sweep_name": sweep_name,
        "base_agent_name": base_agent_name,
        "applied_parameters": parameters,
        "changed": changed,
        "dry_run": dry_run,
        "config_path": config_path_display,
        "next_command": next_command,
    }


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


def _resolve_research_promotion_targets(
    project_config: dict[str, Any],
) -> list[dict[str, str]]:
    try:
        resolved_config = ProjectConfigSchema(**project_config)
    except ValidationError as exc:
        raise ValueError(f"Project config is invalid: {exc}") from exc

    targets = [
        {
            "name": target.name,
            "symbol": target.symbol,
            "path": target.path,
        }
        for target in resolved_config.research.promotion.targets
    ]
    if not targets:
        raise ValueError(
            "research.promotion.targets must define at least one target before promoting a research run."
        )

    available_symbols = set(resolved_config.data.symbols)
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for target in targets:
        if target["symbol"] not in available_symbols:
            raise ValueError(
                "research promotion target symbol must reference one of data.symbols. "
                f"Received: {target['symbol']}"
            )

        if target["name"] in seen_names:
            raise ValueError(
                f"research promotion target names must be unique. Duplicate: {target['name']}"
            )
        seen_names.add(target["name"])

        normalized_path = posixpath.normpath(Path(target["path"]).as_posix())
        if normalized_path in seen_paths:
            raise ValueError(
                "research promotion target paths must be unique. "
                f"Duplicate: {normalized_path}"
            )
        seen_paths.add(normalized_path)

    return targets


def _research_experiment_dir(
    *,
    summary: dict[str, Any],
    config_path: Path,
) -> Path:
    artifacts = dict(summary.get("artifacts") or {})
    experiment_dir_raw = str(artifacts.get("experiment_dir") or "").strip()
    if not experiment_dir_raw:
        raise ValueError(
            "Research run summary is missing artifacts.experiment_dir; cannot locate trained models to promote."
        )

    experiment_dir = resolve_project_path(config_path, experiment_dir_raw)
    if not experiment_dir.is_dir():
        raise ValueError(
            f"Research run experiment directory does not exist: {experiment_dir}"
        )
    return experiment_dir


def _resolve_promoted_model_destination(
    *,
    config_path: Path,
    relative_path: str,
) -> Path:
    raw_path = str(relative_path or "").strip()
    path = Path(raw_path)
    if not raw_path:
        raise ValueError("research promotion destination path must not be blank.")
    if path.is_absolute():
        raise ValueError(
            "research promotion destination path must be project-relative and under models/."
        )

    project_root = infer_project_root(config_path)
    resolved_path = resolve_project_path(config_path, raw_path)
    try:
        relative = resolved_path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(
            "research promotion destination path must resolve inside the project root under models/."
        ) from exc

    if not relative.parts or relative.parts[0] != "models":
        raise ValueError(
            "research promotion destination path must resolve under models/."
        )
    return resolved_path


def _stage_promoted_directory(
    *,
    source_dir: Path,
    destination_dir: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".quanttradeai-promote-{destination_dir.name}-",
            dir=destination_dir.parent,
        )
    )
    staged_dir = temp_root / destination_dir.name
    shutil.copytree(source_dir, staged_dir)
    (staged_dir / "promotion_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return temp_root, staged_dir


def _commit_promoted_directory(
    *,
    staged_dir: Path,
    temp_root: Path,
    destination_dir: Path,
) -> None:
    if destination_dir.exists() and not destination_dir.is_dir():
        raise ValueError(
            f"Promotion destination already exists and is not a directory: {destination_dir}"
        )

    backup_dir: Path | None = None
    try:
        if destination_dir.exists():
            backup_dir = destination_dir.parent / (
                f".quanttradeai-backup-{destination_dir.name}-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            )
            destination_dir.rename(backup_dir)

        staged_dir.rename(destination_dir)
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
    except Exception:
        if (
            backup_dir is not None
            and backup_dir.exists()
            and not destination_dir.exists()
        ):
            backup_dir.rename(destination_dir)
        raise
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def _promote_research_run(
    *,
    record: dict[str, Any],
    summary: dict[str, Any],
    config_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    project_config = _load_yaml_mapping(config_path)
    targets = _resolve_research_promotion_targets(project_config)
    experiment_dir = _research_experiment_dir(summary=summary, config_path=config_path)

    promoted_targets: list[dict[str, Any]] = []
    staged_targets: list[tuple[Path, Path, Path]] = []
    promoted_at = datetime.now(timezone.utc).isoformat()

    for target in targets:
        source_dir = experiment_dir / target["symbol"]
        if not source_dir.is_dir():
            raise ValueError(
                f"Research run is missing a trained model artifact for symbol '{target['symbol']}' at {source_dir}"
            )

        destination_dir = _resolve_promoted_model_destination(
            config_path=config_path,
            relative_path=target["path"],
        )
        manifest = {
            "source_run_id": str(record.get("run_id") or summary.get("run_id") or ""),
            "project_name": summary.get("project_name"),
            "target_name": target["name"],
            "symbol": target["symbol"],
            "source_path": source_dir.as_posix(),
            "destination_path": destination_dir.as_posix(),
            "promoted_at": promoted_at,
        }
        temp_root, staged_dir = _stage_promoted_directory(
            source_dir=source_dir,
            destination_dir=destination_dir,
            manifest=manifest,
        )
        staged_targets.append((temp_root, staged_dir, destination_dir))
        promoted_targets.append(
            {
                "name": target["name"],
                "symbol": target["symbol"],
                "source_path": source_dir.as_posix(),
                "destination_path": destination_dir.as_posix(),
                "manifest_path": (
                    destination_dir / "promotion_manifest.json"
                ).as_posix(),
            }
        )

    try:
        if not dry_run:
            for temp_root, staged_dir, destination_dir in staged_targets:
                _commit_promoted_directory(
                    staged_dir=staged_dir,
                    temp_root=temp_root,
                    destination_dir=destination_dir,
                )
    finally:
        if dry_run:
            for temp_root, _staged_dir, _destination_dir in staged_targets:
                if temp_root.exists():
                    shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "status": "success",
        "source_run_id": str(
            record.get("run_id") or _normalize_run_id(summary.get("run_id") or "")
        ),
        "run_type": "research",
        "project_name": summary.get("project_name"),
        "config_path": config_path.as_posix(),
        "dry_run": dry_run,
        "changed": bool(promoted_targets),
        "experiment_dir": experiment_dir.as_posix(),
        "promoted_targets": promoted_targets,
    }


def _promote_agent_run(
    *,
    record: dict[str, Any],
    summary: dict[str, Any],
    config_path: Path,
    target_mode: str,
    dry_run: bool,
    acknowledge_live: str | None,
) -> dict[str, Any]:
    agent_name, source_mode = _validate_agent_promotable_run(
        record,
        summary,
        target_mode=target_mode,
    )
    if target_mode == "live" and acknowledge_live != agent_name:
        raise ValueError(f"Promoting to live requires --acknowledge-live {agent_name}")

    project_config = _load_yaml_mapping(config_path)
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
        config_path.write_text(
            yaml.safe_dump(promoted_config, sort_keys=False),
            encoding="utf-8",
        )

    config_path_display = config_path.as_posix()
    next_command = (
        f"quanttradeai agent run --agent {agent_name} "
        f"-c {config_path_display} --mode {target_mode}"
    )
    return {
        "status": "success",
        "source_run_id": str(
            record.get("run_id") or _normalize_run_id(summary.get("run_id") or "")
        ),
        "run_type": "agent",
        "agent_name": agent_name,
        "from_mode": source_mode,
        "to_mode": target_mode,
        "changed": changed,
        "config_path": config_path_display,
        "dry_run": dry_run,
        "next_command": next_command,
    }


def promote_run(
    *,
    run_id: str | Path,
    config_path: str | Path = "config/project.yaml",
    target_mode: str = "paper",
    dry_run: bool = False,
    acknowledge_live: str | None = None,
    apply_sweep: bool = False,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any]:
    """Promote a successful research or agent run through the canonical workflow."""

    target_mode = target_mode.strip().lower()
    if target_mode not in {"paper", "live"}:
        raise ValueError("Only promotion to paper or live is supported.")

    record = _find_run_record(run_id, runs_root=runs_root)
    summary_path = _summary_path_for_record(record, runs_root)
    summary = _load_json(summary_path)
    run_type = str(summary.get("run_type") or record.get("run_type") or "")
    project_config_path = Path(config_path)

    if apply_sweep:
        return _apply_sweep_child_run(
            record=record,
            summary=summary,
            config_path=project_config_path,
            target_mode=target_mode,
            dry_run=dry_run,
            acknowledge_live=acknowledge_live,
        )

    if run_type == "research":
        _validate_research_promotable_run(
            record,
            summary,
            target_mode=target_mode,
            acknowledge_live=acknowledge_live,
        )
        return _promote_research_run(
            record=record,
            summary=summary,
            config_path=project_config_path,
            dry_run=dry_run,
        )

    return _promote_agent_run(
        record=record,
        summary=summary,
        config_path=project_config_path,
        target_mode=target_mode,
        dry_run=dry_run,
        acknowledge_live=acknowledge_live,
    )


def promote_agent_run(
    *,
    run_id: str | Path,
    config_path: str | Path = "config/project.yaml",
    target_mode: str = "paper",
    dry_run: bool = False,
    acknowledge_live: str | None = None,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any]:
    """Backward-compatible wrapper around the generalized promotion entrypoint."""

    return promote_run(
        run_id=run_id,
        config_path=config_path,
        target_mode=target_mode,
        dry_run=dry_run,
        acknowledge_live=acknowledge_live,
        runs_root=runs_root,
    )
