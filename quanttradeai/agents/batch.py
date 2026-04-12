"""Batch orchestration for multi-agent backtest runs."""

from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.project_config import load_project_config
from quanttradeai.utils.run_scoreboard import (
    attach_scoreboard,
    render_scoreboard_table,
    sort_run_records,
)

from .runner import run_project_agent


def _slugify(value: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "run").strip())
    return normalized.strip("_").lower() or "run"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_artifact(source: str | Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return str(destination)


def _stdout_log_path(batch_dir: Path, agent_name: str) -> Path:
    path = batch_dir / "logs" / f"{_slugify(agent_name)}.stdout.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stderr_log_path(batch_dir: Path, agent_name: str) -> Path:
    path = batch_dir / "logs" / f"{_slugify(agent_name)}.stderr.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _predict_child_run_dir(*, agent_name: str, run_timestamp: str) -> Path:
    return (
        Path("runs") / "agent" / "backtest" / f"{run_timestamp}_{_slugify(agent_name)}"
    )


def _child_run_timestamp(batch_timestamp: str, index: int) -> str:
    return f"{batch_timestamp}_{index:02d}"


def run_agent_backtest_batch(
    *,
    project_config_path: str = "config/project.yaml",
    skip_validation: bool = False,
    max_concurrency: int = 1,
) -> dict[str, Any]:
    """Run every configured agent through the normal backtest path."""

    if max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    loaded_project = load_project_config(config_path=project_config_path)
    project_name = (loaded_project.raw.get("project") or {}).get("name") or Path(
        project_config_path
    ).stem
    batch_dir = (
        Path("runs")
        / "agent"
        / "batches"
        / f"{timestamp}_{_slugify(project_name)}_backtest"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_project_config(
        config_path=project_config_path,
        output_dir=batch_dir / "validation",
        timestamp_subdir=False,
    )
    resolved_validation_path = Path(validation["artifacts"]["resolved_config"])
    resolved_project_path = batch_dir / "resolved_project_config.yaml"
    _copy_artifact(resolved_validation_path, resolved_project_path)
    resolved_project = (
        yaml.safe_load(resolved_project_path.read_text(encoding="utf-8")) or {}
    )

    agents = sorted(
        [dict(agent) for agent in resolved_project.get("agents") or []],
        key=lambda agent: str(agent.get("name") or ""),
    )
    if not agents:
        raise ValueError("Project config defines no agents to run with --all.")

    agent_specs = [
        {
            "agent_name": str(agent.get("name") or ""),
            "agent_kind": str(agent.get("kind") or ""),
            "configured_mode": str(agent.get("mode") or ""),
            "run_timestamp": _child_run_timestamp(timestamp, index),
        }
        for index, agent in enumerate(agents, start=1)
    ]

    def _run_one(spec: dict[str, Any]) -> dict[str, Any]:
        agent_name = spec["agent_name"]
        stdout_log = _stdout_log_path(batch_dir, agent_name)
        stderr_log = _stderr_log_path(batch_dir, agent_name)

        try:
            summary, warnings = run_project_agent(
                project_config_path=str(resolved_project_path),
                agent_name=agent_name,
                mode="backtest",
                skip_validation=skip_validation,
                run_timestamp=str(spec["run_timestamp"]),
            )
            stdout_log.write_text(
                json.dumps({"summary": summary, "warnings": warnings}, indent=2),
                encoding="utf-8",
            )
            stderr_log.write_text("", encoding="utf-8")
            return {
                **spec,
                "status": "success",
                "warnings": warnings,
                "summary": summary,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        except Exception as exc:
            stderr_log.write_text(traceback.format_exc(), encoding="utf-8")
            stdout_log.write_text("", encoding="utf-8")
            child_run_dir = _predict_child_run_dir(
                agent_name=agent_name,
                run_timestamp=str(spec["run_timestamp"]),
            )
            child_summary = _load_summary(child_run_dir / "summary.json") or {
                "run_id": f"agent/backtest/{child_run_dir.name}",
                "run_type": "agent",
                "mode": "backtest",
                "name": agent_name,
                "status": "failed",
                "timestamps": {},
                "symbols": [],
                "warnings": [],
                "artifacts": {},
                "run_dir": str(child_run_dir),
                "error": str(exc),
            }
            return {
                **spec,
                "status": "failed",
                "error": str(exc),
                "warnings": list(child_summary.get("warnings") or []),
                "summary": child_summary,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [executor.submit(_run_one, spec) for spec in agent_specs]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["agent_name"])

    scoreboard_records = attach_scoreboard([item["summary"] for item in results])
    scoreboard_records = sort_run_records(
        scoreboard_records,
        sort_by="net_sharpe",
        ascending=False,
    )

    results_payload = {
        "results": [
            {
                "agent_name": item["agent_name"],
                "agent_kind": item["agent_kind"],
                "configured_mode": item["configured_mode"],
                "status": item["status"],
                "warnings": item["warnings"],
                "error": item.get("error"),
                "run_timestamp": item["run_timestamp"],
                "run_id": item["summary"].get("run_id"),
                "run_dir": item["summary"].get("run_dir"),
                "stdout_log": item["stdout_log"],
                "stderr_log": item["stderr_log"],
                "scoreboard": next(
                    (
                        record.get("scoreboard")
                        for record in scoreboard_records
                        if record.get("run_id") == item["summary"].get("run_id")
                    ),
                    None,
                ),
            }
            for item in results
        ]
    }
    scoreboard_payload = {
        "sort_by": "net_sharpe",
        "records": scoreboard_records,
    }

    results_path = batch_dir / "results.json"
    scoreboard_json_path = batch_dir / "scoreboard.json"
    scoreboard_txt_path = batch_dir / "scoreboard.txt"
    _write_json(results_path, results_payload)
    _write_json(scoreboard_json_path, scoreboard_payload)
    scoreboard_txt_path.write_text(
        render_scoreboard_table(scoreboard_records),
        encoding="utf-8",
    )

    success_count = sum(1 for item in results if item["status"] == "success")
    failure_count = len(results) - success_count
    batch_status = "success" if failure_count == 0 else "failed"

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": batch_status,
        "project_name": project_name,
        "project_config_path": str(project_config_path),
        "resolved_project_config": str(resolved_project_path),
        "mode": "backtest",
        "max_concurrency": max_concurrency,
        "run_dir": str(batch_dir),
        "agent_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "artifacts": {
            "results": str(results_path),
            "scoreboard_json": str(scoreboard_json_path),
            "scoreboard_txt": str(scoreboard_txt_path),
        },
        "agents": [
            {
                "agent_name": item["agent_name"],
                "agent_kind": item["agent_kind"],
                "configured_mode": item["configured_mode"],
                "status": item["status"],
                "run_timestamp": item["run_timestamp"],
                "run_id": item["summary"].get("run_id"),
                "run_dir": item["summary"].get("run_dir"),
                "stdout_log": item["stdout_log"],
                "stderr_log": item["stderr_log"],
            }
            for item in results
        ],
        "warnings": list(dict.fromkeys(validation.get("warnings", []))),
    }
    manifest_path = batch_dir / "batch_manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "status": batch_status,
        "project_name": project_name,
        "mode": "backtest",
        "run_dir": str(batch_dir),
        "agent_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "artifacts": {
            **manifest["artifacts"],
            "manifest": str(manifest_path),
        },
        "results": results_payload["results"],
        "warnings": manifest["warnings"],
    }
