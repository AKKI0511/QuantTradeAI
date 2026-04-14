"""Batch orchestration for multi-agent and sweep backtest runs."""

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
from quanttradeai.utils.sweeps import expand_agent_backtest_sweep, sweep_summary_payload

from .runner import run_project_agent


def _slugify(value: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "run").strip())
    return normalized.strip("_").lower() or "run"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def _write_child_summary(summary: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(str(summary.get("run_dir") or ""))
    if run_dir:
        _write_json(run_dir / "summary.json", summary)
    return summary


def _attach_sweep_metadata(
    summary: dict[str, Any],
    *,
    sweep: dict[str, Any] | None,
) -> dict[str, Any]:
    if not sweep:
        return summary
    patched = dict(summary)
    patched["sweep"] = dict(sweep)
    return _write_child_summary(patched)


def _default_failed_summary(
    *,
    agent_name: str,
    child_run_dir: Path,
    error: str,
) -> dict[str, Any]:
    return {
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
        "error": error,
    }


def _build_all_agent_specs(
    *,
    resolved_project: dict[str, Any],
    batch_timestamp: str,
    original_project_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    agents = sorted(
        [dict(agent) for agent in resolved_project.get("agents") or []],
        key=lambda agent: str(agent.get("name") or ""),
    )
    if not agents:
        raise ValueError("Project config defines no agents to run with --all.")

    specs = [
        {
            "agent_name": str(agent.get("name") or ""),
            "agent_kind": str(agent.get("kind") or ""),
            "configured_mode": str(agent.get("mode") or ""),
            "run_timestamp": _child_run_timestamp(batch_timestamp, index),
            "display_order": index,
            "project_config_path": str(original_project_path),
            "project_config_override": None,
            "variant_project_config": None,
            "base_agent_name": None,
            "parameters": None,
            "sweep": None,
        }
        for index, agent in enumerate(agents, start=1)
    ]
    return None, specs


def _build_sweep_specs(
    *,
    resolved_project: dict[str, Any],
    batch_timestamp: str,
    batch_dir: Path,
    original_project_path: Path,
    sweep_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expansion = expand_agent_backtest_sweep(resolved_project, sweep_name)
    variants = list(expansion["variants"])
    if not variants:
        raise ValueError(f"Sweep '{sweep_name}' expanded to zero variants.")

    specs: list[dict[str, Any]] = []
    expanded_variants: list[dict[str, Any]] = []
    variants_root = batch_dir / "variants"
    for index, variant in enumerate(variants, start=1):
        variant_dir = variants_root / f"{index:03d}_{_slugify(variant['name'])}"
        variant_config_path = variant_dir / "project.yaml"
        _write_yaml(variant_config_path, dict(variant["project_config"]))

        sweep_metadata = sweep_summary_payload(
            sweep_name=expansion["name"],
            base_agent_name=expansion["base_agent_name"],
            parameters=dict(variant["parameters"]),
        )
        specs.append(
            {
                "agent_name": str(variant["name"]),
                "agent_kind": str(
                    (variant.get("agent_config") or {}).get("kind") or ""
                ),
                "configured_mode": str(
                    (variant.get("agent_config") or {}).get("mode") or ""
                ),
                "run_timestamp": _child_run_timestamp(batch_timestamp, index),
                "display_order": index,
                "project_config_path": str(original_project_path),
                "project_config_override": dict(variant["project_config"]),
                "variant_project_config": str(variant_config_path),
                "base_agent_name": expansion["base_agent_name"],
                "parameters": dict(variant["parameters"]),
                "sweep": sweep_metadata,
            }
        )
        expanded_variants.append(
            {
                "name": str(variant["name"]),
                "base_agent_name": expansion["base_agent_name"],
                "parameters": dict(variant["parameters"]),
                "project_config_path": str(variant_config_path),
            }
        )

    sweep_payload = {
        "name": expansion["name"],
        "kind": expansion["kind"],
        "base_agent_name": expansion["base_agent_name"],
        "parameters": list(expansion["parameters"]),
        "variant_count": len(expanded_variants),
        "expanded_variants": expanded_variants,
    }
    return sweep_payload, specs


def run_agent_backtest_batch(
    *,
    project_config_path: str = "config/project.yaml",
    skip_validation: bool = False,
    max_concurrency: int = 1,
    sweep_name: str | None = None,
) -> dict[str, Any]:
    """Run every configured agent or sweep variant through the backtest path."""

    if max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    original_project_path = Path(project_config_path).resolve()
    loaded_project = load_project_config(config_path=original_project_path)
    project_name = (loaded_project.raw.get("project") or {}).get("name") or Path(
        original_project_path
    ).stem
    batch_name = (
        f"{timestamp}_{_slugify(project_name)}_{_slugify(sweep_name)}_backtest"
        if sweep_name
        else f"{timestamp}_{_slugify(project_name)}_backtest"
    )
    batch_dir = Path("runs") / "agent" / "batches" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_project_config(
        config_path=original_project_path,
        output_dir=batch_dir / "validation",
        timestamp_subdir=False,
    )
    resolved_validation_path = Path(validation["artifacts"]["resolved_config"])
    resolved_project_path = batch_dir / "resolved_project_config.yaml"
    _copy_artifact(resolved_validation_path, resolved_project_path)
    resolved_project = (
        yaml.safe_load(resolved_project_path.read_text(encoding="utf-8")) or {}
    )

    if sweep_name:
        sweep_payload, agent_specs = _build_sweep_specs(
            resolved_project=resolved_project,
            batch_timestamp=timestamp,
            batch_dir=batch_dir,
            original_project_path=original_project_path,
            sweep_name=sweep_name,
        )
        batch_type = "sweep"
    else:
        sweep_payload, agent_specs = _build_all_agent_specs(
            resolved_project=resolved_project,
            batch_timestamp=timestamp,
            original_project_path=original_project_path,
        )
        batch_type = "all_agents"

    def _run_one(spec: dict[str, Any]) -> dict[str, Any]:
        agent_name = str(spec["agent_name"])
        stdout_log = _stdout_log_path(batch_dir, agent_name)
        stderr_log = _stderr_log_path(batch_dir, agent_name)

        try:
            summary, warnings = run_project_agent(
                project_config_path=str(spec["project_config_path"]),
                agent_name=agent_name,
                mode="backtest",
                skip_validation=skip_validation,
                project_config_override=spec.get("project_config_override"),
                run_timestamp=str(spec["run_timestamp"]),
            )
            summary = _attach_sweep_metadata(summary, sweep=spec.get("sweep"))
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
            child_summary = _load_summary(
                child_run_dir / "summary.json"
            ) or _default_failed_summary(
                agent_name=agent_name,
                child_run_dir=child_run_dir,
                error=str(exc),
            )
            child_summary = _attach_sweep_metadata(
                child_summary,
                sweep=spec.get("sweep"),
            )
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

    results.sort(key=lambda item: int(item["display_order"]))

    scoreboard_records = attach_scoreboard([item["summary"] for item in results])
    scoreboard_records = sort_run_records(
        scoreboard_records,
        sort_by="net_sharpe",
        ascending=False,
    )
    scoreboard_order = [
        str(record.get("run_id") or "") for record in scoreboard_records
    ]

    results_payload = {
        "batch_type": batch_type,
        "mode": "backtest",
        "scoreboard_order": scoreboard_order,
        "results": [
            {
                "agent_name": item["agent_name"],
                "base_agent_name": item.get("base_agent_name"),
                "agent_kind": item["agent_kind"],
                "configured_mode": item["configured_mode"],
                "status": item["status"],
                "warnings": item["warnings"],
                "error": item.get("error"),
                "parameters": item.get("parameters"),
                "run_timestamp": item["run_timestamp"],
                "run_id": item["summary"].get("run_id"),
                "run_dir": item["summary"].get("run_dir"),
                "variant_project_config": item.get("variant_project_config"),
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
        ],
    }
    if sweep_payload is not None:
        results_payload["sweep"] = dict(sweep_payload)
        results_payload["expanded_variants"] = list(
            sweep_payload.get("expanded_variants") or []
        )

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
        "batch_type": batch_type,
        "project_name": project_name,
        "project_config_path": str(original_project_path),
        "resolved_project_config": str(resolved_project_path),
        "mode": "backtest",
        "max_concurrency": max_concurrency,
        "run_dir": str(batch_dir),
        "agent_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "scoreboard_order": scoreboard_order,
        "artifacts": {
            "results": str(results_path),
            "scoreboard_json": str(scoreboard_json_path),
            "scoreboard_txt": str(scoreboard_txt_path),
        },
        "agents": [
            {
                "agent_name": item["agent_name"],
                "base_agent_name": item.get("base_agent_name"),
                "parameters": item.get("parameters"),
                "agent_kind": item["agent_kind"],
                "configured_mode": item["configured_mode"],
                "status": item["status"],
                "run_timestamp": item["run_timestamp"],
                "run_id": item["summary"].get("run_id"),
                "run_dir": item["summary"].get("run_dir"),
                "variant_project_config": item.get("variant_project_config"),
                "stdout_log": item["stdout_log"],
                "stderr_log": item["stderr_log"],
            }
            for item in results
        ],
        "warnings": list(dict.fromkeys(validation.get("warnings", []))),
    }
    if sweep_payload is not None:
        manifest["sweep"] = dict(sweep_payload)
        manifest["expanded_variants"] = list(
            sweep_payload.get("expanded_variants") or []
        )

    manifest_path = batch_dir / "batch_manifest.json"
    _write_json(manifest_path, manifest)

    return {
        "status": batch_status,
        "batch_type": batch_type,
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
        **({"sweep": dict(sweep_payload)} if sweep_payload is not None else {}),
    }
