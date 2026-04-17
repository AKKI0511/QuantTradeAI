"""Artifact-backed run comparison helpers for canonical workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from quanttradeai.utils.project_config import resolve_paper_replay_window
from quanttradeai.utils.run_records import RUNS_ROOT, discover_runs
from quanttradeai.utils.run_scoreboard import (
    attach_scoreboard,
    render_scoreboard_table,
    sort_run_records,
)


COMPARE_RUN_LIMIT_MIN = 2
COMPARE_RUN_LIMIT_MAX = 4

RUN_FAMILY_METRICS = {
    "research": ["accuracy", "f1", "net_sharpe", "net_pnl"],
    "agent/backtest": ["net_sharpe", "net_pnl", "net_mdd", "decision_count"],
    "agent/paper": [
        "total_pnl",
        "portfolio_value",
        "execution_count",
        "decision_count",
        "risk_status",
    ],
    "agent/live": [
        "total_pnl",
        "portfolio_value",
        "execution_count",
        "decision_count",
        "risk_status",
    ],
}


def _normalize_run_id(value: str | Path) -> str:
    normalized = str(value).replace("\\", "/").strip().strip("/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if "runs" in parts:
        parts = parts[parts.index("runs") + 1 :]
    return "/".join(parts)


def _run_family(record: dict[str, Any]) -> str:
    run_type = str(record.get("run_type") or "").strip()
    mode = str(record.get("mode") or "").strip()
    if run_type == "research":
        return "research"
    return f"{run_type}/{mode}"


def _resolve_run_artifact_path(
    record: dict[str, Any],
    *,
    artifact_key: str,
    default_filename: str,
) -> Path:
    run_dir = Path(str(record.get("run_dir") or ""))
    artifacts = dict(record.get("artifacts") or {})
    artifact_path = artifacts.get(artifact_key)
    if artifact_path:
        candidate = Path(str(artifact_path))
        if candidate.is_absolute():
            return candidate
        if run_dir.parts and candidate.parts[: len(run_dir.parts)] == run_dir.parts:
            return candidate
        return run_dir / candidate
    return run_dir / default_filename


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Run is missing {label}: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run {label} is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Run {label} must contain a JSON object: {path}")
    return payload


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Run is missing {label}: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run {label} must contain a YAML mapping: {path}")
    return payload


def _ensure_compare_candidates(run_ids: list[str]) -> None:
    if len(run_ids) < COMPARE_RUN_LIMIT_MIN:
        raise ValueError(
            "Compare mode requires at least two explicit --compare values."
        )
    if len(run_ids) > COMPARE_RUN_LIMIT_MAX:
        raise ValueError(
            "Compare mode supports at most four explicit --compare values."
        )

    normalized = [_normalize_run_id(run_id) for run_id in run_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Compare mode does not allow duplicate run ids.")


def _resolve_compare_records(
    *,
    run_ids: list[str],
    runs_root: Path | str = RUNS_ROOT,
) -> list[dict[str, Any]]:
    discovered = discover_runs(runs_root)
    records: list[dict[str, Any]] = []
    for run_id in run_ids:
        expected = _normalize_run_id(run_id)
        matched = None
        for record in discovered:
            candidates = {
                _normalize_run_id(str(record.get("run_id") or "")),
                _normalize_run_id(str(record.get("run_dir") or "")),
            }
            if expected in candidates:
                matched = dict(record)
                break
        if matched is None:
            raise ValueError(f"Run not found for compare: {run_id}")
        records.append(matched)
    return records


def _normalize_context(context_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "features" in context_cfg:
        normalized["features"] = list(context_cfg.get("features") or [])
    if "model_signals" in context_cfg:
        normalized["model_signals"] = list(context_cfg.get("model_signals") or [])

    for key in (
        "market_data",
        "positions",
        "orders",
        "risk_state",
        "news",
        "memory",
        "notes",
    ):
        if key not in context_cfg:
            continue
        value = context_cfg.get(key)
        normalized[key] = dict(value) if isinstance(value, dict) else value

    return normalized


def _extract_research_view(project_config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(project_config.get("data") or {})
    research_cfg = dict(project_config.get("research") or {})
    feature_definitions = list(
        (project_config.get("features") or {}).get("definitions") or []
    )

    return {
        "project": {
            "name": (project_config.get("project") or {}).get("name"),
            "profile": (project_config.get("project") or {}).get("profile"),
        },
        "data": {
            "symbols": list(data_cfg.get("symbols") or []),
            "timeframe": data_cfg.get("timeframe"),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "test_start": data_cfg.get("test_start"),
            "test_end": data_cfg.get("test_end"),
        },
        "features": {
            "definitions": [
                item.get("name") for item in feature_definitions if item.get("name")
            ]
        },
        "labels": dict(research_cfg.get("labels") or {}),
        "model": {
            "kind": (research_cfg.get("model") or {}).get("kind"),
            "family": (research_cfg.get("model") or {}).get("family"),
            "tuning": dict(((research_cfg.get("model") or {}).get("tuning") or {})),
        },
        "evaluation": dict(research_cfg.get("evaluation") or {}),
        "backtest_costs": dict(
            ((research_cfg.get("backtest") or {}).get("costs") or {})
        ),
    }


def _find_agent_config(
    *,
    project_config: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    for agent in project_config.get("agents") or []:
        if isinstance(agent, dict) and agent.get("name") == agent_name:
            return dict(agent)
    raise ValueError(
        f"Resolved project config does not contain compared agent '{agent_name}'."
    )


def _extract_agent_view(
    project_config: dict[str, Any],
    *,
    family: str,
    agent_name: str,
) -> dict[str, Any]:
    data_cfg = dict(project_config.get("data") or {})
    streaming_cfg = dict(data_cfg.get("streaming") or {})
    agent_cfg = _find_agent_config(project_config=project_config, agent_name=agent_name)
    context_cfg = dict(agent_cfg.get("context") or {})

    view: dict[str, Any] = {
        "project": {
            "name": (project_config.get("project") or {}).get("name"),
            "profile": (project_config.get("project") or {}).get("profile"),
        },
        "data": {
            "symbols": list(data_cfg.get("symbols") or []),
            "timeframe": data_cfg.get("timeframe"),
            "start_date": data_cfg.get("start_date"),
            "end_date": data_cfg.get("end_date"),
            "test_start": data_cfg.get("test_start"),
            "test_end": data_cfg.get("test_end"),
        },
        "agent": {
            "name": agent_cfg.get("name"),
            "kind": agent_cfg.get("kind"),
            "mode": agent_cfg.get("mode"),
            "tools": list(agent_cfg.get("tools") or []),
            "context": _normalize_context(context_cfg),
            "risk": dict(agent_cfg.get("risk") or {}),
        },
    }

    if agent_cfg.get("kind") == "rule":
        view["agent"]["rule"] = dict(agent_cfg.get("rule") or {})
    if agent_cfg.get("kind") == "model":
        view["agent"]["model"] = dict(agent_cfg.get("model") or {})
    if agent_cfg.get("kind") in {"llm", "hybrid"}:
        llm_cfg = dict(agent_cfg.get("llm") or {})
        view["agent"]["llm"] = {
            "provider": llm_cfg.get("provider"),
            "model": llm_cfg.get("model"),
            "prompt_file": llm_cfg.get("prompt_file"),
        }
    if agent_cfg.get("kind") == "hybrid":
        view["agent"]["model_signal_sources"] = list(
            agent_cfg.get("model_signal_sources") or []
        )

    if family == "agent/paper":
        replay_window = resolve_paper_replay_window(project_config)
        view["paper"] = {
            "source": "replay" if replay_window is not None else "realtime",
            "streaming": {
                "provider": streaming_cfg.get("provider"),
                "symbols": list(streaming_cfg.get("symbols") or []),
                "channels": list(streaming_cfg.get("channels") or []),
                "auth_method": streaming_cfg.get("auth_method"),
            },
            "replay_window": (
                {
                    "start_date": replay_window.start_date,
                    "end_date": replay_window.end_date,
                    "pace_delay_ms": replay_window.pace_delay_ms,
                }
                if replay_window is not None
                else None
            ),
        }

    if family == "agent/live":
        view["live"] = {
            "streaming": {
                "provider": streaming_cfg.get("provider"),
                "symbols": list(streaming_cfg.get("symbols") or []),
                "channels": list(streaming_cfg.get("channels") or []),
                "auth_method": streaming_cfg.get("auth_method"),
            },
            "risk": dict(project_config.get("risk") or {}),
            "position_manager": dict(project_config.get("position_manager") or {}),
        }

    return view


def _extract_config_view(
    *,
    family: str,
    record: dict[str, Any],
    summary: dict[str, Any],
    project_config: dict[str, Any],
) -> dict[str, Any]:
    if family == "research":
        return _extract_research_view(project_config)

    agent_name = str(
        summary.get("agent_name") or summary.get("name") or record.get("name") or ""
    ).strip()
    if not agent_name:
        raise ValueError("Compared agent run is missing an agent name.")
    return _extract_agent_view(project_config, family=family, agent_name=agent_name)


def _flatten_view(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value and prefix:
            flattened[prefix] = {}
            return flattened
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_view(value[key], prefix=next_prefix))
        return flattened

    flattened[prefix] = value
    return flattened


def _diff_config_views(run_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    flattened_views = {
        entry["run_id"]: _flatten_view(entry["config_view"]) for entry in run_entries
    }
    fields = sorted(
        {field for flattened in flattened_views.values() for field in flattened.keys()}
    )

    differences: dict[str, dict[str, Any]] = {}
    for field in fields:
        values_by_run = {
            entry["run_id"]: flattened_views[entry["run_id"]].get(field)
            for entry in run_entries
        }
        unique_values = list(values_by_run.values())
        if unique_values[1:] and any(
            value != unique_values[0] for value in unique_values[1:]
        ):
            differences[field] = values_by_run
    return differences


def build_run_comparison(
    *,
    run_ids: list[str],
    sort_by: str = "started_at",
    ascending: bool = False,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any]:
    """Return a stable comparison payload for explicit run ids."""

    _ensure_compare_candidates(run_ids)
    records = _resolve_compare_records(run_ids=run_ids, runs_root=runs_root)
    enriched_records = attach_scoreboard(records)

    run_families = {_run_family(record) for record in enriched_records}
    if len(run_families) != 1:
        raise ValueError(
            "Compare mode requires runs from the same family only. Use `quanttradeai runs list --scoreboard` to rank mixed runs first."
        )
    run_family = run_families.pop()
    if run_family not in RUN_FAMILY_METRICS:
        raise ValueError(f"Compare mode does not support run family: {run_family}")

    enriched_records = sort_run_records(
        enriched_records,
        sort_by=sort_by,
        ascending=ascending,
    )

    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for record in enriched_records:
        summary_path = _resolve_run_artifact_path(
            record,
            artifact_key="summary",
            default_filename="summary.json",
        )
        metrics_path = _resolve_run_artifact_path(
            record,
            artifact_key="metrics",
            default_filename="metrics.json",
        )
        config_path = _resolve_run_artifact_path(
            record,
            artifact_key="resolved_project_config",
            default_filename="resolved_project_config.yaml",
        )

        summary_payload = _load_json_mapping(summary_path, label="summary.json")
        _load_json_mapping(metrics_path, label="metrics.json")
        project_config = _load_yaml_mapping(
            config_path,
            label="resolved_project_config.yaml",
        )
        scoreboard = dict(record.get("scoreboard") or {})
        if scoreboard.get("error"):
            warnings.append(f"{record['run_id']}: {scoreboard['error']}")

        run_entry = {
            "run_id": record.get("run_id"),
            "run_type": record.get("run_type"),
            "mode": record.get("mode"),
            "name": record.get("name"),
            "status": record.get("status"),
            "symbols": list(record.get("symbols") or []),
            "timestamps": dict(record.get("timestamps") or {}),
            "artifacts": {
                "summary": str(summary_path),
                "metrics": str(metrics_path),
                "resolved_project_config": str(config_path),
            },
            "warnings": list(record.get("warnings") or []),
            "scoreboard": scoreboard,
            "config_view": _extract_config_view(
                family=run_family,
                record=record,
                summary=summary_payload,
                project_config=project_config,
            ),
        }
        warnings.extend(run_entry["warnings"])
        runs.append(run_entry)

    metric_columns = list(RUN_FAMILY_METRICS[run_family])
    rows = [
        {
            "run_id": run["run_id"],
            "name": run["name"],
            "status": run["status"],
            "started_at": (run.get("timestamps") or {}).get("started_at"),
            "symbols": list(run.get("symbols") or []),
            "metrics": {
                metric_name: (run.get("scoreboard") or {}).get(metric_name)
                for metric_name in metric_columns
            },
        }
        for run in runs
    ]

    return {
        "kind": "run_comparison",
        "run_family": run_family,
        "runs": runs,
        "metric_columns": metric_columns,
        "rows": rows,
        "config_differences": _diff_config_views(runs),
        "warnings": list(dict.fromkeys(warnings)),
    }


def render_run_comparison(comparison: dict[str, Any]) -> str:
    """Render a human-readable run comparison."""

    runs = list(comparison.get("runs") or [])
    lines = [
        f"Run comparison: {comparison.get('run_family')}",
        "",
        "Metrics:",
        render_scoreboard_table(runs),
        "",
        "Config differences:",
    ]

    config_differences = dict(comparison.get("config_differences") or {})
    if not config_differences:
        lines.append("- No config differences detected.")
    else:
        for field, values in config_differences.items():
            lines.append(f"- {field}")
            for run in runs:
                run_id = str(run.get("run_id"))
                lines.append(f"  {run_id}: {_render_value(values.get(run_id))}")

    lines.extend(["", "Artifacts:"])
    for run in runs:
        lines.append(f"- {run.get('run_id')}")
        for artifact_name, artifact_path in dict(run.get("artifacts") or {}).items():
            lines.append(f"  {artifact_name}: {artifact_path}")

    warnings = list(comparison.get("warnings") or [])
    if warnings:
        lines.extend(["", "Warnings:"])
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def _render_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, sort_keys=True)
    return str(value)
