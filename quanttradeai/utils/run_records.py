"""Helpers for standardized run directories and run discovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any, Iterable


RUNS_ROOT = Path("runs")
RUN_TYPES = {"research", "agent"}
RUN_MODES = {"research", "backtest", "paper", "live"}
RUN_STATUSES = {"success", "failed"}


@dataclass(frozen=True, slots=True)
class RunFilters:
    """Filters supported by the runs list command."""

    run_type: str = "all"
    mode: str = "all"
    status: str = "all"
    limit: int = 20


def _slugify_name(name: str | None) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "run").strip())
    return normalized.strip("_").lower() or "run"


def _normalize_relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def create_run_dir(
    *,
    run_type: str,
    mode: str,
    name: str | None,
    runs_root: Path | str = RUNS_ROOT,
    timestamp: str | None = None,
) -> tuple[Path, str]:
    """Create and return the standardized run directory and run_id."""

    if run_type not in RUN_TYPES:
        raise ValueError(f"Unsupported run_type: {run_type}")

    runs_root = Path(runs_root)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = _slugify_name(name)

    if run_type == "research":
        run_dir = runs_root / "research" / f"{timestamp}_{safe_name}"
    else:
        if mode not in RUN_MODES:
            raise ValueError(f"Unsupported mode for agent run: {mode}")
        run_dir = runs_root / "agent" / mode / f"{timestamp}_{safe_name}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, _normalize_relative_path(run_dir, runs_root)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_timestamps(summary: dict[str, Any]) -> dict[str, Any]:
    timestamps = dict(summary.get("timestamps") or {})
    started_at = timestamps.get("started_at")
    completed_at = timestamps.get("completed_at")
    return {
        "started_at": started_at,
        "completed_at": completed_at,
    }


def infer_run_type(summary: dict[str, Any], run_dir: Path) -> str | None:
    explicit = summary.get("run_type")
    if explicit in RUN_TYPES:
        return str(explicit)
    if summary.get("agent_name"):
        return "agent"
    if summary.get("project_name") or "research" in run_dir.parts:
        return "research"
    return None


def infer_run_mode(summary: dict[str, Any], run_type: str) -> str | None:
    mode = summary.get("mode")
    if mode in RUN_MODES:
        return str(mode)
    if run_type == "research":
        return "research"
    return None


def infer_run_name(summary: dict[str, Any]) -> str:
    for key in ("name", "project_name", "agent_name"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "run"


def normalize_run_summary(
    summary: dict[str, Any],
    *,
    run_dir: Path,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any] | None:
    """Normalize research and agent run summaries into a shared shape."""

    if not isinstance(summary, dict) or "status" not in summary:
        return None

    runs_root = Path(runs_root)
    run_type = infer_run_type(summary, run_dir)
    if run_type is None:
        return None

    mode = infer_run_mode(summary, run_type)
    if mode is None:
        return None

    timestamps = _normalize_timestamps(summary)
    return {
        "run_id": str(
            summary.get("run_id") or _normalize_relative_path(run_dir, runs_root)
        ),
        "run_type": run_type,
        "mode": mode,
        "name": infer_run_name(summary),
        "status": str(summary.get("status")),
        "timestamps": timestamps,
        "symbols": list(summary.get("symbols") or []),
        "warnings": list(summary.get("warnings") or []),
        "artifacts": dict(summary.get("artifacts") or {}),
        "run_dir": str(summary.get("run_dir") or run_dir),
    }


def apply_required_run_fields(
    summary: dict[str, Any],
    *,
    run_dir: Path,
    run_type: str,
    mode: str,
    name: str | None = None,
    runs_root: Path | str = RUNS_ROOT,
) -> dict[str, Any]:
    """Populate the standardized required run fields on a summary payload."""

    normalized_name = name or infer_run_name(summary)
    timestamps = dict(summary.get("timestamps") or {})
    timestamps.setdefault("started_at", None)
    timestamps.setdefault("completed_at", None)
    summary["run_id"] = _normalize_relative_path(run_dir, Path(runs_root))
    summary["run_type"] = run_type
    summary["mode"] = mode
    summary["name"] = normalized_name
    summary["timestamps"] = timestamps
    summary["symbols"] = list(summary.get("symbols") or [])
    summary["warnings"] = list(summary.get("warnings") or [])
    summary["artifacts"] = dict(summary.get("artifacts") or {})
    summary["run_dir"] = str(run_dir)
    return summary


def discover_runs(runs_root: Path | str = RUNS_ROOT) -> list[dict[str, Any]]:
    """Discover normalized run summaries under the runs directory."""

    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []

    records: list[dict[str, Any]] = []
    for summary_path in runs_root.rglob("summary.json"):
        payload = _load_json(summary_path)
        if payload is None:
            continue
        run_dir = summary_path.parent
        record = normalize_run_summary(payload, run_dir=run_dir, runs_root=runs_root)
        if record is None:
            continue
        records.append(record)

    records.sort(key=_sort_key, reverse=True)
    return records


def _sort_key(record: dict[str, Any]) -> tuple[str, str]:
    timestamps = dict(record.get("timestamps") or {})
    started_at = str(timestamps.get("started_at") or "")
    return started_at, str(record.get("run_id") or "")


def filter_runs(
    records: Iterable[dict[str, Any]],
    filters: RunFilters,
) -> list[dict[str, Any]]:
    """Filter normalized run records for CLI output."""

    filtered: list[dict[str, Any]] = []
    for record in records:
        if filters.run_type != "all" and record.get("run_type") != filters.run_type:
            continue
        if filters.mode != "all" and record.get("mode") != filters.mode:
            continue
        if filters.status != "all" and record.get("status") != filters.status:
            continue
        filtered.append(record)

    return filtered[: filters.limit]
