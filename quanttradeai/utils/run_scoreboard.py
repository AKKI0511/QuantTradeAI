"""Scoreboard helpers for comparing research and agent runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCOREBOARD_SORT_FIELDS = {
    "started_at",
    "name",
    "status",
    "accuracy",
    "f1",
    "net_sharpe",
    "net_pnl",
    "total_pnl",
    "execution_count",
    "decision_count",
}

_NUMERIC_SORT_FIELDS = {
    "accuracy",
    "f1",
    "net_sharpe",
    "net_pnl",
    "total_pnl",
    "execution_count",
    "decision_count",
}

_DESCENDING_DEFAULT_SORT_FIELDS = _NUMERIC_SORT_FIELDS | {"started_at"}


def _safe_json_mapping(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"Missing JSON artifact: {path}"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON artifact {path}: {exc}"
    except OSError as exc:
        return None, f"Unable to read JSON artifact {path}: {exc}"

    if not isinstance(payload, dict):
        return None, f"JSON artifact must contain an object: {path}"
    return payload, None


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


def _mean_metric(metrics_by_symbol: Any, key: str) -> float | None:
    if not isinstance(metrics_by_symbol, dict):
        return None

    values = []
    for metrics in metrics_by_symbol.values():
        if not isinstance(metrics, dict):
            continue
        metric_value = _coerce_float(metrics.get(key))
        if metric_value is not None:
            values.append(metric_value)

    if not values:
        return None
    return sum(values) / len(values)


def _normalize_risk_status(value: Any) -> str | None:
    if isinstance(value, dict):
        status = value.get("status")
        return str(status) if status is not None else None
    if value is None:
        return None
    return str(value)


def _resolve_metrics_path(record: dict[str, Any]) -> Path:
    run_dir = Path(str(record.get("run_dir") or ""))
    artifacts = dict(record.get("artifacts") or {})
    metrics_path = artifacts.get("metrics")
    if metrics_path:
        candidate = Path(str(metrics_path))
        if candidate.is_absolute():
            return candidate
        return run_dir / candidate
    return run_dir / "metrics.json"


def _resolve_summary_path(record: dict[str, Any]) -> Path:
    return Path(str(record.get("run_dir") or "")) / "summary.json"


def _empty_scoreboard(
    *,
    metrics_path: Path,
) -> dict[str, Any]:
    return {
        "metrics_path": str(metrics_path),
        "status": "missing",
        "error": None,
        "primary_metric_name": None,
        "primary_metric": None,
        "accuracy": None,
        "f1": None,
        "net_sharpe": None,
        "net_pnl": None,
        "net_mdd": None,
        "total_pnl": None,
        "portfolio_value": None,
        "execution_count": None,
        "decision_count": None,
        "risk_status": None,
    }


def load_scoreboard_record(record: dict[str, Any]) -> dict[str, Any]:
    """Load and normalize scoreboard metrics for a run record."""

    metrics_path = _resolve_metrics_path(record)
    summary_path = _resolve_summary_path(record)
    scoreboard = _empty_scoreboard(metrics_path=metrics_path)

    metrics_payload, metrics_error = _safe_json_mapping(metrics_path)
    summary_payload, _summary_error = _safe_json_mapping(summary_path)
    summary_payload = summary_payload or {}
    run_type = str(record.get("run_type") or "")
    mode = str(record.get("mode") or "")

    if metrics_payload is None:
        scoreboard["error"] = metrics_error
        scoreboard["status"] = "invalid" if metrics_path.exists() else "missing"
        return scoreboard

    scoreboard["status"] = str(metrics_payload.get("status") or "available")

    if run_type == "research":
        research_metrics = metrics_payload.get("research_metrics_by_symbol") or {}
        backtest_metrics = metrics_payload.get("backtest_metrics_by_symbol") or {}
        scoreboard["accuracy"] = _mean_metric(research_metrics, "accuracy")
        scoreboard["f1"] = _mean_metric(research_metrics, "f1")
        scoreboard["net_sharpe"] = _mean_metric(backtest_metrics, "net_sharpe")
        scoreboard["net_pnl"] = _mean_metric(backtest_metrics, "net_pnl")
        if scoreboard["net_sharpe"] is not None:
            scoreboard["primary_metric_name"] = "net_sharpe"
            scoreboard["primary_metric"] = scoreboard["net_sharpe"]
        elif scoreboard["accuracy"] is not None:
            scoreboard["primary_metric_name"] = "accuracy"
            scoreboard["primary_metric"] = scoreboard["accuracy"]
        return scoreboard

    if run_type == "agent" and mode == "backtest":
        scoreboard["net_sharpe"] = _coerce_float(metrics_payload.get("net_sharpe"))
        scoreboard["net_pnl"] = _coerce_float(metrics_payload.get("net_pnl"))
        scoreboard["net_mdd"] = _coerce_float(metrics_payload.get("net_mdd"))
        scoreboard["decision_count"] = _coerce_int(
            summary_payload.get("decision_count")
        ) or _coerce_int(metrics_payload.get("decision_count"))
        if scoreboard["net_sharpe"] is not None:
            scoreboard["primary_metric_name"] = "net_sharpe"
            scoreboard["primary_metric"] = scoreboard["net_sharpe"]
        return scoreboard

    if run_type == "agent" and mode in {"paper", "live"}:
        scoreboard["total_pnl"] = _coerce_float(metrics_payload.get("total_pnl"))
        scoreboard["portfolio_value"] = _coerce_float(
            metrics_payload.get("portfolio_value")
        )
        scoreboard["execution_count"] = _coerce_int(
            metrics_payload.get("execution_count")
        ) or _coerce_int(summary_payload.get("execution_count"))
        scoreboard["decision_count"] = _coerce_int(
            metrics_payload.get("decision_count")
        ) or _coerce_int(summary_payload.get("decision_count"))
        scoreboard["risk_status"] = _normalize_risk_status(
            metrics_payload.get("risk_status")
        )
        if scoreboard["total_pnl"] is not None:
            scoreboard["primary_metric_name"] = "total_pnl"
            scoreboard["primary_metric"] = scoreboard["total_pnl"]
        return scoreboard

    return scoreboard


def attach_scoreboard(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return run records annotated with additive scoreboard payloads."""

    annotated: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["scoreboard"] = load_scoreboard_record(record)
        annotated.append(enriched)
    return annotated


def _sort_value(record: dict[str, Any], sort_by: str) -> Any:
    if sort_by == "started_at":
        timestamps = dict(record.get("timestamps") or {})
        return timestamps.get("started_at")
    if sort_by in {"name", "status"}:
        return record.get(sort_by)
    scoreboard = dict(record.get("scoreboard") or {})
    return scoreboard.get(sort_by)


def sort_run_records(
    records: list[dict[str, Any]],
    *,
    sort_by: str,
    ascending: bool,
) -> list[dict[str, Any]]:
    """Sort records by base or scoreboard fields, keeping missing values last."""

    with_values: list[tuple[Any, dict[str, Any]]] = []
    missing_values: list[dict[str, Any]] = []
    for record in records:
        value = _sort_value(record, sort_by)
        if value is None:
            missing_values.append(record)
            continue
        with_values.append((value, record))

    reverse = not ascending and sort_by in _DESCENDING_DEFAULT_SORT_FIELDS
    with_values.sort(key=lambda item: item[0], reverse=reverse)
    return [record for _, record in with_values] + missing_values


def _format_symbols(symbols: list[str]) -> str:
    if not symbols:
        return "-"
    if len(symbols) <= 3:
        return ",".join(symbols)
    return ",".join(symbols[:3]) + f"+{len(symbols) - 3}"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)] + "..."


def _format_float(value: float | None, *, decimals: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return str(value)


def _metric_text(record: dict[str, Any], field_name: str) -> str:
    scoreboard = dict(record.get("scoreboard") or {})

    if field_name == "SYMBOLS":
        return _format_symbols(list(record.get("symbols") or []))
    if field_name == "ACC":
        return _format_float(_coerce_float(scoreboard.get("accuracy")))
    if field_name == "F1":
        return _format_float(_coerce_float(scoreboard.get("f1")))
    if field_name == "NET_SHARPE":
        return _format_float(_coerce_float(scoreboard.get("net_sharpe")))
    if field_name == "NET_PNL":
        return _format_float(_coerce_float(scoreboard.get("net_pnl")))
    if field_name == "NET_MDD":
        return _format_float(_coerce_float(scoreboard.get("net_mdd")))
    if field_name == "TOTAL_PNL":
        return _format_float(_coerce_float(scoreboard.get("total_pnl")))
    if field_name == "PORTFOLIO":
        return _format_float(
            _coerce_float(scoreboard.get("portfolio_value")), decimals=2
        )
    if field_name == "EXEC":
        return _format_int(_coerce_int(scoreboard.get("execution_count")))
    if field_name == "DECISIONS":
        return _format_int(_coerce_int(scoreboard.get("decision_count")))
    if field_name == "RISK":
        return str(scoreboard.get("risk_status") or "-")
    if field_name == "PRIMARY":
        return _format_float(_coerce_float(scoreboard.get("primary_metric")))
    if field_name == "PNL":
        total_pnl = _coerce_float(scoreboard.get("total_pnl"))
        if total_pnl is not None:
            return _format_float(total_pnl)
        return _format_float(_coerce_float(scoreboard.get("net_pnl")))
    if field_name == "SHARPE":
        return _format_float(_coerce_float(scoreboard.get("net_sharpe")))

    if field_name == "RUN_ID":
        return str(record.get("run_id") or "-")
    if field_name == "TYPE":
        return str(record.get("run_type") or "-")
    if field_name == "MODE":
        return str(record.get("mode") or "-")
    if field_name == "STATUS":
        return str(record.get("status") or "-")
    if field_name == "NAME":
        return str(record.get("name") or "-")
    if field_name == "STARTED_AT":
        timestamps = dict(record.get("timestamps") or {})
        return str(timestamps.get("started_at") or "-")

    return "-"


def _scoreboard_columns(records: list[dict[str, Any]]) -> list[tuple[str, int]]:
    run_shapes = {(record.get("run_type"), record.get("mode")) for record in records}

    if run_shapes and all(run_type == "research" for run_type, _mode in run_shapes):
        return [
            ("RUN_ID", 40),
            ("STATUS", 8),
            ("NAME", 24),
            ("STARTED_AT", 25),
            ("SYMBOLS", 20),
            ("ACC", 8),
            ("F1", 8),
            ("NET_SHARPE", 11),
            ("NET_PNL", 10),
        ]

    if run_shapes and all(shape == ("agent", "backtest") for shape in run_shapes):
        return [
            ("RUN_ID", 40),
            ("STATUS", 8),
            ("NAME", 24),
            ("STARTED_AT", 25),
            ("SYMBOLS", 20),
            ("NET_SHARPE", 11),
            ("NET_PNL", 10),
            ("NET_MDD", 10),
            ("DECISIONS", 10),
        ]

    if run_shapes and all(
        run_type == "agent" and mode in {"paper", "live"}
        for run_type, mode in run_shapes
    ):
        return [
            ("RUN_ID", 40),
            ("MODE", 10),
            ("STATUS", 8),
            ("NAME", 24),
            ("STARTED_AT", 25),
            ("SYMBOLS", 20),
            ("TOTAL_PNL", 10),
            ("PORTFOLIO", 12),
            ("EXEC", 6),
            ("DECISIONS", 10),
            ("RISK", 10),
        ]

    return [
        ("RUN_ID", 40),
        ("TYPE", 8),
        ("MODE", 10),
        ("STATUS", 8),
        ("NAME", 24),
        ("STARTED_AT", 25),
        ("SYMBOLS", 20),
        ("PRIMARY", 9),
        ("PNL", 10),
        ("SHARPE", 10),
        ("EXEC", 6),
        ("DECISIONS", 10),
        ("RISK", 10),
    ]


def render_scoreboard_table(records: list[dict[str, Any]]) -> str:
    """Render an adaptive metrics-aware scoreboard table."""

    columns = _scoreboard_columns(records)
    header = "  ".join(f"{label:<{width}}" for label, width in columns)
    lines = [header]
    for record in records:
        lines.append(
            "  ".join(
                f"{_truncate(_metric_text(record, label), width):<{width}}"
                for label, width in columns
            )
        )
    return "\n".join(lines)
