"""Sparse run-result contracts for CLI output and run summaries."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


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
    """Attach the shared sparse result payload to a run summary."""

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
    """Build the durable result analysis stored under summary.json.run_result."""

    del project_config_path
    if summary.get("run_type") == "batch":
        run_result = _build_batch_run_result(
            summary,
            batch_results=batch_results or [],
            scoreboard_order=scoreboard_order or [],
            scoreboard_sort_by=scoreboard_sort_by
            or str(summary.get("scoreboard_sort_by") or ""),
            top_n=top_n,
        )
    else:
        run_result = {
            "schema_version": SCHEMA_VERSION,
            "metrics": _build_metrics(summary, metrics_payload=metrics_payload),
        }
    return _json_safe(run_result)


def compact_cli_result(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the no-noise JSON result printed by completion-oriented commands."""

    run_result = summary.get("run_result")
    if not isinstance(run_result, dict):
        run_result = build_run_result(summary)

    payload: dict[str, Any] = {
        "run_id": summary.get("run_id"),
        "status": summary.get("status"),
        "run_dir": summary.get("run_dir"),
        "run_type": summary.get("run_type"),
        "mode": summary.get("mode"),
        "name": summary.get("name"),
    }

    metrics = run_result.get("metrics")
    if isinstance(metrics, dict) and metrics:
        payload["metrics"] = dict(metrics)

    if summary.get("run_type") == "batch":
        batch = run_result.get("batch") if isinstance(run_result, dict) else None
        if isinstance(batch, dict):
            winner = _compact_winner(batch.get("winner"))
            batch_metrics = _compact_batch_metrics(summary)
            if batch_metrics:
                payload["metrics"] = batch_metrics
            if winner:
                payload["winner"] = winner
        for key in (
            "batch_type",
            "agent_count",
            "success_count",
            "failure_count",
        ):
            if key in summary:
                payload[key] = summary[key]
        sweep = summary.get("sweep")
        if isinstance(sweep, dict):
            payload["sweep"] = _drop_none(
                {
                    "name": sweep.get("name"),
                    "base_agent_name": sweep.get("base_agent_name"),
                }
            )
    else:
        for key in (
            "agent_kind",
            "paper_source",
            "execution_backend",
            "broker_provider",
        ):
            if key in summary:
                payload[key] = summary[key]

    if summary.get("status") != "success" and summary.get("error"):
        payload["error"] = summary.get("error")

    return _drop_none(_json_safe(payload))


def _build_batch_run_result(
    summary: dict[str, Any],
    *,
    batch_results: list[dict[str, Any]],
    scoreboard_order: list[str],
    scoreboard_sort_by: str,
    top_n: int,
) -> dict[str, Any]:
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
        for entry in ranked_successes[1 : max(top_n, 1)]
    ]
    failures = [
        _failure_entry_payload(entry)
        for entry in batch_results
        if entry.get("status") != "success"
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "batch": _drop_none(
            {
                "winner": winner,
                "top_candidates": top_candidates,
                "failures": failures,
            }
        ),
    }


def _build_metrics(
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
            "risk_status": risk_status,
        }
        return _drop_none(payload)

    return _drop_none({"metrics_status": _metrics_status(metrics or {})})


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
            "parameters": entry.get("parameters"),
            "score": score,
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
            "parameters": entry.get("parameters"),
        }
    )


def _compact_winner(winner: Any) -> dict[str, Any]:
    if not isinstance(winner, dict):
        return {}
    return _drop_none(
        {
            "agent_name": winner.get("agent_name"),
            "run_id": winner.get("run_id"),
            "score": winner.get("score"),
        }
    )


def _compact_batch_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return _drop_none({"scoreboard_sort_by": summary.get("scoreboard_sort_by")})


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
