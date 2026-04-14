"""Shared dispatcher for project-defined agent runs."""

from __future__ import annotations

from typing import Any

from quanttradeai.utils.project_config import load_project_config


def _load_agent_config(
    *,
    project_config_path: str,
    agent_name: str,
    project_config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_config = (
        dict(project_config_override)
        if project_config_override is not None
        else load_project_config(config_path=project_config_path).raw
    )
    agent_config = next(
        (
            dict(item)
            for item in project_config.get("agents") or []
            if item.get("name") == agent_name
        ),
        None,
    )
    if agent_config is None:
        raise ValueError(f"Agent '{agent_name}' not found in project config.")
    return agent_config


def run_project_agent(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
    mode: str = "backtest",
    skip_validation: bool = False,
    project_config_override: dict[str, Any] | None = None,
    run_timestamp: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Dispatch a project-defined agent run and return its summary plus warnings."""

    from .backtest import run_agent_backtest
    from .model_agent import (
        run_model_agent_backtest,
        run_model_agent_live,
        run_model_agent_paper,
    )
    from .paper import run_agent_live, run_agent_paper

    if project_config_override is not None and mode != "backtest":
        raise ValueError(
            "project_config_override is supported only for backtest agent runs."
        )

    agent_config = _load_agent_config(
        project_config_path=project_config_path,
        agent_name=agent_name,
        project_config_override=project_config_override,
    )
    warnings: list[str] = []

    configured_mode = str(agent_config.get("mode") or "").strip().lower()
    if mode == "live":
        if skip_validation:
            raise ValueError("--skip-validation is not supported for live agent runs.")
        if configured_mode != "live":
            raise ValueError(
                f"Agent '{agent_name}' must be configured with mode=live before running `quanttradeai agent run --mode live`."
            )
    elif configured_mode and configured_mode != mode:
        warnings.append(
            f"Warning: agent '{agent_name}' is configured with mode={configured_mode} but CLI requested mode={mode}; continuing with CLI mode."
        )

    agent_kind = agent_config.get("kind")
    if agent_kind == "model":
        if mode == "backtest":
            summary = run_model_agent_backtest(
                project_config_path=project_config_path,
                agent_name=agent_name,
                skip_validation=skip_validation,
                project_config_override=project_config_override,
                run_timestamp=run_timestamp,
            )
        elif mode == "paper":
            if skip_validation:
                warnings.append(
                    "Warning: --skip-validation is ignored for model agent paper runs."
                )
            summary = run_model_agent_paper(
                project_config_path=project_config_path,
                agent_name=agent_name,
                run_timestamp=run_timestamp,
            )
        elif mode == "live":
            summary = run_model_agent_live(
                project_config_path=project_config_path,
                agent_name=agent_name,
                run_timestamp=run_timestamp,
            )
        else:
            raise ValueError(
                "Model agents currently support only --mode backtest, --mode paper, or --mode live."
            )
    elif agent_kind in {"llm", "hybrid", "rule"}:
        if mode == "backtest":
            summary = run_agent_backtest(
                project_config_path=project_config_path,
                agent_name=agent_name,
                mode=mode,
                skip_validation=skip_validation,
                project_config_override=project_config_override,
                run_timestamp=run_timestamp,
            )
        elif mode == "paper":
            if skip_validation:
                warnings.append(
                    "Warning: --skip-validation is ignored for rule/llm/hybrid agent paper runs."
                )
            summary = run_agent_paper(
                project_config_path=project_config_path,
                agent_name=agent_name,
                run_timestamp=run_timestamp,
            )
        elif mode == "live":
            summary = run_agent_live(
                project_config_path=project_config_path,
                agent_name=agent_name,
                run_timestamp=run_timestamp,
            )
        else:
            raise ValueError(
                "Rule, LLM, and hybrid agents currently support only --mode backtest, --mode paper, or --mode live."
            )
    else:
        raise ValueError(f"Unsupported agent kind: {agent_kind}")

    return summary, warnings
