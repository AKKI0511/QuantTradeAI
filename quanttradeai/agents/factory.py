"""Shared strategy factory for project-defined agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseStrategy
from .llm import LLMAgentStrategy
from .rule import RuleAgentStrategy


def build_strategy(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
) -> BaseStrategy:
    """Return the concrete strategy implementation for the given agent kind."""

    agent_kind = str(agent_config.get("kind") or "").strip().lower()
    if agent_kind in {"llm", "hybrid"}:
        return LLMAgentStrategy(
            project_config_path=project_config_path,
            llm_config=dict(agent_config.get("llm") or {}),
        )
    if agent_kind == "rule":
        return RuleAgentStrategy(agent_config=agent_config)

    raise ValueError(f"Unsupported agent kind for strategy factory: {agent_kind}")
