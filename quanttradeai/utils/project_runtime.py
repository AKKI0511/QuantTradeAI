"""Compatibility helpers for project runtime config compilation."""

from __future__ import annotations

from typing import Any

from .project_config import compile_research_runtime_configs


def project_to_runtime_configs(
    project_config: dict[str, Any],
    *,
    require_research: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return canonical runtime model/features configs for project workflows."""

    model_cfg, features_cfg, _ = compile_research_runtime_configs(
        project_config,
        require_research=require_research,
    )
    return model_cfg, features_cfg
