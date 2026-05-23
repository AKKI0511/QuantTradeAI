"""Helpers for canonical project-config parameter sweeps."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import re
from typing import Any


DISALLOWED_SWEEP_PATHS = {"name", "kind", "mode"}


def is_scalar_sweep_value(value: Any) -> bool:
    """Return whether the value is a supported scalar sweep leaf."""

    return value is None or isinstance(value, (str, int, float, bool))


def _slugify_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return normalized.strip("_").lower() or "value"


def slug_scalar_sweep_value(value: Any) -> str:
    """Return a deterministic scalar slug for variant naming."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return _slugify_token(str(value))


def _path_tokens(path: str) -> list[str]:
    normalized = str(path or "").strip()
    if not normalized:
        raise ValueError("Sweep parameter path must not be blank.")
    tokens = [token.strip() for token in normalized.split(".")]
    if any(not token for token in tokens):
        raise ValueError(
            f"Sweep parameter path '{path}' must use dot-separated non-empty tokens."
        )
    if tokens[0] in DISALLOWED_SWEEP_PATHS:
        raise ValueError(
            f"Sweep parameter path '{path}' is not allowed. "
            "Sweeps may not modify agent identity or execution-mode fields."
        )
    return tokens


def _is_existing_scalar_leaf(value: Any) -> bool:
    return is_scalar_sweep_value(value)


def resolve_agent_scalar_path(
    agent_config: dict[str, Any],
    path: str,
) -> tuple[dict[str, Any], str, Any]:
    """Resolve a sweep path against an agent config and ensure it is scalar."""

    tokens = _path_tokens(path)
    current: Any = agent_config
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise ValueError(
                f"Sweep parameter path '{path}' must resolve to an existing scalar leaf."
            )
        current = current[token]
    if not isinstance(current, dict) or tokens[-1] not in current:
        raise ValueError(
            f"Sweep parameter path '{path}' must resolve to an existing scalar leaf."
        )

    leaf_key = tokens[-1]
    leaf_value = current[leaf_key]
    if not is_scalar_sweep_value(leaf_value):
        raise ValueError(
            f"Sweep parameter path '{path}' must resolve to an existing scalar leaf."
        )
    return current, leaf_key, leaf_value


RESEARCH_SWEEP_DATA_LEAVES = {
    "timeframe",
    "start_date",
    "end_date",
    "test_start",
    "test_end",
    "cache_path",
    "cache_dir",
    "cache_expiration_days",
    "use_cache",
    "refresh",
    "max_workers",
}

RESEARCH_SWEEP_RESEARCH_PATHS = {
    ("research", "model", "tuning", "enabled"),
    ("research", "model", "tuning", "trials"),
    ("research", "evaluation", "use_configured_test_window"),
}

RESEARCH_SWEEP_RESEARCH_PREFIXES = {
    ("research", "labels"),
    ("research", "backtest", "costs"),
}


def _feature_definition_by_name(
    project_config: dict[str, Any],
    feature_name: str,
) -> dict[str, Any] | None:
    for definition in (project_config.get("features") or {}).get("definitions") or []:
        if isinstance(definition, dict) and definition.get("name") == feature_name:
            return definition
    return None


def _feature_path_parent(
    project_config: dict[str, Any],
    tokens: list[str],
    path: str,
) -> tuple[dict[str, Any], str, Any]:
    if len(tokens) != 4 or tokens[0] != "features" or tokens[2] != "params":
        raise ValueError(
            f"Research sweep parameter path '{path}' must use features.<feature_name>.params.<param>."
        )

    feature_name = tokens[1]
    feature_definition = _feature_definition_by_name(project_config, feature_name)
    if feature_definition is None:
        raise ValueError(
            f"Research sweep parameter path '{path}' references unknown feature '{feature_name}'."
        )

    params = feature_definition.get("params")
    if not isinstance(params, dict) or tokens[3] not in params:
        raise ValueError(
            f"Research sweep parameter path '{path}' must resolve to an existing feature params scalar leaf."
        )

    leaf_value = params[tokens[3]]
    if not _is_existing_scalar_leaf(leaf_value):
        raise ValueError(
            f"Research sweep parameter path '{path}' must resolve to an existing scalar leaf."
        )
    return params, tokens[3], leaf_value


def _generic_path_parent(
    project_config: dict[str, Any],
    tokens: list[str],
    path: str,
) -> tuple[dict[str, Any], str, Any]:
    if tokens[0] == "data":
        if len(tokens) != 2 or tokens[1] not in RESEARCH_SWEEP_DATA_LEAVES:
            raise ValueError(
                f"Research sweep parameter path '{path}' is not supported for data sweeps."
            )
    elif tuple(tokens) not in RESEARCH_SWEEP_RESEARCH_PATHS and not any(
        tuple(tokens[: len(prefix)]) == prefix and len(tokens) == len(prefix) + 1
        for prefix in RESEARCH_SWEEP_RESEARCH_PREFIXES
    ):
        raise ValueError(
            f"Research sweep parameter path '{path}' is not supported. Use research label/cost/tuning/evaluation leaves, selected data leaves, or features.<feature_name>.params.<param>."
        )

    current: Any = project_config
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise ValueError(
                f"Research sweep parameter path '{path}' must resolve to an existing scalar leaf."
            )
        current = current[token]

    if not isinstance(current, dict) or tokens[-1] not in current:
        raise ValueError(
            f"Research sweep parameter path '{path}' must resolve to an existing scalar leaf."
        )

    leaf_key = tokens[-1]
    leaf_value = current[leaf_key]
    if not _is_existing_scalar_leaf(leaf_value):
        raise ValueError(
            f"Research sweep parameter path '{path}' must resolve to an existing scalar leaf."
        )
    return current, leaf_key, leaf_value


def resolve_research_scalar_path(
    project_config: dict[str, Any],
    path: str,
) -> tuple[dict[str, Any], str, Any]:
    """Resolve a research sweep path against a project config."""

    tokens = _path_tokens(path)
    if tokens[0] == "features":
        return _feature_path_parent(project_config, tokens, path)
    if tokens[0] not in {"data", "research"}:
        raise ValueError(
            f"Research sweep parameter path '{path}' must start with data, research, or features."
        )
    return _generic_path_parent(project_config, tokens, path)


def apply_agent_scalar_overrides(
    agent_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a cloned agent config with validated scalar overrides applied."""

    updated = deepcopy(agent_config)
    for path, value in overrides.items():
        if not is_scalar_sweep_value(value):
            raise ValueError(
                f"Sweep parameter path '{path}' contains a non-scalar sweep value."
            )
        parent, leaf_key, _leaf_value = resolve_agent_scalar_path(updated, path)
        parent[leaf_key] = value
    return updated


def apply_research_scalar_overrides(
    project_config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a cloned project config with validated research overrides applied."""

    updated = deepcopy(project_config)
    for path, value in overrides.items():
        if not is_scalar_sweep_value(value):
            raise ValueError(
                f"Research sweep parameter path '{path}' contains a non-scalar sweep value."
            )
        parent, leaf_key, _leaf_value = resolve_research_scalar_path(updated, path)
        parent[leaf_key] = value
    return updated


def build_agent_sweep_variant_name(
    *,
    base_agent_name: str,
    sweep_name: str,
    parameters: dict[str, Any],
) -> str:
    """Build the deterministic variant name required by the CLI workflow."""

    parts = [
        _slugify_token(base_agent_name),
        _slugify_token(sweep_name),
    ]
    for path, value in parameters.items():
        leaf_name = _path_tokens(path)[-1]
        parts.append(f"{_slugify_token(leaf_name)}-{slug_scalar_sweep_value(value)}")
    return "__".join(parts)


def build_research_sweep_variant_name(
    *,
    project_name: str,
    sweep_name: str,
    parameters: dict[str, Any],
) -> str:
    """Build a deterministic research sweep variant name."""

    parts = [
        _slugify_token(project_name),
        _slugify_token(sweep_name),
    ]
    for path, value in parameters.items():
        leaf_name = _path_tokens(path)[-1]
        parts.append(f"{_slugify_token(leaf_name)}-{slug_scalar_sweep_value(value)}")
    return "__".join(parts)


def resolve_project_sweep(
    project_config: dict[str, Any],
    sweep_name: str,
) -> dict[str, Any]:
    """Return the named sweep definition from the project config."""

    for sweep in project_config.get("sweeps") or []:
        if str(sweep.get("name") or "").strip() == sweep_name:
            return dict(sweep)
    raise ValueError(f"Sweep '{sweep_name}' not found in project config.")


def expand_agent_backtest_sweep(
    project_config: dict[str, Any],
    sweep_name: str,
) -> dict[str, Any]:
    """Expand a named backtest sweep into fully materialized project variants."""

    sweep = resolve_project_sweep(project_config, sweep_name)
    if str(sweep.get("kind") or "").strip() != "agent_backtest":
        raise ValueError(
            f"Sweep '{sweep_name}' has unsupported kind '{sweep.get('kind')}'."
        )

    base_agent_name = str(sweep.get("agent") or "").strip()
    base_agent = next(
        (
            dict(agent)
            for agent in project_config.get("agents") or []
            if str(agent.get("name") or "").strip() == base_agent_name
        ),
        None,
    )
    if base_agent is None:
        raise ValueError(
            f"Sweep '{sweep_name}' references unknown agent '{base_agent_name}'."
        )

    parameters = [dict(parameter) for parameter in (sweep.get("parameters") or [])]
    if not parameters:
        raise ValueError(f"Sweep '{sweep_name}' must define at least one parameter.")

    parameter_paths: list[str] = []
    parameter_values: list[list[Any]] = []
    for parameter in parameters:
        path = str(parameter.get("path") or "").strip()
        values = list(parameter.get("values") or [])
        if not values:
            raise ValueError(
                f"Sweep '{sweep_name}' parameter '{path or '<blank>'}' must define at least one value."
            )
        resolve_agent_scalar_path(base_agent, path)
        parameter_paths.append(path)
        parameter_values.append(values)

    variants: list[dict[str, Any]] = []
    base_agents = list(project_config.get("agents") or [])
    for combination in product(*parameter_values):
        overrides = {
            path: value
            for path, value in zip(parameter_paths, combination, strict=True)
        }
        variant_agent = apply_agent_scalar_overrides(base_agent, overrides)
        variant_name = build_agent_sweep_variant_name(
            base_agent_name=base_agent_name,
            sweep_name=sweep_name,
            parameters=overrides,
        )
        variant_agent["name"] = variant_name

        variant_project = deepcopy(project_config)
        variant_project["agents"] = [
            (
                deepcopy(variant_agent)
                if str(agent.get("name") or "").strip() == base_agent_name
                else deepcopy(agent)
            )
            for agent in base_agents
        ]
        variant_project.pop("sweeps", None)
        variants.append(
            {
                "name": variant_name,
                "base_agent_name": base_agent_name,
                "parameters": dict(overrides),
                "project_config": variant_project,
                "agent_config": variant_agent,
            }
        )

    return {
        "name": sweep_name,
        "kind": "agent_backtest",
        "base_agent_name": base_agent_name,
        "parameters": parameters,
        "variants": variants,
    }


def expand_research_sweep(
    project_config: dict[str, Any],
    sweep_name: str,
) -> dict[str, Any]:
    """Expand a named research sweep into materialized project variants."""

    sweep = resolve_project_sweep(project_config, sweep_name)
    if str(sweep.get("kind") or "").strip() != "research_run":
        raise ValueError(
            f"Sweep '{sweep_name}' has unsupported kind '{sweep.get('kind')}'."
        )

    parameters = [dict(parameter) for parameter in (sweep.get("parameters") or [])]
    if not parameters:
        raise ValueError(f"Sweep '{sweep_name}' must define at least one parameter.")

    parameter_paths: list[str] = []
    parameter_values: list[list[Any]] = []
    for parameter in parameters:
        path = str(parameter.get("path") or "").strip()
        values = list(parameter.get("values") or [])
        if not values:
            raise ValueError(
                f"Sweep '{sweep_name}' parameter '{path or '<blank>'}' must define at least one value."
            )
        resolve_research_scalar_path(project_config, path)
        parameter_paths.append(path)
        parameter_values.append(values)

    project_name = str((project_config.get("project") or {}).get("name") or "research")
    variants: list[dict[str, Any]] = []
    for combination in product(*parameter_values):
        overrides = {
            path: value
            for path, value in zip(parameter_paths, combination, strict=True)
        }
        variant_project = apply_research_scalar_overrides(project_config, overrides)
        variant_project.pop("sweeps", None)
        variant_name = build_research_sweep_variant_name(
            project_name=project_name,
            sweep_name=sweep_name,
            parameters=overrides,
        )
        variants.append(
            {
                "name": variant_name,
                "parameters": dict(overrides),
                "project_config": variant_project,
            }
        )

    return {
        "name": sweep_name,
        "kind": "research_run",
        "parameters": parameters,
        "variants": variants,
    }


def sweep_summary_payload(
    *,
    sweep_name: str,
    base_agent_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Return the sweep summary payload stored in child run summaries."""

    return {
        "name": sweep_name,
        "base_agent_name": base_agent_name,
        "parameters": dict(parameters),
        "materializable": True,
    }
