"""Helpers for loading and merging market impact defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping

import yaml


class ImpactConfigError(ValueError):
    """Raised when the impact configuration file is malformed."""


def load_impact_config(
    config_path: str | Path = "config/impact_config.yaml",
) -> Dict[str, dict]:
    """Load per-asset-class market impact defaults.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to ``impact_config.yaml``. Defaults to ``config/impact_config.yaml``.

    Returns
    -------
    Dict[str, dict]
        Mapping of asset class name to impact parameter dictionary. Returns an
        empty dictionary when the file does not exist.
    """

    path = Path(config_path)
    if not path.is_file():
        return {}

    try:
        with path.open("r") as file:
            raw_cfg = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive parsing guard
        raise ImpactConfigError(f"Failed to parse impact config at {path}: {exc}")

    asset_classes = raw_cfg.get("asset_classes")
    if not isinstance(asset_classes, Mapping):
        raise ImpactConfigError(
            f"impact config at {path} must define an 'asset_classes' mapping."
        )

    validated: Dict[str, dict] = {}
    for name, params in asset_classes.items():
        if not isinstance(params, Mapping):
            raise ImpactConfigError(
                f"Impact parameters for asset class {name!r} must be a mapping."
            )
        required_keys = {"alpha", "beta", "model"}
        missing = sorted(k for k in required_keys if k not in params)
        if missing:
            raise ImpactConfigError(
                f"Impact config for asset class {name!r} is missing keys: {', '.join(missing)}."
            )
        validated[name] = dict(params)

    return validated


def merge_execution_with_impact(
    execution_cfg: dict | None,
    impact_defaults: Mapping[str, dict],
    asset_class: str,
) -> dict:
    """Merge impact defaults for an asset class into an execution config.

    The merge order is:
    1. Asset-class defaults from ``impact_defaults`` (lowest priority)
    2. Provided ``execution_cfg`` values (highest priority, including CLI overrides)
    """

    merged = deepcopy(execution_cfg) if execution_cfg else {}
    defaults = impact_defaults.get(asset_class)
    if not defaults:
        return merged

    default_impact = dict(defaults)
    default_impact.setdefault("enabled", True)

    merged.setdefault("impact", {})
    merged["impact"] = {**default_impact, **merged["impact"]}
    return merged
