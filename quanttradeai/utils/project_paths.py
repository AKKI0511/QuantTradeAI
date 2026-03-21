"""Helpers for resolving project-relative paths."""

from __future__ import annotations

from pathlib import Path


def infer_project_root(config_path: str | Path) -> Path:
    """Infer the project root from a project config path."""

    path = Path(config_path).resolve()
    if path.parent.name == "config":
        return path.parent.parent
    return path.parent


def resolve_project_path(config_path: str | Path, candidate: str | Path) -> Path:
    """Resolve an asset path relative to the inferred project root."""

    path = Path(candidate)
    if path.is_absolute():
        return path
    return infer_project_root(config_path) / path
