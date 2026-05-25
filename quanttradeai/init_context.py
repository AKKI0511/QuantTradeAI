"""Package-resource backed workspace guidance files for `quanttradeai init`."""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Iterator


_TEMPLATE_ROOT_PARTS = ("templates", "workspace_context")
_PREFERRED_TEMPLATE_ORDER = (
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/skills/quanttradeai-research/SKILL.md",
    ".claude/skills/quanttradeai-research/workflow.md",
    ".claude/skills/quanttradeai-research/artifacts.md",
    ".claude/skills/quanttradeai-research/safety.md",
)


def _template_root() -> Traversable:
    root = resources.files("quanttradeai")
    for part in _TEMPLATE_ROOT_PARTS:
        root = root.joinpath(part)
    return root


def _normalize_template_path(relative_path: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(str(relative_path))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Invalid init context template path: {relative_path!r}")
    return path


def _walk_template_files(root: Traversable, prefix: PurePosixPath) -> Iterator[str]:
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        child_path = prefix / child.name
        if child.is_dir():
            yield from _walk_template_files(child, child_path)
        elif child.is_file():
            yield child_path.as_posix()


def iter_init_context_templates() -> tuple[str, ...]:
    """Return template-relative paths copied into initialized workspaces."""

    discovered = tuple(_walk_template_files(_template_root(), PurePosixPath()))
    preferred_order = {
        relative_path: index
        for index, relative_path in enumerate(_PREFERRED_TEMPLATE_ORDER)
    }
    return tuple(
        sorted(
            discovered,
            key=lambda relative_path: (
                preferred_order.get(relative_path, len(preferred_order)),
                relative_path,
            ),
        )
    )


def read_init_context_template(relative_path: str | PurePosixPath) -> str:
    """Read one init context template by template-relative path."""

    path = _normalize_template_path(relative_path)
    resource = _template_root()
    for part in path.parts:
        resource = resource.joinpath(part)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown init context template: {path.as_posix()}")
    return resource.read_text(encoding="utf-8")


def write_init_context_files(workspace: Path, force: bool) -> None:
    """Copy all init context templates into a workspace."""

    for relative_path in iter_init_context_templates():
        output_path = workspace / Path(*PurePosixPath(relative_path).parts)
        if output_path.exists() and not force:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = read_init_context_template(relative_path).rstrip() + "\n"
        output_path.write_text(content, encoding="utf-8")
