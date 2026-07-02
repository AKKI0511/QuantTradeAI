"""Read-only workspace health checks for QuantTradeAI."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import yaml

from quanttradeai import __version__ as PACKAGE_VERSION
from quanttradeai.utils.config_validator import validate_project_config_readonly
from quanttradeai.utils.project_paths import infer_project_root


DOCTOR_EXIT_OK = 0
DOCTOR_EXIT_ERROR = 1
QUANTTRADEAI_DEPENDENCY_RE = re.compile(
    r"^\s*quanttradeai\s*==\s*([A-Za-z0-9][A-Za-z0-9.!+_-]*)\s*$",
    re.IGNORECASE,
)
LLM_PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def _installed_quanttradeai_version() -> str:
    try:
        return metadata.version("quanttradeai")
    except metadata.PackageNotFoundError:
        return PACKAGE_VERSION


def _diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    message: str,
    action: str,
    path: str | None = None,
) -> None:
    item = {
        "severity": severity,
        "code": code,
        "message": message,
        "action": action,
    }
    if path:
        item["path"] = path
    diagnostics.append(item)


def _check(
    checks: list[dict[str, str]],
    *,
    name: str,
    status: str,
    summary: str,
) -> None:
    checks.append({"name": name, "status": status, "summary": summary})


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_pyproject(
    pyproject_path: Path,
    diagnostics: list[dict[str, Any]],
    root: Path,
) -> dict[str, Any] | None:
    if not pyproject_path.exists():
        _diagnostic(
            diagnostics,
            severity="warning",
            code="workspace.pyproject_missing",
            message="pyproject.toml was not found in the workspace root.",
            action="Run `quanttradeai init` from the workspace root, or add the generated pyproject.toml.",
            path=_relative(pyproject_path, root),
        )
        return None
    if not pyproject_path.is_file():
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.pyproject_not_file",
            message="pyproject.toml exists but is not a regular file.",
            action="Replace it with the generated QuantTradeAI pyproject.toml.",
            path=_relative(pyproject_path, root),
        )
        return None
    try:
        return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.pyproject_invalid",
            message=f"pyproject.toml is not valid TOML: {exc}",
            action="Fix pyproject.toml syntax, then rerun `quanttradeai doctor`.",
            path=_relative(pyproject_path, root),
        )
        return None


def _read_workspace_metadata(
    metadata_path: Path,
    diagnostics: list[dict[str, Any]],
    root: Path,
    *,
    source_checkout: bool,
) -> dict[str, Any] | None:
    if not metadata_path.exists():
        if not source_checkout:
            _diagnostic(
                diagnostics,
                severity="warning",
                code="workspace.metadata_missing",
                message=".quanttradeai/workspace.yaml was not found.",
                action="Run `quanttradeai init --force` only if this directory should be regenerated as a QuantTradeAI workspace.",
                path=_relative(metadata_path, root),
            )
        return None
    if not metadata_path.is_file():
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.metadata_not_file",
            message=".quanttradeai/workspace.yaml exists but is not a regular file.",
            action="Replace it with generated workspace metadata.",
            path=_relative(metadata_path, root),
        )
        return None
    try:
        payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.metadata_invalid",
            message=f".quanttradeai/workspace.yaml is not valid YAML: {exc}",
            action="Regenerate workspace metadata with `quanttradeai init --force` after preserving local changes.",
            path=_relative(metadata_path, root),
        )
        return None
    if not isinstance(payload, dict):
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.metadata_invalid",
            message=".quanttradeai/workspace.yaml must contain a YAML mapping.",
            action="Regenerate workspace metadata with `quanttradeai init --force` after preserving local changes.",
            path=_relative(metadata_path, root),
        )
        return None
    return payload


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[<>=!~ @\[]", requirement.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _find_quanttradeai_dependency(dependencies: Any) -> str | None:
    if not isinstance(dependencies, list):
        return None
    for dependency in dependencies:
        if (
            isinstance(dependency, str)
            and _dependency_name(dependency) == "quanttradeai"
        ):
            return dependency.strip()
    return None


def _looks_like_local_dependency(dependency: str) -> bool:
    normalized = dependency.lower()
    return bool(
        "file://" in normalized
        or re.search(r"@\s*file:", dependency, re.IGNORECASE)
        or re.search(r"@\s*(?:\.{1,2}[\\/]|/|[A-Za-z]:[\\/])", dependency)
    )


def _check_package_metadata(
    *,
    pyproject: dict[str, Any] | None,
    workspace_metadata: dict[str, Any] | None,
    root: Path,
    diagnostics: list[dict[str, Any]],
) -> tuple[bool, str, str | None]:
    installed_version = _installed_quanttradeai_version()
    project = dict((pyproject or {}).get("project") or {})
    project_name = str(project.get("name") or "").strip()
    pyproject_version = project.get("version")
    source_checkout = (
        project_name == "quanttradeai" and (root / "quanttradeai").is_dir()
    )
    pinned_dependency: str | None = None

    if source_checkout:
        if pyproject_version and str(pyproject_version) != installed_version:
            _diagnostic(
                diagnostics,
                severity="warning",
                code="package.version_mismatch",
                message=(
                    "The source checkout pyproject version differs from the imported "
                    f"QuantTradeAI version ({pyproject_version} != {installed_version})."
                ),
                action="Reinstall the editable package or update version metadata before release testing.",
                path="pyproject.toml",
            )
        return True, installed_version, None

    dependencies = project.get("dependencies")
    pinned_dependency = _find_quanttradeai_dependency(dependencies)
    if pinned_dependency is None:
        _diagnostic(
            diagnostics,
            severity="error",
            code="package.dependency_missing",
            message="Workspace pyproject.toml does not declare a QuantTradeAI dependency.",
            action="Regenerate with `quanttradeai init --force` or add `quanttradeai==<version>` to project.dependencies.",
            path="pyproject.toml",
        )
    elif _looks_like_local_dependency(pinned_dependency):
        _diagnostic(
            diagnostics,
            severity="error",
            code="package.dependency_local",
            message="Workspace pyproject.toml depends on a local QuantTradeAI checkout.",
            action="Replace the dependency with an exact released package pin such as `quanttradeai==0.1.0`.",
            path="pyproject.toml",
        )
    else:
        match = QUANTTRADEAI_DEPENDENCY_RE.fullmatch(pinned_dependency)
        if not match:
            _diagnostic(
                diagnostics,
                severity="error",
                code="package.dependency_unpinned",
                message="Workspace pyproject.toml must pin QuantTradeAI with `quanttradeai==<version>`.",
                action="Regenerate with `quanttradeai init --force` or change the dependency to an exact version pin.",
                path="pyproject.toml",
            )
        elif match.group(1) != installed_version:
            _diagnostic(
                diagnostics,
                severity="error",
                code="package.version_mismatch",
                message=(
                    "The installed QuantTradeAI CLI version does not match the workspace pin "
                    f"({installed_version} != {match.group(1)})."
                ),
                action="Run `uv sync` and then `uv run quanttradeai doctor`, or update the workspace pin intentionally.",
                path="pyproject.toml",
            )

    python_project = dict((workspace_metadata or {}).get("python_project") or {})
    metadata_dependency = python_project.get("quanttradeai_dependency")
    if (
        metadata_dependency
        and pinned_dependency
        and metadata_dependency != pinned_dependency
    ):
        _diagnostic(
            diagnostics,
            severity="error",
            code="package.metadata_mismatch",
            message=".quanttradeai/workspace.yaml disagrees with pyproject.toml about the QuantTradeAI dependency.",
            action="Regenerate workspace metadata with `quanttradeai init --force` after preserving local changes.",
            path=".quanttradeai/workspace.yaml",
        )
    if isinstance(metadata_dependency, str) and _looks_like_local_dependency(
        metadata_dependency
    ):
        _diagnostic(
            diagnostics,
            severity="error",
            code="package.metadata_local",
            message=".quanttradeai/workspace.yaml records a local QuantTradeAI checkout dependency.",
            action="Regenerate with a released package pin using the fixed `quanttradeai init`.",
            path=".quanttradeai/workspace.yaml",
        )

    return source_checkout, installed_version, pinned_dependency


def _uv_version() -> str | None:
    executable = shutil.which("uv")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (result.stdout or result.stderr).strip() or "unknown"


def _check_python_and_uv(
    *,
    root: Path,
    source_checkout: bool,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 11):
        _diagnostic(
            diagnostics,
            severity="error",
            code="environment.python_unsupported",
            message=f"Python {python_version} is active, but QuantTradeAI requires Python 3.11 or newer.",
            action="Switch to Python 3.11+ and rerun `uv sync`.",
        )

    python_version_file = root / ".python-version"
    if python_version_file.is_file():
        requested = python_version_file.read_text(encoding="utf-8").strip()
        if requested and not python_version.startswith(requested):
            _diagnostic(
                diagnostics,
                severity="warning",
                code="environment.python_version_file_mismatch",
                message=f".python-version requests {requested}, but the active Python is {python_version}.",
                action="Use the requested Python version or update .python-version intentionally.",
                path=".python-version",
            )

    uv = _uv_version()
    if uv is None:
        _diagnostic(
            diagnostics,
            severity="warning",
            code="environment.uv_missing",
            message="uv was not found on PATH.",
            action="Install uv before running workspace commands such as `uv sync`.",
        )

    if not source_checkout and not (root / ".venv").exists():
        _diagnostic(
            diagnostics,
            severity="warning",
            code="environment.venv_missing",
            message="The workspace .venv directory does not exist yet.",
            action="Run `uv sync` in the workspace before running research or agent commands.",
            path=".venv",
        )

    return {
        "python": python_version,
        "executable": sys.executable,
        "uv": uv,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
    }


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists():
        if current.parent == current:
            return None
        current = current.parent
    return current


def _check_output_paths(
    *,
    root: Path,
    diagnostics: list[dict[str, Any]],
) -> None:
    for relative_path in ("data", "models", "reports", "runs"):
        path = root / relative_path
        if path.exists() and not path.is_dir():
            _diagnostic(
                diagnostics,
                severity="error",
                code="outputs.path_not_directory",
                message=f"{relative_path}/ exists but is not a directory.",
                action=f"Move or remove {relative_path} so QuantTradeAI can write output artifacts there.",
                path=relative_path,
            )
            continue
        access_target = path if path.exists() else _nearest_existing_parent(path)
        if access_target is None or not os.access(access_target, os.W_OK):
            _diagnostic(
                diagnostics,
                severity="error",
                code="outputs.not_writable",
                message=f"QuantTradeAI cannot write to {relative_path}/ or its nearest existing parent.",
                action=f"Fix filesystem permissions for {relative_path}/ before running workflows.",
                path=relative_path,
            )


def _missing_env_vars(env: Mapping[str, str], names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not str(env.get(name) or "").strip()]


def _check_required_config(
    *,
    resolved: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    data = dict(resolved.get("data") or {})
    features = dict(resolved.get("features") or {})
    research = dict(resolved.get("research") or {})
    agents = list(resolved.get("agents") or [])
    project = dict(resolved.get("project") or {})
    profiles = dict(resolved.get("profiles") or {})

    profile = str(project.get("profile") or "").strip()
    if profile and profile not in profiles:
        _diagnostic(
            diagnostics,
            severity="error",
            code="config.profile_unknown",
            message=f"project.profile references unknown profile '{profile}'.",
            action="Set project.profile to one of the keys under profiles.",
            path="config/project.yaml",
        )
    if not data.get("symbols"):
        _diagnostic(
            diagnostics,
            severity="error",
            code="config.symbols_missing",
            message="data.symbols is empty.",
            action="Add at least one ticker to data.symbols.",
            path="config/project.yaml",
        )
    if not (features.get("definitions") or []):
        _diagnostic(
            diagnostics,
            severity="error",
            code="config.features_missing",
            message="features.definitions is empty.",
            action="Add at least one reusable feature definition.",
            path="config/project.yaml",
        )
    if not bool(research.get("enabled", False)) and not agents:
        _diagnostic(
            diagnostics,
            severity="error",
            code="config.no_workflow_enabled",
            message="The project has no enabled research workflow and no agents.",
            action="Enable research.enabled or define at least one agent.",
            path="config/project.yaml",
        )


def _check_credentials(
    *,
    resolved: dict[str, Any],
    env: Mapping[str, str],
    diagnostics: list[dict[str, Any]],
) -> None:
    data_streaming = dict((resolved.get("data") or {}).get("streaming") or {})
    replay_enabled = bool(
        dict(data_streaming.get("replay") or {}).get("enabled", False)
    )
    agents = list(resolved.get("agents") or [])

    for agent in agents:
        agent_name = str(agent.get("name") or "<unnamed>")
        agent_kind = str(agent.get("kind") or "").strip().lower()
        agent_mode = str(agent.get("mode") or "").strip().lower()
        llm_cfg = dict(agent.get("llm") or {})
        if agent_kind in {"llm", "hybrid"}:
            provider = str(llm_cfg.get("provider") or "").strip().lower()
            if not provider:
                _diagnostic(
                    diagnostics,
                    severity="error",
                    code="credentials.llm_provider_missing",
                    message=f"Agent '{agent_name}' does not configure llm.provider.",
                    action="Set llm.provider and the matching provider credential environment variable.",
                    path="config/project.yaml",
                )
            elif provider in LLM_PROVIDER_ENV:
                candidates = LLM_PROVIDER_ENV[provider]
                if all(not str(env.get(name) or "").strip() for name in candidates):
                    _diagnostic(
                        diagnostics,
                        severity="error",
                        code="credentials.llm_missing",
                        message=(
                            f"Agent '{agent_name}' uses provider '{provider}' but none "
                            f"of these environment variables are set: {', '.join(candidates)}."
                        ),
                        action="Set the provider API key in the environment before running this agent.",
                        path="config/project.yaml",
                    )
            else:
                _diagnostic(
                    diagnostics,
                    severity="warning",
                    code="credentials.llm_provider_unknown",
                    message=f"Agent '{agent_name}' uses unrecognized llm.provider '{provider}'.",
                    action="Verify the provider's required environment variables before running the agent.",
                    path="config/project.yaml",
                )

        execution_backend = (
            str(dict(agent.get("execution") or {}).get("backend") or "simulated")
            .strip()
            .lower()
        )
        uses_alpaca_backend = execution_backend == "alpaca"
        uses_realtime_alpaca = (
            agent_mode == "live"
            and str(data_streaming.get("provider") or "").strip().lower() == "alpaca"
        )
        if uses_alpaca_backend or (uses_realtime_alpaca and not replay_enabled):
            missing = _missing_env_vars(env, ("ALPACA_API_KEY", "ALPACA_API_SECRET"))
            if missing:
                _diagnostic(
                    diagnostics,
                    severity="error",
                    code="credentials.alpaca_missing",
                    message=(
                        f"Agent '{agent_name}' needs Alpaca credentials but these "
                        f"environment variables are missing: {', '.join(missing)}."
                    ),
                    action="Set ALPACA_API_KEY and ALPACA_API_SECRET before running Alpaca-backed paper or live workflows.",
                    path="config/project.yaml",
                )


def _check_safety_defaults(
    *,
    resolved: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> None:
    agents = list(resolved.get("agents") or [])
    live_agents = [
        agent
        for agent in agents
        if str(agent.get("mode") or "").strip().lower() == "live"
    ]
    paper_agents = [
        agent
        for agent in agents
        if str(agent.get("mode") or "").strip().lower() == "paper"
    ]
    risk = dict(resolved.get("risk") or {})
    position_manager = dict(resolved.get("position_manager") or {})
    data_streaming = dict((resolved.get("data") or {}).get("streaming") or {})
    replay_enabled = bool(
        dict(data_streaming.get("replay") or {}).get("enabled", False)
    )

    if live_agents:
        drawdown = dict(risk.get("drawdown_protection") or {})
        if not risk:
            _diagnostic(
                diagnostics,
                severity="error",
                code="safety.live_risk_missing",
                message="At least one agent is configured for live mode but top-level risk is missing.",
                action="Add a top-level risk section with drawdown protection before live runs.",
                path="config/project.yaml",
            )
        elif drawdown.get("enabled") is not True:
            _diagnostic(
                diagnostics,
                severity="error",
                code="safety.drawdown_disabled",
                message="Live mode is configured but risk.drawdown_protection.enabled is not true.",
                action="Enable drawdown protection or keep agents in paper mode.",
                path="config/project.yaml",
            )
        if not position_manager:
            _diagnostic(
                diagnostics,
                severity="error",
                code="safety.position_manager_missing",
                message="At least one agent is configured for live mode but position_manager is missing.",
                action="Add top-level position_manager settings before live runs.",
                path="config/project.yaml",
            )
        elif str(position_manager.get("mode") or "").strip().lower() != "live":
            _diagnostic(
                diagnostics,
                severity="warning",
                code="safety.position_manager_mode",
                message="Live agents are configured but position_manager.mode is not live.",
                action="Set position_manager.mode: live before live operation.",
                path="config/project.yaml",
            )

        for agent in live_agents:
            agent_name = str(agent.get("name") or "<unnamed>")
            agent_risk = dict(agent.get("risk") or {})
            if "max_position_pct" not in agent_risk:
                _diagnostic(
                    diagnostics,
                    severity="error",
                    code="safety.agent_position_limit_missing",
                    message=f"Live agent '{agent_name}' does not set risk.max_position_pct.",
                    action="Add a per-agent max_position_pct limit before live operation.",
                    path="config/project.yaml",
                )

    if paper_agents and data_streaming.get("enabled") and not replay_enabled:
        _diagnostic(
            diagnostics,
            severity="warning",
            code="safety.paper_replay_disabled",
            message="Paper agents are configured but data.streaming.replay.enabled is not true.",
            action="Prefer replay-backed paper mode unless you intentionally need realtime paper data.",
            path="config/project.yaml",
        )


def doctor_workspace(
    *,
    workspace: Path | str | None = None,
    config_path: Path | str = "config/project.yaml",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run read-only health checks for the current QuantTradeAI workspace."""

    env = env or os.environ
    cwd = Path.cwd().resolve() if workspace is None else Path(workspace).resolve()
    requested_config = Path(config_path).expanduser()
    config_abs = (
        requested_config if requested_config.is_absolute() else cwd / requested_config
    )
    root = infer_project_root(config_abs)
    diagnostics: list[dict[str, Any]] = []
    checks: list[dict[str, str]] = []

    pyproject_path = root / "pyproject.toml"
    pyproject = _read_pyproject(pyproject_path, diagnostics, root)
    project = dict((pyproject or {}).get("project") or {})
    source_checkout = (
        str(project.get("name") or "").strip() == "quanttradeai"
        and (root / "quanttradeai").is_dir()
    )
    metadata_path = root / ".quanttradeai" / "workspace.yaml"
    workspace_metadata = _read_workspace_metadata(
        metadata_path,
        diagnostics,
        root,
        source_checkout=source_checkout,
    )

    source_checkout, installed_version, pinned_dependency = _check_package_metadata(
        pyproject=pyproject,
        workspace_metadata=workspace_metadata,
        root=root,
        diagnostics=diagnostics,
    )

    if not config_abs.exists():
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.config_missing",
            message=f"Project config was not found at {_relative(config_abs, root)}.",
            action="Run `quanttradeai init` or pass the correct path with `--config`.",
            path=_relative(config_abs, root),
        )
        validation: dict[str, Any] | None = None
    elif not config_abs.is_file():
        _diagnostic(
            diagnostics,
            severity="error",
            code="workspace.config_not_file",
            message=f"Project config path is not a regular file: {_relative(config_abs, root)}.",
            action="Replace it with a valid config/project.yaml file.",
            path=_relative(config_abs, root),
        )
        validation = None
    else:
        try:
            validation = validate_project_config_readonly(config_abs)
        except Exception as exc:
            _diagnostic(
                diagnostics,
                severity="error",
                code="project.config_invalid",
                message=f"config/project.yaml failed validation: {exc}",
                action="Fix the reported project YAML issue, then rerun `quanttradeai doctor`.",
                path=_relative(config_abs, root),
            )
            validation = None

    environment = _check_python_and_uv(
        root=root,
        source_checkout=source_checkout,
        diagnostics=diagnostics,
    )
    _check_output_paths(root=root, diagnostics=diagnostics)

    if validation is not None:
        for warning in validation.get("warnings") or []:
            _diagnostic(
                diagnostics,
                severity="warning",
                code="project.config_warning",
                message=str(warning),
                action="Update config/project.yaml to use the canonical setting.",
                path=_relative(config_abs, root),
            )
        resolved = dict(validation.get("resolved") or {})
        _check_required_config(resolved=resolved, diagnostics=diagnostics)
        _check_credentials(resolved=resolved, env=env, diagnostics=diagnostics)
        _check_safety_defaults(resolved=resolved, diagnostics=diagnostics)

    error_count = sum(1 for item in diagnostics if item["severity"] == "error")
    warning_count = sum(1 for item in diagnostics if item["severity"] == "warning")

    _check(
        checks,
        name="workspace",
        status=(
            "failed"
            if any(
                d["code"].startswith("workspace.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Workspace files are present and readable.",
    )
    _check(
        checks,
        name="environment",
        status=(
            "failed"
            if any(
                d["code"].startswith("environment.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Python and uv environment checked.",
    )
    _check(
        checks,
        name="package",
        status=(
            "failed"
            if any(
                d["code"].startswith("package.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="QuantTradeAI package metadata checked.",
    )
    _check(
        checks,
        name="project_config",
        status=(
            "failed"
            if any(
                (d["code"].startswith("project.") or d["code"].startswith("config."))
                and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Project YAML validated read-only.",
    )
    _check(
        checks,
        name="outputs",
        status=(
            "failed"
            if any(
                d["code"].startswith("outputs.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Output paths checked without creating files.",
    )
    _check(
        checks,
        name="credentials",
        status=(
            "failed"
            if any(
                d["code"].startswith("credentials.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Relevant credential environment variables checked.",
    )
    _check(
        checks,
        name="safety",
        status=(
            "failed"
            if any(
                d["code"].startswith("safety.") and d["severity"] == "error"
                for d in diagnostics
            )
            else "passed"
        ),
        summary="Paper/live safety defaults checked.",
    )

    exit_code = DOCTOR_EXIT_ERROR if error_count else DOCTOR_EXIT_OK
    return {
        "status": "error" if error_count else "ok",
        "exit_code": exit_code,
        "workspace": str(root),
        "config_path": str(config_abs),
        "source_checkout": source_checkout,
        "package": {
            "installed_version": installed_version,
            "workspace_dependency": pinned_dependency,
        },
        "environment": environment,
        "summary": {
            "checks": len(checks),
            "errors": error_count,
            "warnings": warning_count,
        },
        "checks": checks,
        "diagnostics": diagnostics,
    }


def render_doctor_text(result: Mapping[str, Any]) -> str:
    """Render concise text diagnostics for humans and coding agents."""

    summary = dict(result.get("summary") or {})
    diagnostics = list(result.get("diagnostics") or [])
    lines = [
        "QuantTradeAI doctor: "
        f"{str(result.get('status', 'unknown')).upper()} "
        f"({summary.get('errors', 0)} errors, {summary.get('warnings', 0)} warnings)",
        f"workspace: {result.get('workspace')}",
        f"config: {result.get('config_path')}",
    ]
    if not diagnostics:
        lines.append("All required checks passed.")
        return "\n".join(lines)

    for item in diagnostics:
        location = f" ({item['path']})" if item.get("path") else ""
        lines.append(
            f"[{item['severity']}] {item['code']}{location}: {item['message']}"
        )
        lines.append(f"  action: {item['action']}")
    return "\n".join(lines)
