"""Deployment bundle generation for canonical project agents."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from quanttradeai.brokers import resolve_execution_backend
from quanttradeai.streaming.env_vars import provider_env_var_prefix
from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.project_config import (
    compile_live_position_manager_runtime_config,
    compile_live_risk_runtime_config,
    compile_live_streaming_runtime_config,
    compile_paper_streaming_runtime_config,
)
from quanttradeai.utils.project_paths import infer_project_root, resolve_project_path


DEFAULT_LLM_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}
SUPPORTED_DEPLOY_TARGETS = {"docker-compose", "local"}
SUPPORTED_DEPLOY_MODES = {"paper", "live"}
SUPPORTED_AGENT_KINDS = {"rule", "model", "llm", "hybrid"}


@dataclass
class PreparedDeployment:
    config_path: Path
    project_root: Path
    output_path: Path
    project_config: dict[str, Any]
    warnings: list[str]
    target: str
    mode: str
    agent_config: dict[str, Any]
    service_name: str
    safety_requirements: list[str]
    execution_backend: str
    broker_provider: str | None
    env_vars: list[tuple[str, str]]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip().lower())
    return normalized.strip("-_") or "agent"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _path_to_posix(value: Path | str) -> str:
    return Path(value).as_posix()


def _relative_posix(target: Path, base: Path) -> str:
    return Path(os.path.relpath(target, start=base)).as_posix()


def _ensure_project_relative_dir(
    *,
    project_root: Path,
    resolved_path: Path,
    expected_top_level_dir: str,
    field_name: str,
) -> None:
    try:
        relative = resolved_path.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must resolve inside the project root for docker-compose deployment."
        ) from exc

    if not relative.parts or relative.parts[0] != expected_top_level_dir:
        raise ValueError(
            f"{field_name} must resolve under {expected_top_level_dir}/ for docker-compose deployment."
        )


def _required_env_vars(
    *,
    agent_config: dict[str, Any],
    project_config: dict[str, Any],
) -> list[tuple[str, str]]:
    required: OrderedDict[str, str] = OrderedDict()
    llm_cfg = dict(agent_config.get("llm") or {})
    if llm_cfg:
        provider = str(llm_cfg.get("provider") or "").strip().lower()
        env_var = str(
            llm_cfg.get("api_key_env_var")
            or DEFAULT_LLM_API_KEY_ENV_VARS.get(provider)
            or ""
        ).strip()
        if env_var:
            required[env_var] = f"API key for the {provider or 'LLM'} provider."

    streaming_cfg = dict((project_config.get("data") or {}).get("streaming") or {})
    provider = str(streaming_cfg.get("provider") or "").strip()
    if provider:
        prefix = provider_env_var_prefix(provider)
        required[f"{prefix}_API_KEY"] = f"API key for streaming provider '{provider}'."
        required[f"{prefix}_API_SECRET"] = (
            f"API secret for streaming provider '{provider}'."
        )

    return list(required.items())


def _compose_environment(env_vars: list[tuple[str, str]]) -> dict[str, str]:
    return {name: f"${{{name}:-}}" for name, _ in env_vars}


def _render_env_example(env_vars: list[tuple[str, str]], *, target_label: str) -> str:
    lines = [
        f"# QuantTradeAI {target_label} deployment environment",
        "# Fill the values needed by this deployment bundle before running it.",
        "",
    ]
    if not env_vars:
        lines.append(
            "# No provider environment variables were inferred for this bundle."
        )
        return "\n".join(lines) + "\n"

    for name, description in env_vars:
        lines.append(f"# {description}")
        lines.append(f'{name}=""')
        lines.append("")
    return "\n".join(lines)


def _render_dockerfile() -> str:
    return """FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN mkdir -p /app/config /app/data /app/models /app/prompts /app/reports /app/runs

COPY pyproject.toml README.md /app/
COPY quanttradeai /app/quanttradeai

RUN pip install --upgrade pip && pip install .
"""


def _render_bundle_readme(
    *,
    agent_name: str,
    agent_kind: str,
    mode: str,
    execution_backend: str,
    broker_provider: str | None,
    service_name: str,
    next_command: str,
    env_vars: list[tuple[str, str]],
    safety_requirements: list[str],
) -> str:
    lines = [
        "# QuantTradeAI Deployment Bundle",
        "",
        f"This bundle deploys the `{agent_name}` `{agent_kind}` agent in `{mode}` mode with Docker Compose.",
        "",
        "## Mode",
        "",
        f"- `mode: {mode}`",
        f"- `execution.backend: {execution_backend}`",
        "",
        "## Files",
        "",
        f"- `docker-compose.yml`: runs the {mode} agent with the resolved project config mounted at `/app/config/project.yaml`.",
        "- `Dockerfile`: builds a minimal QuantTradeAI runtime image from the project source.",
        "- `.env.example`: provider environment variables inferred from the project config.",
        "- `resolved_project_config.yaml`: exact project config snapshot used for deployment generation.",
        "- `deployment_manifest.json`: machine-readable summary of the generated bundle.",
        "",
        "## Run",
        "",
        "From this directory:",
        "",
        "```bash",
        next_command,
        "```",
        "",
        "The compose service mounts project `data/`, `runs/`, and `reports/` read-write so runtime artifacts stay on the host.",
    ]

    if execution_backend == "alpaca" and broker_provider:
        lines.extend(
            [
                "",
                "This bundle uses broker-backed execution.",
                f"In `{mode}` mode it will submit real `{broker_provider}` market orders instead of simulated local fills.",
            ]
        )

    if safety_requirements:
        lines.extend(
            [
                "",
                "## Safety Requirements",
                "",
            ]
        )
        lines.extend([f"- {requirement}" for requirement in safety_requirements])

    if env_vars:
        lines.extend(
            [
                "",
                "## Environment",
                "",
                "Set the variables listed in `.env.example` before starting the service.",
            ]
        )

    return "\n".join(lines) + "\n"


def _render_local_runner_script(
    *,
    agent_name: str,
    mode: str,
    project_root: Path,
) -> str:
    project_root_literal = json.dumps(project_root.as_posix())
    agent_name_literal = json.dumps(agent_name)
    mode_literal = json.dumps(mode)
    return f'''"""Run a QuantTradeAI agent from this local deployment bundle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path({project_root_literal})
CONFIG_PATH = BUNDLE_DIR / "resolved_project_config.yaml"
AGENT_NAME = {agent_name_literal}
MODE = {mode_literal}


def _strip_simple_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {{"'", '"'}}:
        return value[1:-1]
    return value


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_simple_quotes(value.strip())
        if not key or not value:
            continue
        os.environ.setdefault(key, value)


def main() -> int:
    _load_env_file(BUNDLE_DIR / ".env")
    command = [
        sys.executable,
        "-m",
        "quanttradeai.cli",
        "agent",
        "run",
        "--agent",
        AGENT_NAME,
        "-c",
        str(CONFIG_PATH),
        "--mode",
        MODE,
    ]
    return subprocess.call(command, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _render_local_bundle_readme(
    *,
    agent_name: str,
    agent_kind: str,
    mode: str,
    execution_backend: str,
    broker_provider: str | None,
    next_command: str,
    env_vars: list[tuple[str, str]],
    safety_requirements: list[str],
) -> str:
    lines = [
        "# QuantTradeAI Local Deployment Bundle",
        "",
        f"This bundle runs the `{agent_name}` `{agent_kind}` agent in `{mode}` mode on the local machine.",
        "",
        "## Mode",
        "",
        f"- `mode: {mode}`",
        f"- `execution.backend: {execution_backend}`",
        "",
        "## Files",
        "",
        "- `run.py`: starts the agent with the resolved project config in this bundle.",
        "- `.env.example`: provider environment variables inferred from the project config.",
        "- `resolved_project_config.yaml`: exact project config snapshot used for deployment generation.",
        "- `deployment_manifest.json`: machine-readable summary of the generated bundle.",
        "",
        "## Run",
        "",
        "From the project environment:",
        "",
        "```bash",
        next_command,
        "```",
        "",
        "The runner executes from the original project root so `data/`, `runs/`, and `reports/` stay in the project.",
    ]

    if execution_backend == "alpaca" and broker_provider:
        lines.extend(
            [
                "",
                "This bundle uses broker-backed execution.",
                f"In `{mode}` mode it will submit real `{broker_provider}` market orders instead of simulated local fills.",
            ]
        )

    if safety_requirements:
        lines.extend(["", "## Safety Requirements", ""])
        lines.extend([f"- {requirement}" for requirement in safety_requirements])

    if env_vars:
        lines.extend(
            [
                "",
                "## Environment",
                "",
                "Create a `.env` file next to `run.py` with the variables listed in `.env.example`, or export them before starting the runner.",
            ]
        )

    return "\n".join(lines) + "\n"


def _copy_resolved_project_config(
    *,
    config_path: Path | str,
    destination: Path,
) -> tuple[dict[str, Any], list[str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        validation = validate_project_config(
            config_path=config_path,
            output_dir=temp_dir,
            timestamp_subdir=False,
        )
        warnings = list(validation.get("warnings", []))
        resolved_path = Path(validation["artifacts"]["resolved_config"])
        resolved_project = (
            yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        )

    destination.write_text(
        yaml.safe_dump(resolved_project, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_project, warnings


def _disable_replay_for_deployment(project_config: dict[str, Any]) -> dict[str, Any]:
    deployment_project = copy.deepcopy(project_config)
    streaming_cfg = dict((deployment_project.get("data") or {}).get("streaming") or {})
    if not streaming_cfg:
        return deployment_project
    replay_cfg = dict(streaming_cfg.get("replay") or {})
    if replay_cfg:
        replay_cfg["enabled"] = False
        streaming_cfg["replay"] = replay_cfg
        deployment_project.setdefault("data", {})["streaming"] = streaming_cfg
    return deployment_project


def _absolute_project_path(config_path: Path, candidate: Any) -> str:
    if candidate is None:
        return ""
    candidate_text = str(candidate).strip()
    if not candidate_text:
        return candidate_text
    return resolve_project_path(config_path, candidate_text).resolve().as_posix()


def _absolutize_local_project_paths(
    *,
    project_config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Make bundle config asset paths independent from the bundle location."""

    local_project = copy.deepcopy(project_config)
    for agent in local_project.get("agents") or []:
        if not isinstance(agent, dict):
            continue

        llm_cfg = agent.get("llm")
        if isinstance(llm_cfg, dict) and llm_cfg.get("prompt_file"):
            llm_cfg["prompt_file"] = _absolute_project_path(
                config_path, llm_cfg["prompt_file"]
            )

        model_cfg = agent.get("model")
        if isinstance(model_cfg, dict) and model_cfg.get("path"):
            model_cfg["path"] = _absolute_project_path(config_path, model_cfg["path"])

        for source in agent.get("model_signal_sources") or []:
            if isinstance(source, dict) and source.get("path"):
                source["path"] = _absolute_project_path(config_path, source["path"])

        context_cfg = agent.get("context")
        if not isinstance(context_cfg, dict):
            continue

        notes_cfg = context_cfg.get("notes")
        agent_name = str(agent.get("name") or "agent").strip() or "agent"
        if notes_cfg is True:
            context_cfg["notes"] = {
                "enabled": True,
                "file": _absolute_project_path(config_path, f"notes/{agent_name}.md"),
            }
        elif isinstance(notes_cfg, dict) and notes_cfg.get("enabled", True):
            notes_cfg["file"] = _absolute_project_path(
                config_path, notes_cfg.get("file") or f"notes/{agent_name}.md"
            )

    return local_project


def _deployment_safety_requirements(mode: str) -> list[str]:
    if mode != "live":
        return []

    return [
        "The agent must already be configured with `mode: live` in `config/project.yaml`.",
        "Real-time streaming settings must include `data.streaming.provider`, `data.streaming.websocket_url`, and `data.streaming.channels`.",
        "Top-level `risk` must be present and pass live-runtime validation.",
        "Top-level `position_manager` must be present and pass live-runtime validation.",
    ]


def _resolve_agent_config(
    *,
    project_config: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    agent_config = next(
        (
            dict(agent)
            for agent in project_config.get("agents") or []
            if agent.get("name") == agent_name
        ),
        None,
    )
    if agent_config is None:
        raise ValueError(f"Agent '{agent_name}' not found in project config.")
    agent_kind = str(agent_config.get("kind") or "").strip().lower()
    if agent_kind not in SUPPORTED_AGENT_KINDS:
        raise ValueError(
            f"Unsupported agent kind for deployment: {agent_kind or '<missing>'}."
        )
    return agent_config


def _required_mounts(
    *,
    project_root: Path,
    config_path: Path,
    agent_config: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    mounts = [
        {
            "host": output_dir / "resolved_project_config.yaml",
            "container": Path("/app/config/project.yaml"),
            "read_only": True,
        },
        {
            "host": project_root / "data",
            "container": Path("/app/data"),
            "read_only": False,
        },
        {
            "host": project_root / "runs",
            "container": Path("/app/runs"),
            "read_only": False,
        },
        {
            "host": project_root / "reports",
            "container": Path("/app/reports"),
            "read_only": False,
        },
    ]

    llm_cfg = dict(agent_config.get("llm") or {})
    if llm_cfg:
        prompt_path = resolve_project_path(config_path, llm_cfg.get("prompt_file", ""))
        _ensure_project_relative_dir(
            project_root=project_root,
            resolved_path=prompt_path,
            expected_top_level_dir="prompts",
            field_name="agents.<name>.llm.prompt_file",
        )
        mounts.append(
            {
                "host": project_root / "prompts",
                "container": Path("/app/prompts"),
                "read_only": True,
            }
        )

    needs_models_mount = False
    model_cfg = dict(agent_config.get("model") or {})
    if model_cfg:
        model_path = resolve_project_path(config_path, model_cfg.get("path", ""))
        _ensure_project_relative_dir(
            project_root=project_root,
            resolved_path=model_path,
            expected_top_level_dir="models",
            field_name="agents.<name>.model.path",
        )
        needs_models_mount = True

    for source in agent_config.get("model_signal_sources") or []:
        if not isinstance(source, dict):
            raise ValueError(
                "Docker-compose deployment requires object model_signal_sources entries with name and path."
            )
        source_path = resolve_project_path(config_path, source.get("path", ""))
        _ensure_project_relative_dir(
            project_root=project_root,
            resolved_path=source_path,
            expected_top_level_dir="models",
            field_name=f"agents.<name>.model_signal_sources.{source.get('name')}.path",
        )
        needs_models_mount = True

    if needs_models_mount:
        mounts.append(
            {
                "host": project_root / "models",
                "container": Path("/app/models"),
                "read_only": False,
            }
        )

    # Keep order stable and avoid duplicate mounts if an agent reuses prompts/models.
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Path, Path]] = set()
    for mount in mounts:
        key = (Path(mount["host"]), Path(mount["container"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mount)
    return deduped


def _prepare_deployment(
    *,
    agent_name: str,
    config_path: Path | str,
    target: str | None,
    mode: str | None,
    output_dir: Path | str | None,
    force: bool,
) -> PreparedDeployment:
    config_path = Path(config_path).resolve()
    project_root = infer_project_root(config_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_path = project_root / "reports" / "deployments" / agent_name / timestamp
    else:
        output_path = Path(output_dir)
    output_path = output_path.resolve()

    if output_path.exists() and not output_path.is_dir():
        raise FileExistsError(
            f"Deployment output path already exists and is not a directory: {output_path}"
        )
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing deployment bundle: {output_path}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_resolved_project_path = Path(temp_dir) / "resolved_project_config.yaml"
        resolved_project, warnings = _copy_resolved_project_config(
            config_path=config_path,
            destination=temp_resolved_project_path,
        )

    deployment_cfg = dict((resolved_project.get("deployment") or {}))
    resolved_target = str(target or deployment_cfg.get("target") or "docker-compose")
    resolved_target = resolved_target.strip().lower()
    if resolved_target not in SUPPORTED_DEPLOY_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_DEPLOY_TARGETS))
        raise ValueError(
            f"Unsupported deployment target '{resolved_target}'. Supported targets: {supported}."
        )

    resolved_mode = str(mode or deployment_cfg.get("mode") or "paper")
    resolved_mode = resolved_mode.strip().lower()
    if resolved_mode not in SUPPORTED_DEPLOY_MODES:
        supported = ", ".join(sorted(SUPPORTED_DEPLOY_MODES))
        raise ValueError(
            f"Unsupported deployment mode '{resolved_mode}'. Supported modes: {supported}."
        )

    deployment_project = copy.deepcopy(resolved_project)
    if resolved_mode == "paper":
        deployment_project = _disable_replay_for_deployment(deployment_project)
        if (
            ((resolved_project.get("data") or {}).get("streaming") or {}).get("replay")
            or {}
        ).get("enabled") is True:
            warnings.append(
                "Paper deployment bundles always use real-time streaming; replay was disabled in the generated bundle."
            )

    if resolved_target == "local":
        deployment_project = _absolutize_local_project_paths(
            project_config=deployment_project,
            config_path=config_path,
        )

    agent_config = _resolve_agent_config(
        project_config=deployment_project,
        agent_name=agent_name,
    )
    configured_mode = str(agent_config.get("mode") or "").strip().lower()
    if resolved_mode == "live" and configured_mode != "live":
        raise ValueError(
            f"Agent '{agent_name}' must be configured with mode=live before generating a live deployment bundle."
        )
    if resolved_mode != "live" and configured_mode and configured_mode != resolved_mode:
        warnings.append(
            f"Agent '{agent_name}' is configured with mode={configured_mode} but deployment mode={resolved_mode}; continuing with deployment mode."
        )

    if resolved_mode == "paper":
        compile_paper_streaming_runtime_config(
            deployment_project,
            require_realtime=True,
        )
    else:
        compile_live_streaming_runtime_config(deployment_project)
        compile_live_risk_runtime_config(deployment_project)
        compile_live_position_manager_runtime_config(deployment_project)

    execution_backend = resolve_execution_backend(agent_config)
    broker_provider = (
        str(
            ((deployment_project.get("data") or {}).get("streaming") or {}).get(
                "provider"
            )
            or ""
        ).strip()
        if execution_backend == "alpaca"
        else None
    )

    return PreparedDeployment(
        config_path=config_path,
        project_root=project_root,
        output_path=output_path,
        project_config=deployment_project,
        warnings=warnings,
        target=resolved_target,
        mode=resolved_mode,
        agent_config=agent_config,
        service_name=_slugify(f"{agent_name}-{resolved_mode}"),
        safety_requirements=_deployment_safety_requirements(resolved_mode),
        execution_backend=execution_backend,
        broker_provider=broker_provider,
        env_vars=_required_env_vars(
            agent_config=agent_config,
            project_config=deployment_project,
        ),
    )


def _write_docker_compose_bundle(
    prepared: PreparedDeployment,
    *,
    agent_name: str,
) -> dict[str, Any]:
    output_path = prepared.output_path
    mounts = _required_mounts(
        project_root=prepared.project_root,
        config_path=prepared.config_path,
        agent_config=prepared.agent_config,
        output_dir=output_path,
    )

    compose_command = [
        "quanttradeai",
        "agent",
        "run",
        "--agent",
        agent_name,
        "-c",
        "config/project.yaml",
        "--mode",
        prepared.mode,
    ]

    build_context = _relative_posix(prepared.project_root, output_path)
    dockerfile_rel = _relative_posix(output_path / "Dockerfile", prepared.project_root)
    compose_payload = {
        "services": {
            prepared.service_name: {
                "build": {
                    "context": build_context,
                    "dockerfile": dockerfile_rel,
                },
                "working_dir": "/app",
                "command": compose_command,
                "restart": "unless-stopped",
                "environment": _compose_environment(prepared.env_vars),
                "volumes": [
                    (
                        f"{_relative_posix(Path(mount['host']), output_path)}:"
                        f"{_path_to_posix(Path(mount['container']))}"
                        f"{':ro' if mount['read_only'] else ''}"
                    )
                    for mount in mounts
                ],
            }
        }
    }

    output_path.mkdir(parents=True, exist_ok=True)

    resolved_project_path = output_path / "resolved_project_config.yaml"
    resolved_project_path.write_text(
        yaml.safe_dump(prepared.project_config, sort_keys=False),
        encoding="utf-8",
    )

    dockerfile_path = output_path / "Dockerfile"
    dockerfile_path.write_text(_render_dockerfile(), encoding="utf-8")

    compose_path = output_path / "docker-compose.yml"
    compose_path.write_text(
        yaml.safe_dump(compose_payload, sort_keys=False),
        encoding="utf-8",
    )

    env_example_path = output_path / ".env.example"
    env_example_path.write_text(
        _render_env_example(prepared.env_vars, target_label="docker-compose"),
        encoding="utf-8",
    )

    next_command = f"docker compose -f {compose_path.as_posix()} up --build {prepared.service_name}"
    bundle_readme_path = output_path / "README.md"
    bundle_readme_path.write_text(
        _render_bundle_readme(
            agent_name=agent_name,
            agent_kind=str(prepared.agent_config.get("kind") or ""),
            mode=prepared.mode,
            execution_backend=prepared.execution_backend,
            broker_provider=prepared.broker_provider,
            service_name=prepared.service_name,
            next_command=next_command,
            env_vars=prepared.env_vars,
            safety_requirements=prepared.safety_requirements,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "compose": compose_path.as_posix(),
        "dockerfile": dockerfile_path.as_posix(),
        "env_example": env_example_path.as_posix(),
        "readme": bundle_readme_path.as_posix(),
        "resolved_project_config": resolved_project_path.as_posix(),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "agent_kind": prepared.agent_config.get("kind"),
        "target": prepared.target,
        "mode": prepared.mode,
        "execution_backend": prepared.execution_backend,
        "broker_provider": prepared.broker_provider,
        "service_name": prepared.service_name,
        "project_root": prepared.project_root.as_posix(),
        "source_config": prepared.config_path.as_posix(),
        "command": compose_command,
        "safety_requirements": prepared.safety_requirements,
        "environment_variables": [name for name, _ in prepared.env_vars],
        "volumes": [
            {
                "host": _path_to_posix(Path(mount["host"])),
                "container": _path_to_posix(Path(mount["container"])),
                "read_only": bool(mount["read_only"]),
            }
            for mount in mounts
        ],
        "artifacts": artifacts,
        "warnings": prepared.warnings,
        "next_command": next_command,
    }
    return _write_manifest_and_result(
        prepared=prepared,
        manifest=manifest,
        artifacts=artifacts,
        next_command=next_command,
    )


def _write_local_bundle(
    prepared: PreparedDeployment,
    *,
    agent_name: str,
) -> dict[str, Any]:
    output_path = prepared.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    resolved_project_path = output_path / "resolved_project_config.yaml"
    resolved_project_path.write_text(
        yaml.safe_dump(prepared.project_config, sort_keys=False),
        encoding="utf-8",
    )

    runner_path = output_path / "run.py"
    runner_path.write_text(
        _render_local_runner_script(
            agent_name=agent_name,
            mode=prepared.mode,
            project_root=prepared.project_root,
        ),
        encoding="utf-8",
    )

    env_example_path = output_path / ".env.example"
    env_example_path.write_text(
        _render_env_example(prepared.env_vars, target_label="local"),
        encoding="utf-8",
    )

    next_command = f"python {runner_path.as_posix()}"
    bundle_readme_path = output_path / "README.md"
    bundle_readme_path.write_text(
        _render_local_bundle_readme(
            agent_name=agent_name,
            agent_kind=str(prepared.agent_config.get("kind") or ""),
            mode=prepared.mode,
            execution_backend=prepared.execution_backend,
            broker_provider=prepared.broker_provider,
            next_command=next_command,
            env_vars=prepared.env_vars,
            safety_requirements=prepared.safety_requirements,
        ),
        encoding="utf-8",
    )

    artifacts = {
        "runner": runner_path.as_posix(),
        "env_example": env_example_path.as_posix(),
        "readme": bundle_readme_path.as_posix(),
        "resolved_project_config": resolved_project_path.as_posix(),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "agent_kind": prepared.agent_config.get("kind"),
        "target": prepared.target,
        "mode": prepared.mode,
        "execution_backend": prepared.execution_backend,
        "broker_provider": prepared.broker_provider,
        "service_name": prepared.service_name,
        "project_root": prepared.project_root.as_posix(),
        "source_config": prepared.config_path.as_posix(),
        "command": ["python", runner_path.as_posix()],
        "agent_command": [
            "python",
            "-m",
            "quanttradeai.cli",
            "agent",
            "run",
            "--agent",
            agent_name,
            "-c",
            resolved_project_path.as_posix(),
            "--mode",
            prepared.mode,
        ],
        "safety_requirements": prepared.safety_requirements,
        "environment_variables": [name for name, _ in prepared.env_vars],
        "artifacts": artifacts,
        "warnings": prepared.warnings,
        "next_command": next_command,
    }
    return _write_manifest_and_result(
        prepared=prepared,
        manifest=manifest,
        artifacts=artifacts,
        next_command=next_command,
    )


def _write_manifest_and_result(
    *,
    prepared: PreparedDeployment,
    manifest: dict[str, Any],
    artifacts: dict[str, str],
    next_command: str,
) -> dict[str, Any]:
    manifest_path = prepared.output_path / "deployment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )

    return {
        "status": "success",
        "agent_name": manifest["agent_name"],
        "target": prepared.target,
        "mode": prepared.mode,
        "output_dir": prepared.output_path.as_posix(),
        "artifacts": {
            **artifacts,
            "manifest": manifest_path.as_posix(),
        },
        "warnings": prepared.warnings,
        "next_command": next_command,
    }


def deploy_project_agent(
    *,
    agent_name: str,
    config_path: Path | str = "config/project.yaml",
    target: str | None = None,
    mode: str | None = None,
    output_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate a deployment bundle for a project-defined agent."""

    prepared = _prepare_deployment(
        agent_name=agent_name,
        config_path=config_path,
        target=target,
        mode=mode,
        output_dir=output_dir,
        force=force,
    )

    if prepared.target == "docker-compose":
        return _write_docker_compose_bundle(prepared, agent_name=agent_name)
    if prepared.target == "local":
        return _write_local_bundle(prepared, agent_name=agent_name)

    supported = ", ".join(sorted(SUPPORTED_DEPLOY_TARGETS))
    raise ValueError(
        f"Unsupported deployment target '{prepared.target}'. Supported targets: {supported}."
    )
