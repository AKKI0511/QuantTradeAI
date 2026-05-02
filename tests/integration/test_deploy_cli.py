import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quanttradeai.cli import app


runner = CliRunner()


def _init_template(tmp_path: Path, monkeypatch, template: str) -> Path:
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    result = runner.invoke(
        app,
        ["init", "--template", template, "--output", str(config_path)],
    )
    assert result.exit_code == 0, result.stdout
    return config_path


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _configure_live_deploy(
    config_path: Path,
    *,
    deployment_mode: str = "live",
) -> dict:
    payload = _read_yaml(config_path)
    payload["agents"][0]["mode"] = "live"
    payload["deployment"]["mode"] = deployment_mode
    _write_yaml(config_path, payload)
    return payload


def _deploy_agent(
    *,
    agent_name: str,
    config_path: Path,
    output_dir: str,
    extra_args: list[str] | None = None,
):
    args = [
        "deploy",
        "--agent",
        agent_name,
        "--config",
        str(config_path),
        "--output",
        output_dir,
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(app, args)


def _load_deploy_bundle(output_dir: Path) -> tuple[str, str, str, dict, dict]:
    compose_text = (output_dir / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (output_dir / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    resolved_project = _read_yaml(output_dir / "resolved_project_config.yaml")
    return compose_text, env_example, readme, manifest, resolved_project


def _load_local_deploy_bundle(output_dir: Path) -> tuple[str, str, str, dict, dict]:
    runner_text = (output_dir / "run.py").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (output_dir / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    resolved_project = _read_yaml(output_dir / "resolved_project_config.yaml")
    return runner_text, env_example, readme, manifest, resolved_project


def _load_render_deploy_bundle(
    output_dir: Path,
) -> tuple[dict, str, str, str, dict, dict]:
    render_yaml = _read_yaml(output_dir / "render.yaml")
    dockerfile_text = (output_dir / "Dockerfile").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (output_dir / "deployment_manifest.json").read_text(encoding="utf-8")
    )
    resolved_project = _read_yaml(output_dir / "resolved_project_config.yaml")
    return render_yaml, dockerfile_text, env_example, readme, manifest, resolved_project


def test_rule_agent_deploy_writes_paper_bundle_and_preserves_project_config(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    compose_text, env_example, readme, manifest, resolved_project = _load_deploy_bundle(
        output_dir
    )

    assert payload["status"] == "success"
    assert payload["target"] == "docker-compose"
    assert payload["mode"] == "paper"
    assert config_path.read_text(encoding="utf-8") == original_config
    assert (output_dir / "docker-compose.yml").is_file()
    assert (output_dir / "Dockerfile").is_file()
    assert (output_dir / ".env.example").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "resolved_project_config.yaml").is_file()
    assert (output_dir / "deployment_manifest.json").is_file()
    assert "--agent" in compose_text
    assert "rsi_reversion" in compose_text
    assert "--mode" in compose_text
    assert "paper" in compose_text
    assert "ALPACA_API_KEY" in env_example
    assert "ALPACA_API_SECRET" in env_example
    assert "OPENAI_API_KEY" not in env_example
    assert "in `paper` mode" in readme
    assert manifest["agent_name"] == "rsi_reversion"
    assert manifest["mode"] == "paper"
    assert manifest["target"] == "docker-compose"
    assert manifest["command"][-1] == "paper"
    assert manifest["safety_requirements"] == []
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is False


def test_rule_agent_local_deploy_writes_paper_bundle_and_preserves_project_config(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-local",
        extra_args=["--target", "local"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    runner_text, env_example, readme, manifest, resolved_project = (
        _load_local_deploy_bundle(output_dir)
    )

    assert payload["status"] == "success"
    assert payload["target"] == "local"
    assert payload["mode"] == "paper"
    assert payload["next_command"] == f"python {(output_dir / 'run.py').as_posix()}"
    assert payload["artifacts"]["runner"].endswith("/run.py")
    assert "compose" not in payload["artifacts"]
    assert "dockerfile" not in payload["artifacts"]
    assert config_path.read_text(encoding="utf-8") == original_config
    assert (output_dir / "run.py").is_file()
    assert not (output_dir / "docker-compose.yml").exists()
    assert not (output_dir / "Dockerfile").exists()
    assert "quanttradeai.cli" in runner_text
    assert "resolved_project_config.yaml" in runner_text
    assert "ALPACA_API_KEY" in env_example
    assert "ALPACA_API_SECRET" in env_example
    assert "docker compose" not in readme.lower()
    assert "local machine" in readme
    assert manifest["agent_name"] == "rsi_reversion"
    assert manifest["mode"] == "paper"
    assert manifest["target"] == "local"
    assert manifest["command"] == ["python", (output_dir / "run.py").as_posix()]
    assert manifest["safety_requirements"] == []
    assert "volumes" not in manifest
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is False


def test_local_deploy_can_use_project_config_default_target(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _read_yaml(config_path)
    payload["deployment"]["target"] = "local"
    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-local-default",
    )

    assert result.exit_code == 0, result.stdout
    deploy_payload = json.loads(result.stdout)
    output_dir = Path(deploy_payload["output_dir"])
    _runner_text, _env_example, _readme, manifest, _resolved_project = (
        _load_local_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert deploy_payload["target"] == "local"
    assert manifest["target"] == "local"
    assert (output_dir / "run.py").is_file()
    assert not (output_dir / "docker-compose.yml").exists()


def test_rule_agent_render_deploy_writes_paper_worker_bundle_and_preserves_project_config(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-render",
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    render_yaml, dockerfile_text, env_example, readme, manifest, resolved_project = (
        _load_render_deploy_bundle(output_dir)
    )

    service = render_yaml["services"][0]
    assert payload["status"] == "success"
    assert payload["target"] == "render"
    assert payload["mode"] == "paper"
    assert config_path.read_text(encoding="utf-8") == original_config
    assert (output_dir / "render.yaml").is_file()
    assert (output_dir / "Dockerfile").is_file()
    assert (output_dir / ".env.example").is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "resolved_project_config.yaml").is_file()
    assert (output_dir / "deployment_manifest.json").is_file()
    assert (output_dir / "assets").is_dir()
    assert service["type"] == "worker"
    assert service["runtime"] == "docker"
    assert service["dockerContext"] == "."
    assert service["dockerfilePath"] == "deployments/rule-render/Dockerfile"
    assert service["dockerCommand"] == (
        "quanttradeai agent run --agent rsi_reversion "
        "-c config/project.yaml --mode paper"
    )
    assert service["numInstances"] == 1
    assert service["maxShutdownDelaySeconds"] == 60
    assert service["disk"] == {"name": "runs", "mountPath": "/app/runs", "sizeGB": 1}
    assert {"key": "ALPACA_API_KEY", "sync": False} in service["envVars"]
    assert {"key": "ALPACA_API_SECRET", "sync": False} in service["envVars"]
    assert "deployments/rule-render/resolved_project_config.yaml" in dockerfile_text
    assert "deployments/rule-render/assets/" in dockerfile_text
    assert "ALPACA_API_KEY" in env_example
    assert "Render Background Worker" in readme
    assert manifest["target"] == "render"
    assert manifest["platform"] == "render"
    assert manifest["service_type"] == "worker"
    assert manifest["docker_context"] == "."
    assert manifest["dockerfile_path"] == "deployments/rule-render/Dockerfile"
    assert manifest["render_blueprint"].endswith("/render.yaml")
    assert manifest["asset_manifest"] == []
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is False
    assert any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )


def test_render_deploy_can_use_project_config_default_target(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _read_yaml(config_path)
    payload["deployment"]["target"] = "render"
    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-render-default",
    )

    assert result.exit_code == 0, result.stdout
    deploy_payload = json.loads(result.stdout)
    output_dir = Path(deploy_payload["output_dir"])
    (
        render_yaml,
        _dockerfile_text,
        _env_example,
        _readme,
        manifest,
        _resolved_project,
    ) = _load_render_deploy_bundle(output_dir)

    assert config_path.read_text(encoding="utf-8") == original_config
    assert deploy_payload["target"] == "render"
    assert manifest["target"] == "render"
    assert render_yaml["services"][0]["type"] == "worker"
    assert (output_dir / "render.yaml").is_file()
    assert not (output_dir / "docker-compose.yml").exists()


def test_rule_agent_live_deploy_uses_config_default_mode_and_preserves_project_config(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    _configure_live_deploy(config_path, deployment_mode="live")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-live",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    compose_text, env_example, readme, manifest, resolved_project = _load_deploy_bundle(
        output_dir
    )

    assert payload["status"] == "success"
    assert payload["mode"] == "live"
    assert config_path.read_text(encoding="utf-8") == original_config
    assert manifest["mode"] == "live"
    assert manifest["command"][-1] == "live"
    assert manifest["safety_requirements"]
    assert "--mode" in compose_text
    assert "live" in compose_text
    assert ":/app/data" in compose_text
    assert ":/app/reports" in compose_text
    assert ":/app/runs" in compose_text
    assert "ALPACA_API_KEY" in env_example
    assert "ALPACA_API_SECRET" in env_example
    assert "in `live` mode" in readme
    assert "Safety Requirements" in readme
    assert "`mode: live`" in readme
    assert not any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is True


def test_rule_agent_render_live_deploy_uses_worker_safety_gates(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    _configure_live_deploy(config_path, deployment_mode="live")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-render-live",
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    render_yaml, _dockerfile_text, env_example, readme, manifest, resolved_project = (
        _load_render_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert payload["status"] == "success"
    assert payload["target"] == "render"
    assert payload["mode"] == "live"
    assert render_yaml["services"][0]["dockerCommand"].endswith("--mode live")
    assert manifest["mode"] == "live"
    assert manifest["service_type"] == "worker"
    assert manifest["safety_requirements"]
    assert "ALPACA_API_KEY" in env_example
    assert "Safety Requirements" in readme
    assert "`mode: live`" in readme
    assert not any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is True


def test_hybrid_agent_local_live_deploy_absolutizes_prompt_and_model_paths(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "hybrid")
    _configure_live_deploy(config_path, deployment_mode="paper")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="hybrid_swing_agent",
        config_path=config_path,
        output_dir="deployments/hybrid-local-live",
        extra_args=["--target", "local", "--mode", "live"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    runner_text, env_example, readme, manifest, resolved_project = (
        _load_local_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert payload["status"] == "success"
    assert payload["target"] == "local"
    assert payload["mode"] == "live"
    assert manifest["agent_name"] == "hybrid_swing_agent"
    assert manifest["mode"] == "live"
    assert manifest["target"] == "local"
    assert manifest["safety_requirements"]
    assert "OPENAI_API_KEY" in env_example
    assert "ALPACA_API_KEY" in env_example
    assert "quanttradeai.cli" in runner_text
    assert "in `live` mode" in readme
    assert "Safety Requirements" in readme
    agent_config = resolved_project["agents"][0]
    assert Path(agent_config["llm"]["prompt_file"]).is_absolute()
    assert Path(agent_config["model_signal_sources"][0]["path"]).is_absolute()
    assert not (output_dir / "docker-compose.yml").exists()
    assert not any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )


def test_llm_agent_render_deploy_copies_prompt_assets(tmp_path: Path, monkeypatch):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="breakout_gpt",
        config_path=config_path,
        output_dir="deployments/llm-render",
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    render_yaml, _dockerfile_text, env_example, readme, manifest, resolved_project = (
        _load_render_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert (output_dir / "assets" / "prompts" / "breakout.md").is_file()
    assert "OPENAI_API_KEY" in env_example
    assert {"key": "OPENAI_API_KEY", "sync": False} in render_yaml["services"][0][
        "envVars"
    ]
    assert "prompts/breakout.md" in readme
    assert manifest["asset_manifest"] == [
        {
            "type": "prompt",
            "source": "prompts/breakout.md",
            "bundle_path": "assets/prompts/breakout.md",
            "image_path": "/app/prompts/breakout.md",
        }
    ]
    assert resolved_project["agents"][0]["llm"]["prompt_file"] == "prompts/breakout.md"


@pytest.mark.parametrize(
    (
        "template",
        "agent_name",
        "expected_assets",
        "expect_openai_env",
    ),
    [
        (
            "model-agent",
            "paper_momentum",
            ["models/promoted/aapl_daily_classifier"],
            False,
        ),
        (
            "hybrid",
            "hybrid_swing_agent",
            ["prompts/hybrid_swing.md", "models/promoted/aapl_daily_classifier"],
            True,
        ),
    ],
)
def test_render_deploy_copies_model_and_hybrid_assets(
    tmp_path: Path,
    monkeypatch,
    template: str,
    agent_name: str,
    expected_assets: list[str],
    expect_openai_env: bool,
):
    config_path = _init_template(tmp_path, monkeypatch, template)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name=agent_name,
        config_path=config_path,
        output_dir=f"deployments/{template}-render",
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    render_yaml, _dockerfile_text, env_example, readme, manifest, _resolved_project = (
        _load_render_deploy_bundle(output_dir)
    )
    asset_sources = [item["source"] for item in manifest["asset_manifest"]]

    assert config_path.read_text(encoding="utf-8") == original_config
    for expected_asset in expected_assets:
        assert expected_asset in asset_sources
        assert (output_dir / "assets" / expected_asset).exists()
        assert expected_asset in readme
    assert ("OPENAI_API_KEY" in env_example) is expect_openai_env
    assert {"key": "ALPACA_API_KEY", "sync": False} in render_yaml["services"][0][
        "envVars"
    ]


def test_deploy_readme_and_manifest_call_out_alpaca_backed_execution(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _read_yaml(config_path)
    payload["data"]["streaming"]["replay"]["enabled"] = False
    payload["agents"][0]["execution"] = {"backend": "alpaca"}
    _write_yaml(config_path, payload)

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-broker",
    )

    assert result.exit_code == 0, result.stdout
    deploy_payload = json.loads(result.stdout)
    output_dir = Path(deploy_payload["output_dir"])
    _compose_text, _env_example, readme, manifest, _resolved_project = (
        _load_deploy_bundle(output_dir)
    )

    assert manifest["execution_backend"] == "alpaca"
    assert manifest["broker_provider"] == "alpaca"
    assert "broker-backed execution" in readme
    assert "submit real `alpaca` market orders" in readme


@pytest.mark.parametrize(
    (
        "template",
        "agent_name",
        "expect_openai_env",
        "expect_prompt_mount",
        "expect_models_mount",
    ),
    [
        ("llm-agent", "breakout_gpt", True, True, False),
        ("model-agent", "paper_momentum", False, False, True),
        ("hybrid", "hybrid_swing_agent", True, True, True),
    ],
)
def test_live_deploy_supports_llm_model_and_hybrid_agents(
    tmp_path: Path,
    monkeypatch,
    template: str,
    agent_name: str,
    expect_openai_env: bool,
    expect_prompt_mount: bool,
    expect_models_mount: bool,
):
    config_path = _init_template(tmp_path, monkeypatch, template)
    _configure_live_deploy(config_path, deployment_mode="paper")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name=agent_name,
        config_path=config_path,
        output_dir=f"deployments/{template}-live",
        extra_args=["--mode", "live"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    compose_text, env_example, readme, manifest, _resolved_project = (
        _load_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert payload["status"] == "success"
    assert payload["mode"] == "live"
    assert manifest["agent_name"] == agent_name
    assert manifest["mode"] == "live"
    assert manifest["command"][-1] == "live"
    assert manifest["safety_requirements"]
    assert "ALPACA_API_KEY" in env_example
    assert "ALPACA_API_SECRET" in env_example
    assert ("OPENAI_API_KEY" in env_example) is expect_openai_env
    assert (":/app/prompts:ro" in compose_text) is expect_prompt_mount
    assert (":/app/models" in compose_text) is expect_models_mount
    assert "in `live` mode" in readme
    assert "Safety Requirements" in readme
    assert not any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )


def test_deploy_normalizes_streaming_provider_env_var_prefix(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    project_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        project_text.replace("provider: alpaca", "provider: foo-bar"),
        encoding="utf-8",
    )

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-hyphen",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    _compose_text, env_example, _readme, _manifest, _resolved_project = (
        _load_deploy_bundle(output_dir)
    )

    assert "FOO_BAR_API_KEY" in env_example
    assert "FOO_BAR_API_SECRET" in env_example
    assert "FOO-BAR_API_KEY" not in env_example


def test_deploy_disables_replay_in_generated_paper_bundle_and_warns(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-replay",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    _compose_text, _env_example, _readme, _manifest, resolved_project = (
        _load_deploy_bundle(output_dir)
    )

    assert config_path.read_text(encoding="utf-8") != ""
    assert resolved_project["data"]["streaming"]["replay"]["enabled"] is False
    assert any(
        "replay was disabled in the generated bundle" in warning.lower()
        for warning in payload["warnings"]
    )


def test_deploy_rejects_replay_only_project_without_realtime_streaming_fields(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    config_payload = _read_yaml(config_path)
    config_payload["data"]["streaming"].pop("provider")
    config_payload["data"]["streaming"].pop("websocket_url")
    _write_yaml(config_path, config_payload)

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-missing-realtime",
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "data.streaming.provider must be configured" in combined_output


def test_live_deploy_requires_agent_to_be_configured_live(tmp_path: Path, monkeypatch):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="breakout_gpt",
        config_path=config_path,
        output_dir="deployments/live-not-configured",
        extra_args=["--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must be configured with mode=live" in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config


def test_local_live_deploy_requires_agent_to_be_configured_live(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="breakout_gpt",
        config_path=config_path,
        output_dir="deployments/local-live-not-configured",
        extra_args=["--target", "local", "--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must be configured with mode=live" in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config
    assert not Path("deployments/local-live-not-configured").exists()


def test_render_live_deploy_requires_agent_to_be_configured_live(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="breakout_gpt",
        config_path=config_path,
        output_dir="deployments/render-live-not-configured",
        extra_args=["--target", "render", "--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "must be configured with mode=live" in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config
    assert not Path("deployments/render-live-not-configured").exists()


@pytest.mark.parametrize(
    ("field_name", "expected_error"),
    [
        ("provider", "data.streaming.provider is required"),
        ("websocket_url", "data.streaming.websocket_url is required"),
        ("channels", "data.streaming requires channels"),
    ],
)
def test_live_deploy_rejects_missing_realtime_streaming_fields(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    expected_error: str,
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _configure_live_deploy(config_path)
    payload["data"]["streaming"].pop(field_name)
    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir=f"deployments/live-missing-{field_name}",
        extra_args=["--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert expected_error in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config


@pytest.mark.parametrize(
    ("field_name", "expected_error"),
    [
        ("risk", "risk is required when an agent is configured with mode=live."),
        (
            "position_manager",
            "position_manager is required when an agent is configured with mode=live.",
        ),
    ],
)
def test_live_deploy_requires_top_level_live_safety_sections(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    expected_error: str,
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _configure_live_deploy(config_path)
    payload.pop(field_name)
    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir=f"deployments/live-missing-{field_name}",
        extra_args=["--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert expected_error in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config


@pytest.mark.parametrize(
    ("field_name", "expected_error"),
    [
        ("risk", "risk is required when an agent is configured with mode=live."),
        (
            "position_manager",
            "position_manager is required when an agent is configured with mode=live.",
        ),
    ],
)
def test_local_live_deploy_requires_top_level_live_safety_sections(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    expected_error: str,
):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    payload = _configure_live_deploy(config_path)
    payload.pop(field_name)
    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir=f"deployments/local-live-missing-{field_name}",
        extra_args=["--target", "local", "--mode", "live"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert expected_error in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config


def test_render_deploy_rejects_output_outside_project_root(tmp_path: Path, monkeypatch):
    config_path = _init_template(tmp_path, monkeypatch, "rule-agent")
    outside_output = tmp_path.parent / "outside-render-bundle"

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir=str(outside_output),
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert (
        "Render deployment output must resolve inside the project root"
        in combined_output
    )
    assert not outside_output.exists()


@pytest.mark.parametrize(
    ("template", "agent_name", "mutator", "expected_error"),
    [
        (
            "llm-agent",
            "breakout_gpt",
            "prompt",
            "agents.<name>.llm.prompt_file must resolve under prompts/",
        ),
        (
            "model-agent",
            "paper_momentum",
            "model",
            "agents.<name>.model.path must resolve under models/",
        ),
        (
            "llm-agent",
            "breakout_gpt",
            "notes",
            "agents.<name>.context.notes.file must resolve under notes/",
        ),
    ],
)
def test_render_deploy_rejects_assets_outside_expected_project_dirs(
    tmp_path: Path,
    monkeypatch,
    template: str,
    agent_name: str,
    mutator: str,
    expected_error: str,
):
    config_path = _init_template(tmp_path, monkeypatch, template)
    payload = _read_yaml(config_path)

    if mutator == "prompt":
        bad_prompt = Path("other_prompts") / "agent.md"
        bad_prompt.parent.mkdir(parents=True, exist_ok=True)
        bad_prompt.write_text("Bad prompt location.", encoding="utf-8")
        payload["agents"][0]["llm"]["prompt_file"] = bad_prompt.as_posix()
    elif mutator == "model":
        bad_model = Path("artifacts") / "model"
        bad_model.mkdir(parents=True, exist_ok=True)
        (bad_model / "README.md").write_text("Bad model location.", encoding="utf-8")
        payload["agents"][0]["model"]["path"] = bad_model.as_posix()
    elif mutator == "notes":
        bad_notes = Path("agent_notes") / "breakout.md"
        bad_notes.parent.mkdir(parents=True, exist_ok=True)
        bad_notes.write_text("Bad notes location.", encoding="utf-8")
        payload["agents"][0]["context"]["notes"] = {
            "enabled": True,
            "file": bad_notes.as_posix(),
        }

    _write_yaml(config_path, payload)
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name=agent_name,
        config_path=config_path,
        output_dir=f"deployments/render-bad-{mutator}",
        extra_args=["--target", "render"],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert expected_error in combined_output
    assert config_path.read_text(encoding="utf-8") == original_config
    assert not Path(f"deployments/render-bad-{mutator}").exists()


def test_deploy_failure_cases(
    tmp_path: Path,
    monkeypatch,
):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    existing_output = Path("deployments/existing")
    existing_output.mkdir(parents=True, exist_ok=True)
    (existing_output / "stale.txt").write_text("stale", encoding="utf-8")

    cases = [
        (
            [
                "deploy",
                "--agent",
                "missing_agent",
                "--config",
                str(config_path),
                "--output",
                "deployments/missing",
            ],
            "Agent 'missing_agent' not found",
        ),
        (
            [
                "deploy",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--target",
                "kubernetes",
                "--output",
                "deployments/k8s",
            ],
            "Unsupported deployment target",
        ),
        (
            [
                "deploy",
                "--agent",
                "breakout_gpt",
                "--config",
                str(config_path),
                "--output",
                str(existing_output),
            ],
            "Refusing to overwrite existing deployment bundle",
        ),
    ]

    original_config = config_path.read_text(encoding="utf-8")
    for args, expected_error in cases:
        output_index = args.index("--output") + 1
        requested_output = Path(args[output_index])
        result = runner.invoke(app, args)
        assert result.exit_code == 1
        combined_output = f"{result.stdout}\n{result.stderr}"
        assert expected_error in combined_output
        if expected_error == "Unsupported deployment target":
            assert "docker-compose" in combined_output
            assert "local" in combined_output
            assert "render" in combined_output
        assert config_path.read_text(encoding="utf-8") == original_config
        if requested_output == existing_output:
            assert (requested_output / "stale.txt").is_file()
        else:
            assert not requested_output.exists()
