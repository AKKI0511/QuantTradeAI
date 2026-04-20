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
    _compose_text, _env_example, readme, manifest, _resolved_project = _load_deploy_bundle(
        output_dir
    )

    assert manifest["execution_backend"] == "alpaca"
    assert manifest["broker_provider"] == "alpaca"
    assert "broker-backed execution" in readme
    assert "submit real `alpaca` market orders" in readme


@pytest.mark.parametrize(
    ("template", "agent_name", "expect_openai_env", "expect_prompt_mount", "expect_models_mount"),
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
    compose_text, env_example, readme, manifest, _resolved_project = _load_deploy_bundle(
        output_dir
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
    _compose_text, env_example, _readme, _manifest, _resolved_project = _load_deploy_bundle(
        output_dir
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
    _compose_text, _env_example, _readme, _manifest, resolved_project = _load_deploy_bundle(
        output_dir
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


def test_live_deploy_requires_agent_to_be_configured_live(
    tmp_path: Path, monkeypatch
):
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
        assert config_path.read_text(encoding="utf-8") == original_config
        if requested_output == existing_output:
            assert (requested_output / "stale.txt").is_file()
        else:
            assert not requested_output.exists()
