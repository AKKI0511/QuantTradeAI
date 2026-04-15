import json
from pathlib import Path

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


def test_rule_agent_deploy_writes_bundle_and_preserves_project_config(
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
    compose_text = (output_dir / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")

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


def test_llm_agent_deploy_includes_llm_and_streaming_env_vars(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "llm-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="breakout_gpt",
        config_path=config_path,
        output_dir="deployments/llm",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    compose_text = (output_dir / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    manifest = json.loads(
        (output_dir / "deployment_manifest.json").read_text(encoding="utf-8")
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert "OPENAI_API_KEY" in env_example
    assert "ALPACA_API_KEY" in env_example
    assert "ALPACA_API_SECRET" in env_example
    assert ":/app/prompts:ro" in compose_text
    assert manifest["agent_name"] == "breakout_gpt"
    assert manifest["mode"] == "paper"
    assert manifest["target"] == "docker-compose"
    assert manifest["command"][-1] == "paper"


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
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")

    assert "FOO_BAR_API_KEY" in env_example
    assert "FOO_BAR_API_SECRET" in env_example
    assert "FOO-BAR_API_KEY" not in env_example


def test_model_agent_deploy_writes_manifest_and_models_mount(
    tmp_path: Path, monkeypatch
):
    config_path = _init_template(tmp_path, monkeypatch, "model-agent")
    original_config = config_path.read_text(encoding="utf-8")

    result = _deploy_agent(
        agent_name="paper_momentum",
        config_path=config_path,
        output_dir="deployments/model",
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    output_dir = Path(payload["output_dir"])
    compose_text = (output_dir / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (output_dir / ".env.example").read_text(encoding="utf-8")
    manifest = json.loads(
        (output_dir / "deployment_manifest.json").read_text(encoding="utf-8")
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    assert ":/app/models" in compose_text
    assert "OPENAI_API_KEY" not in env_example
    assert "ALPACA_API_KEY" in env_example
    assert payload["artifacts"]["resolved_project_config"].endswith(
        "resolved_project_config.yaml"
    )
    assert payload["artifacts"]["manifest"].endswith("deployment_manifest.json")
    assert manifest["agent_kind"] == "model"


def test_deploy_disables_replay_in_generated_bundle_and_warns(
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
    resolved_project = yaml.safe_load(
        (output_dir / "resolved_project_config.yaml").read_text(encoding="utf-8")
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
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["data"]["streaming"].pop("provider")
    config_payload["data"]["streaming"].pop("websocket_url")
    config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )

    result = _deploy_agent(
        agent_name="rsi_reversion",
        config_path=config_path,
        output_dir="deployments/rule-missing-realtime",
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "data.streaming.provider must be configured" in combined_output


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
                "--mode",
                "live",
                "--output",
                "deployments/live",
            ],
            "Only deploy --mode paper is supported",
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
