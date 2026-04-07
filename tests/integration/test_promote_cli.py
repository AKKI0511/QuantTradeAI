import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from quanttradeai.cli import PROJECT_TEMPLATES, app


runner = CliRunner()


def _write_project_config(
    path: Path,
    *,
    agent_mode: str = "backtest",
    deployment_mode: str = "backtest",
    streaming_enabled: bool = True,
) -> dict:
    payload = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["llm-agent"], sort_keys=False)
    )
    payload["agents"][0]["mode"] = agent_mode
    payload["deployment"]["mode"] = deployment_mode
    payload["data"]["streaming"]["enabled"] = streaming_enabled
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def _write_run_summary(
    *,
    run_id: str = "agent/backtest/20260101_010000_breakout_gpt",
    run_type: str = "agent",
    mode: str = "backtest",
    status: str = "success",
    agent_name: str | None = "breakout_gpt",
) -> Path:
    run_dir = Path("runs").joinpath(*run_id.split("/"))
    summary = {
        "run_type": run_type,
        "mode": mode,
        "name": agent_name or "research_lab",
        "status": status,
        "symbols": ["AAPL"],
        "timestamps": {
            "started_at": "2026-01-01T01:00:00+00:00",
            "completed_at": "2026-01-01T01:05:00+00:00",
        },
        "artifacts": {},
        "warnings": [],
        "run_id": run_id,
        "run_dir": str(run_dir),
    }
    if agent_name is not None:
        summary["agent_name"] = agent_name
    if run_type == "research":
        summary["project_name"] = "research_lab"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return run_dir


def _load_project_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_promote_agent_backtest_updates_project_yaml_and_prints_next_command(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    _write_project_config(config_path)
    run_id = "agent/backtest/20260101_010000_breakout_gpt"
    _write_run_summary(run_id=run_id)

    run_arg = "runs\\" + run_id.replace("/", "\\")
    result = runner.invoke(
        app,
        ["promote", "--run", run_arg, "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    updated = _load_project_config(config_path)

    assert payload["status"] == "success"
    assert payload["source_run_id"] == run_id
    assert payload["agent_name"] == "breakout_gpt"
    assert payload["from_mode"] == "backtest"
    assert payload["to_mode"] == "paper"
    assert payload["changed"] is True
    assert payload["dry_run"] is False
    assert (
        payload["next_command"]
        == "quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper"
    )
    assert updated["agents"][0]["mode"] == "paper"
    assert updated["deployment"]["mode"] == "paper"


def test_promote_dry_run_reports_change_without_writing(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    _write_project_config(config_path)
    original = config_path.read_text(encoding="utf-8")
    _write_run_summary()

    result = runner.invoke(
        app,
        [
            "promote",
            "--run",
            "agent/backtest/20260101_010000_breakout_gpt",
            "--config",
            str(config_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["changed"] is True
    assert payload["dry_run"] is True
    assert config_path.read_text(encoding="utf-8") == original


def test_promote_already_paper_agent_is_success_with_no_change(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    _write_project_config(
        config_path,
        agent_mode="paper",
        deployment_mode="paper",
    )
    _write_run_summary()

    result = runner.invoke(
        app,
        [
            "promote",
            "--run",
            "agent/backtest/20260101_010000_breakout_gpt",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["changed"] is False


def test_promote_failure_cases_do_not_mutate_project_yaml(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    _write_project_config(config_path)
    original = config_path.read_text(encoding="utf-8")
    _write_run_summary(
        run_id="agent/backtest/20260101_020000_failed_agent",
        status="failed",
    )
    _write_run_summary(
        run_id="research/20260101_000000_research_lab",
        run_type="research",
        mode="research",
        agent_name=None,
    )

    cases = [
        (
            [
                "promote",
                "--run",
                "agent/backtest/missing",
                "--config",
                str(config_path),
            ],
            "Run not found",
        ),
        (
            [
                "promote",
                "--run",
                "agent/backtest/20260101_020000_failed_agent",
                "--config",
                str(config_path),
            ],
            "Only successful agent backtest runs",
        ),
        (
            [
                "promote",
                "--run",
                "research/20260101_000000_research_lab",
                "--config",
                str(config_path),
            ],
            "Only successful agent backtest runs",
        ),
        (
            [
                "promote",
                "--run",
                "agent/backtest/20260101_020000_failed_agent",
                "--config",
                str(config_path),
                "--to",
                "live",
            ],
            "Only promotion to paper is supported",
        ),
    ]

    for command, expected in cases:
        result = runner.invoke(app, command)
        assert result.exit_code == 1
        combined_output = f"{result.stdout}\n{result.stderr}"
        assert expected in combined_output
        assert config_path.read_text(encoding="utf-8") == original


def test_promote_requires_streaming_before_writing_paper_config(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    _write_project_config(config_path, streaming_enabled=False)
    original = config_path.read_text(encoding="utf-8")
    _write_run_summary()

    result = runner.invoke(
        app,
        [
            "promote",
            "--run",
            "agent/backtest/20260101_010000_breakout_gpt",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 1
    combined_output = f"{result.stdout}\n{result.stderr}"
    assert "data.streaming.enabled must be true" in combined_output
    assert config_path.read_text(encoding="utf-8") == original
