import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from quanttradeai.cli import PROJECT_TEMPLATES, app


runner = CliRunner()


def test_init_creates_each_template(tmp_path: Path):
    for template_name, expected in PROJECT_TEMPLATES.items():
        output = tmp_path / f"{template_name}.yaml"
        result = runner.invoke(
            app,
            ["init", "--template", template_name, "--output", str(output)],
        )

        assert result.exit_code == 0, result.stdout
        assert output.is_file()
        actual = yaml.safe_load(output.read_text(encoding="utf-8"))
        assert actual["project"]["name"] == expected["project"]["name"]
        assert set(actual) >= {
            "project",
            "profiles",
            "data",
            "features",
            "research",
            "agents",
            "deployment",
        }


def test_init_refuses_existing_without_force_and_overwrites_with_force(tmp_path: Path):
    output = tmp_path / "config" / "project.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("project:\n  name: old\n", encoding="utf-8")

    fail_result = runner.invoke(
        app,
        ["init", "--template", "research", "--output", str(output)],
    )
    assert fail_result.exit_code == 1

    pass_result = runner.invoke(
        app,
        ["init", "--template", "research", "--output", str(output), "--force"],
    )
    assert pass_result.exit_code == 0

    rendered = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert rendered["project"]["name"] == PROJECT_TEMPLATES["research"]["project"]["name"]


def test_validate_passes_for_generated_templates(tmp_path: Path):
    for template_name in PROJECT_TEMPLATES:
        cfg_path = tmp_path / template_name / "project.yaml"
        init_result = runner.invoke(
            app,
            ["init", "--template", template_name, "--output", str(cfg_path)],
        )
        assert init_result.exit_code == 0, init_result.stdout

        result = runner.invoke(app, ["validate", "--config", str(cfg_path)])
        assert result.exit_code == 0, result.stdout
        assert "Resolved project config summary:" in result.stdout


def test_validate_fails_missing_required_sections(tmp_path: Path):
    cfg_path = tmp_path / "broken_project.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"project": {"name": "demo", "profile": "paper"}}, sort_keys=False),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "--config", str(cfg_path)])

    assert result.exit_code == 1
    assert "missing required section(s)" in result.stderr.lower()


def test_validate_writes_resolved_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["validate", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout[result.stdout.index("{"):])
    resolved_path = Path(payload["artifacts"]["resolved_config"])
    summary_path = Path(payload["artifacts"]["summary"])

    assert resolved_path.is_file()
    assert summary_path.is_file()
    assert resolved_path.parent.parent.name == "config_validation"


def test_validate_preserves_unknown_fields_in_resolved_artifact(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    config_payload = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    config_payload["risk"] = {"enabled": True, "max_drawdown": 0.15}
    config_payload["data"]["streaming"] = {
        "enabled": True,
        "provider": "paper-feed",
    }

    cfg_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["validate", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout[result.stdout.index("{") :])
    resolved_path = Path(payload["artifacts"]["resolved_config"])
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))

    assert resolved["risk"] == {"enabled": True, "max_drawdown": 0.15}
    assert resolved["data"]["streaming"] == {
        "enabled": True,
        "provider": "paper-feed",
    }


def test_research_run_happy_path_writes_run_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")

    init_result = runner.invoke(
        app,
        ["init", "--template", "research", "--output", str(cfg_path)],
    )
    assert init_result.exit_code == 0, init_result.stdout

    def _fake_pipeline(*args, **kwargs):
        experiment_dir = Path("models/experiments/20260101_000000")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "results.json").write_text("{}", encoding="utf-8")
        (experiment_dir / "test_window_coverage.json").write_text("{}", encoding="utf-8")
        return {
            "results": {"AAPL": {"test_metrics": {"accuracy": 0.75}}},
            "coverage": {"path": str(experiment_dir / "test_window_coverage.json"), "fallback_symbols": []},
            "experiment_dir": str(experiment_dir),
        }

    monkeypatch.setattr("quanttradeai.cli.run_pipeline", _fake_pipeline)

    run_result = runner.invoke(app, ["research", "run", "--config", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.stdout
    assert "Research run completed:" in run_result.stdout

    run_root = Path("runs")
    run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    assert run_dirs
    latest_run = run_dirs[-1]

    summary_path = latest_run / "summary.json"
    metrics_path = latest_run / "metrics.json"
    assert summary_path.is_file()
    assert metrics_path.is_file()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert summary_payload["status"] == "success"
    assert summary_payload["project_name"] == PROJECT_TEMPLATES["research"]["project"]["name"]
    assert Path(summary_payload["artifacts"]["resolved_project_config"]).is_file()
    assert metrics_payload["status"] == "available"


def test_research_run_fails_for_malformed_project_config(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump({"project": {"name": "broken", "profile": "research"}}, sort_keys=False),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["research", "run", "--config", str(cfg_path)])

    assert result.exit_code == 1
    assert "research run failed" in result.stderr.lower()

    run_dirs = sorted(path for path in Path("runs").iterdir() if path.is_dir())
    assert run_dirs
    latest_run = run_dirs[-1]
    summary_payload = json.loads((latest_run / "summary.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((latest_run / "metrics.json").read_text(encoding="utf-8"))

    assert summary_payload["status"] == "failed"
    assert metrics_payload["status"] == "placeholder"
