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
