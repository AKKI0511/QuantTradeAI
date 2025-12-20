import json
from pathlib import Path

from typer.testing import CliRunner

from quanttradeai.cli import app
from quanttradeai.utils.config_validator import (
    DEFAULT_CONFIG_PATHS,
    validate_all,
)


def test_validate_all_passes_default_configs(tmp_path: Path):
    report_dir = tmp_path / "reports"
    summary = validate_all(output_dir=report_dir)

    assert summary["all_passed"] is True
    assert Path(summary["report_paths"]["json"]).is_file()
    assert Path(summary["report_paths"]["csv"]).is_file()
    # ensure each known config was evaluated
    for name in DEFAULT_CONFIG_PATHS:
        assert name in summary["results"]
        assert summary["results"][name]["passed"] is True


def test_validate_all_handles_missing_file(tmp_path: Path):
    missing_path = tmp_path / "absent.yaml"
    summary = validate_all({"model_config": missing_path}, output_dir=tmp_path / "out")

    assert summary["all_passed"] is False
    model_result = summary["results"]["model_config"]
    assert model_result["passed"] is False
    assert "not found" in model_result["error"]


def test_cli_validate_config_success(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app, ["validate-config", "--output-dir", str(tmp_path / "reports")]
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["all_passed"] is True
    for path in payload["report_paths"].values():
        assert Path(path).is_file()


def test_cli_validate_config_failure(tmp_path: Path):
    runner = CliRunner()
    bad_model_path = tmp_path / "missing.yaml"
    result = runner.invoke(
        app,
        [
            "validate-config",
            "--model-config",
            str(bad_model_path),
            "--output-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["all_passed"] is False
    assert payload["results"]["model_config"]["passed"] is False
