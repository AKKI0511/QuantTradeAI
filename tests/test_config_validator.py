from pathlib import Path

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
