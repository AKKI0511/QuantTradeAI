import json

import pytest

from quanttradeai.agents.model_agent import _initialize_model_agent_run, run_model_agent_backtest


def test_model_agent_backtest_writes_failure_summary_for_init_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception, match="not found|No such file"):
        run_model_agent_backtest(
            project_config_path="config/missing-project.yaml",
            agent_name="momentum",
        )

    run_dirs = sorted((tmp_path / "runs" / "agent" / "backtest").glob("*"))
    assert run_dirs
    summary_payload = json.loads((run_dirs[-1] / "summary.json").read_text("utf-8"))

    assert summary_payload["status"] == "failed"
    assert summary_payload["agent_name"] == "momentum"
    assert "error" in summary_payload
    assert summary_payload["timestamps"].get("completed_at")


def test_initialize_model_agent_run_rejects_live_when_agent_mode_is_not_live(tmp_path, monkeypatch):
    resolved_config_path = tmp_path / "resolved_project_config.yaml"
    resolved_config_path.write_text(
        """
agents:
  - name: momentum
    kind: model
    mode: paper
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quanttradeai.agents.model_agent.validate_project_config",
        lambda **_: {
            "artifacts": {"resolved_config": str(resolved_config_path)},
            "warnings": [],
        },
    )

    with pytest.raises(ValueError, match="must be configured with mode=live"):
        _initialize_model_agent_run(
            run_dir=tmp_path,
            summary={"artifacts": {}},
            project_config_path="config/project.yaml",
            agent_name="momentum",
            mode="live",
        )
