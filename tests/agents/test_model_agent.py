import json

import pytest

from quanttradeai.agents.model_agent import run_model_agent_backtest


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
