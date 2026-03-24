import json
from pathlib import Path

import pytest

from quanttradeai.agents.model_agent import _initialize_model_agent_run


def test_initialize_model_agent_run_writes_failure_summary_for_config_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception, match="not found|No such file"):
        _initialize_model_agent_run(
            project_config_path="config/missing-project.yaml",
            agent_name="momentum",
            mode="backtest",
        )

    run_dirs = sorted((tmp_path / "runs" / "agent" / "backtest").glob("*"))
    assert run_dirs
    summary_payload = json.loads((run_dirs[-1] / "summary.json").read_text("utf-8"))

    assert summary_payload["status"] == "failed"
    assert summary_payload["agent_name"] == "momentum"
    assert "error" in summary_payload
    assert summary_payload["timestamps"].get("completed_at")
