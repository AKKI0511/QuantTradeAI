import json
from pathlib import Path
import re
from unittest.mock import patch

import pandas as pd
import yaml
from typer.testing import CliRunner

from quanttradeai.cli import PROJECT_TEMPLATES, app


runner = CliRunner()


def _normalize_cli_output(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}"
    combined = re.sub(r"\x1b\[[0-9;]*m", "", combined)
    return " ".join(combined.lower().split())


def _write_project_with_all_agent_kinds(config_path: Path) -> None:
    project_config = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["hybrid"], sort_keys=False)
    )
    project_config["project"]["name"] = "multi_agent_lab"
    project_config["research"]["enabled"] = False
    project_config["agents"] = [
        yaml.safe_load(yaml.safe_dump(PROJECT_TEMPLATES["rule-agent"]["agents"][0])),
        yaml.safe_load(yaml.safe_dump(PROJECT_TEMPLATES["model-agent"]["agents"][0])),
        yaml.safe_load(yaml.safe_dump(PROJECT_TEMPLATES["llm-agent"]["agents"][0])),
        yaml.safe_load(yaml.safe_dump(PROJECT_TEMPLATES["hybrid"]["agents"][0])),
    ]
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(project_config, sort_keys=False),
        encoding="utf-8",
    )


def _seed_agent_assets(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "project.yaml"

    llm_init = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", str(config_path)],
    )
    assert llm_init.exit_code == 0, llm_init.stdout

    hybrid_init = runner.invoke(
        app,
        ["init", "--template", "hybrid", "--output", str(config_path), "--force"],
    )
    assert hybrid_init.exit_code == 0, hybrid_init.stdout

    _write_project_with_all_agent_kinds(config_path)
    return config_path


def _set_all_agents_live(config_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["project"]["name"] = "multi_agent_lab"
    for agent in payload["agents"]:
        agent["mode"] = "live"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return payload


def _seed_sweep_assets(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "project.yaml"

    init_result = runner.invoke(
        app,
        ["init", "--template", "rule-agent", "--output", str(config_path)],
    )
    assert init_result.exit_code == 0, init_result.stdout

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["sweeps"] = [
        {
            "name": "rsi_threshold_grid",
            "kind": "agent_backtest",
            "agent": "rsi_reversion",
            "parameters": [
                {"path": "rule.buy_below", "values": [25.0, 30.0]},
                {"path": "rule.sell_above", "values": [70.0, 75.0]},
            ],
        }
    ]
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_agent_run_requires_exactly_one_of_agent_or_all(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    neither = runner.invoke(app, ["agent", "run", "--config", str(config_path)])
    assert neither.exit_code == 2
    neither_output = _normalize_cli_output(neither.stdout, neither.stderr)
    assert "choose exactly one of" in neither_output
    assert "--sweep" in neither_output

    both = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "rsi_reversion",
            "--all",
            "--config",
            str(config_path),
        ],
    )
    assert both.exit_code == 2
    both_output = _normalize_cli_output(both.stdout, both.stderr)
    assert "choose exactly one of" in both_output
    assert "--sweep" in both_output


def test_agent_run_rejects_mixing_agent_and_sweep(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_sweep_assets(tmp_path)

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "rsi_reversion",
            "--sweep",
            "rsi_threshold_grid",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    combined = _normalize_cli_output(result.stdout, result.stderr)
    assert "choose exactly one of" in combined
    assert "--sweep" in combined


def test_agent_run_all_live_requires_matching_project_acknowledgement(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)
    _set_all_agents_live(config_path)

    missing = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--all",
            "--config",
            str(config_path),
            "--mode",
            "live",
        ],
    )

    assert missing.exit_code == 1
    missing_output = f"{missing.stdout}\n{missing.stderr}"
    assert "requires --acknowledge-live multi_agent_lab" in missing_output

    wrong = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--all",
            "--config",
            str(config_path),
            "--mode",
            "live",
            "--acknowledge-live",
            "wrong_project",
        ],
    )

    assert wrong.exit_code == 1
    wrong_output = f"{wrong.stdout}\n{wrong.stderr}"
    assert "requires --acknowledge-live multi_agent_lab" in wrong_output


def test_agent_run_rejects_acknowledge_live_outside_all_live(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--agent",
            "rsi_reversion",
            "--config",
            str(config_path),
            "--mode",
            "paper",
            "--acknowledge-live",
            "multi_agent_lab",
        ],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "--acknowledge-live is only supported with --all --mode live" in combined


def test_agent_run_all_live_rejects_skip_validation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)
    _set_all_agents_live(config_path)

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--all",
            "--config",
            str(config_path),
            "--mode",
            "live",
            "--acknowledge-live",
            "multi_agent_lab",
            "--skip-validation",
        ],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "--skip-validation is not supported for live agent batches" in combined


def test_agent_run_all_live_requires_every_agent_configured_live(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)
    payload = _set_all_agents_live(config_path)
    payload["agents"][1]["mode"] = "paper"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--all",
            "--config",
            str(config_path),
            "--mode",
            "live",
            "--acknowledge-live",
            "multi_agent_lab",
        ],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "All agents must be configured with mode=live" in combined
    assert "paper_momentum(paper)" in combined


def test_agent_run_sweep_rejects_non_backtest_modes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_sweep_assets(tmp_path)

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--sweep",
            "rsi_threshold_grid",
            "--config",
            str(config_path),
            "--mode",
            "paper",
        ],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "--sweep currently supports only --mode backtest" in combined


def test_agent_run_all_rejects_invalid_max_concurrency(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    result = runner.invoke(
        app,
        [
            "agent",
            "run",
            "--all",
            "--config",
            str(config_path),
            "--max-concurrency",
            "0",
        ],
    )

    assert result.exit_code == 2


def test_agent_run_all_errors_when_project_has_no_agents(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config" / "project.yaml"

    init_result = runner.invoke(
        app,
        ["init", "--template", "research", "--output", str(config_path)],
    )
    assert init_result.exit_code == 0, init_result.stdout

    result = runner.invoke(
        app,
        ["agent", "run", "--all", "--config", str(config_path)],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "defines no agents to run with --all" in combined


def test_agent_run_all_writes_batch_artifacts_and_sorts_scoreboard(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "backtest"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is None
        run_dir = (
            Path("runs")
            / "agent"
            / "backtest"
            / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        sharpe_by_agent = {
            "rsi_reversion": 0.8,
            "paper_momentum": 1.1,
            "breakout_gpt": 2.4,
            "hybrid_swing_agent": 1.7,
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "net_sharpe": sharpe_by_agent[agent_name],
                    "net_pnl": sharpe_by_agent[agent_name] / 10.0,
                    "net_mdd": -0.05,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/backtest/{run_dir.name}",
            "run_type": "agent",
            "mode": "backtest",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    batch_dir = Path(payload["run_dir"])

    assert payload["status"] == "success"
    assert payload["agent_count"] == 4
    assert (batch_dir / "batch_manifest.json").is_file()
    assert (batch_dir / "results.json").is_file()
    assert (batch_dir / "scoreboard.json").is_file()
    assert (batch_dir / "scoreboard.txt").is_file()
    assert (batch_dir / "summary.json").is_file()
    assert (batch_dir / "experiment_brief.json").is_file()
    assert (batch_dir / "experiment_brief.md").is_file()
    assert payload["artifacts"]["experiment_brief_json"] == str(
        batch_dir / "experiment_brief.json"
    )
    assert payload["artifacts"]["experiment_brief_md"] == str(
        batch_dir / "experiment_brief.md"
    )

    results_payload = json.loads((batch_dir / "results.json").read_text("utf-8"))
    assert [item["agent_name"] for item in results_payload["results"]] == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]

    scoreboard_payload = json.loads((batch_dir / "scoreboard.json").read_text("utf-8"))
    ordered_names = [record["name"] for record in scoreboard_payload["records"]]
    assert ordered_names == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]
    assert "NET_SHARPE" in (batch_dir / "scoreboard.txt").read_text("utf-8")

    brief = json.loads((batch_dir / "experiment_brief.json").read_text("utf-8"))
    assert brief["kind"] == "quanttradeai.experiment_brief"
    assert brief["schema_version"] == 1
    assert brief["winner"]["agent_name"] == "breakout_gpt"
    assert brief["recommended_next_action"]["action"] == "promote_winner_to_paper"
    assert brief["commands"]["promote_winner"] == (
        f"quanttradeai promote --run {brief['winner']['run_id']} "
        f"-c {config_path.resolve().as_posix()}"
    )
    assert "compare_top_runs" in brief["commands"]

    manifest = json.loads((batch_dir / "batch_manifest.json").read_text("utf-8"))
    assert manifest["artifacts"]["experiment_brief_json"] == str(
        batch_dir / "experiment_brief.json"
    )

    batch_summary = json.loads((batch_dir / "summary.json").read_text("utf-8"))
    assert batch_summary["run_type"] == "batch"
    assert batch_summary["run_id"] == f"agent/batches/{batch_dir.name}"

    runs_result = runner.invoke(app, ["runs", "list", "--type", "batch", "--json"])
    assert runs_result.exit_code == 0, runs_result.stdout
    run_records = json.loads(runs_result.stdout)
    assert [record["run_id"] for record in run_records] == [
        f"agent/batches/{batch_dir.name}"
    ]


def test_strategy_lab_agent_run_all_backtest_enumerates_both_agents(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/project.yaml")
    init_result = runner.invoke(
        app,
        ["init", "--template", "strategy-lab", "--output", str(config_path)],
    )
    assert init_result.exit_code == 0, init_result.stdout

    observed_agents: list[str] = []

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "backtest"
        assert skip_validation is False
        assert project_config_override is None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert run_timestamp is not None
        observed_agents.append(agent_name)

        run_dir = Path("runs") / "agent" / "backtest" / f"{run_timestamp}_{agent_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"net_sharpe": 1.0, "net_pnl": 0.1, "net_mdd": -0.05},
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/backtest/{run_dir.name}",
            "run_type": "agent",
            "mode": "backtest",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["agent_count"] == 2
    assert observed_agents == ["rsi_reversion", "sma_trend"]
    assert [item["agent_name"] for item in payload["results"]] == [
        "rsi_reversion",
        "sma_trend",
    ]


def test_agent_run_all_paper_writes_batch_artifacts_and_sorts_by_total_pnl(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "paper"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is None
        run_dir = (
            Path("runs") / "agent" / "paper" / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        pnl_by_agent = {
            "rsi_reversion": 80.0,
            "paper_momentum": 120.0,
            "breakout_gpt": 240.0,
            "hybrid_swing_agent": 170.0,
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "total_pnl": pnl_by_agent[agent_name],
                    "portfolio_value": 100000.0 + pnl_by_agent[agent_name],
                    "execution_count": int(pnl_by_agent[agent_name] / 10.0),
                    "decision_count": int(pnl_by_agent[agent_name] / 5.0),
                    "risk_status": "ok",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/paper/{run_dir.name}",
            "run_type": "agent",
            "mode": "paper",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "paper_source": "replay",
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "paper",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    batch_dir = Path(payload["run_dir"])

    assert payload["status"] == "success"
    assert payload["mode"] == "paper"
    assert batch_dir.name.endswith("_paper")
    assert (batch_dir / "results.json").is_file()
    assert (batch_dir / "scoreboard.json").is_file()

    results_payload = json.loads((batch_dir / "results.json").read_text("utf-8"))
    assert results_payload["mode"] == "paper"
    assert results_payload["scoreboard_sort_by"] == "total_pnl"
    assert [item["agent_name"] for item in results_payload["results"]] == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]
    assert all(
        item["run_id"].startswith("agent/paper/") for item in results_payload["results"]
    )
    assert all(
        Path(item["run_dir"]).parts[:3] == ("runs", "agent", "paper")
        for item in results_payload["results"]
    )
    assert all(item["paper_source"] == "replay" for item in results_payload["results"])

    scoreboard_payload = json.loads((batch_dir / "scoreboard.json").read_text("utf-8"))
    assert scoreboard_payload["sort_by"] == "total_pnl"
    assert [record["name"] for record in scoreboard_payload["records"]] == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]
    scoreboard_text = (batch_dir / "scoreboard.txt").read_text("utf-8")
    assert "TOTAL_PNL" in scoreboard_text
    assert "RISK" in scoreboard_text


def test_agent_run_all_paper_preserves_child_failures_and_logs(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "paper"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is None
        run_dir = (
            Path("runs") / "agent" / "paper" / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        if agent_name == "hybrid_swing_agent":
            child_summary = {
                "run_id": f"agent/paper/{run_dir.name}",
                "run_type": "agent",
                "mode": "paper",
                "name": agent_name,
                "status": "failed",
                "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
                "symbols": ["AAPL"],
                "warnings": ["child warning"],
                "artifacts": {},
                "run_dir": str(run_dir),
                "error": "paper child failed",
            }
            (run_dir / "summary.json").write_text(
                json.dumps(child_summary, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError("simulated paper failure")

        pnl_by_agent = {
            "rsi_reversion": 80.0,
            "paper_momentum": 120.0,
            "breakout_gpt": 240.0,
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "total_pnl": pnl_by_agent[agent_name],
                    "portfolio_value": 100000.0 + pnl_by_agent[agent_name],
                    "execution_count": int(pnl_by_agent[agent_name] / 10.0),
                    "decision_count": int(pnl_by_agent[agent_name] / 5.0),
                    "risk_status": "ok",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/paper/{run_dir.name}",
            "run_type": "agent",
            "mode": "paper",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "paper_source": "replay",
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "paper",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 1
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["status"] == "failed"
    assert payload["failure_count"] == 1
    assert payload["success_count"] == 3

    failed_entry = next(
        item
        for item in payload["results"]
        if item["agent_name"] == "hybrid_swing_agent"
    )
    assert failed_entry["status"] == "failed"
    assert failed_entry["warnings"] == ["child warning"]
    assert failed_entry["run_id"].startswith("agent/paper/")
    assert Path(failed_entry["stderr_log"]).read_text("utf-8")
    successful_names = {
        item["agent_name"] for item in payload["results"] if item["status"] == "success"
    }
    assert successful_names == {"breakout_gpt", "paper_momentum", "rsi_reversion"}


def test_agent_run_all_live_writes_batch_artifacts_and_sorts_by_total_pnl(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)
    _set_all_agents_live(config_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "live"
        assert skip_validation is False
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is None
        run_dir = (
            Path("runs") / "agent" / "live" / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        pnl_by_agent = {
            "rsi_reversion": 80.0,
            "paper_momentum": 120.0,
            "breakout_gpt": 240.0,
            "hybrid_swing_agent": 170.0,
        }
        backend_by_agent = {
            "rsi_reversion": "simulated",
            "paper_momentum": "alpaca",
            "breakout_gpt": "simulated",
            "hybrid_swing_agent": "simulated",
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "status": "available",
                    "total_pnl": pnl_by_agent[agent_name],
                    "portfolio_value": 100000.0 + pnl_by_agent[agent_name],
                    "execution_count": int(pnl_by_agent[agent_name] / 10.0),
                    "decision_count": int(pnl_by_agent[agent_name] / 5.0),
                    "risk_status": "ok",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/live/{run_dir.name}",
            "run_type": "agent",
            "mode": "live",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "execution_backend": backend_by_agent[agent_name],
            "broker_provider": (
                "alpaca" if backend_by_agent[agent_name] == "alpaca" else None
            ),
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "live",
                "--acknowledge-live",
                "multi_agent_lab",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    batch_dir = Path(payload["run_dir"])

    assert payload["status"] == "success"
    assert payload["mode"] == "live"
    assert batch_dir.name.endswith("_live")
    assert (batch_dir / "batch_manifest.json").is_file()
    assert (batch_dir / "results.json").is_file()
    assert (batch_dir / "scoreboard.json").is_file()
    assert (batch_dir / "scoreboard.txt").is_file()

    results_payload = json.loads((batch_dir / "results.json").read_text("utf-8"))
    assert results_payload["mode"] == "live"
    assert results_payload["scoreboard_sort_by"] == "total_pnl"
    assert [item["agent_name"] for item in results_payload["results"]] == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]
    assert all(
        item["run_id"].startswith("agent/live/") for item in results_payload["results"]
    )
    assert all(
        Path(item["run_dir"]).parts[:3] == ("runs", "agent", "live")
        for item in results_payload["results"]
    )
    paper_momentum = next(
        item
        for item in results_payload["results"]
        if item["agent_name"] == "paper_momentum"
    )
    assert paper_momentum["execution_backend"] == "alpaca"
    assert paper_momentum["broker_provider"] == "alpaca"

    manifest = json.loads((batch_dir / "batch_manifest.json").read_text("utf-8"))
    manifest_paper_momentum = next(
        item for item in manifest["agents"] if item["agent_name"] == "paper_momentum"
    )
    assert manifest_paper_momentum["execution_backend"] == "alpaca"
    assert manifest_paper_momentum["broker_provider"] == "alpaca"

    scoreboard_payload = json.loads((batch_dir / "scoreboard.json").read_text("utf-8"))
    assert scoreboard_payload["sort_by"] == "total_pnl"
    assert [record["name"] for record in scoreboard_payload["records"]] == [
        "breakout_gpt",
        "hybrid_swing_agent",
        "paper_momentum",
        "rsi_reversion",
    ]
    scoreboard_text = (batch_dir / "scoreboard.txt").read_text("utf-8")
    assert "TOTAL_PNL" in scoreboard_text
    assert "RISK" in scoreboard_text


def test_agent_run_all_live_preserves_child_failures_and_logs(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)
    _set_all_agents_live(config_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "live"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is None
        run_dir = (
            Path("runs") / "agent" / "live" / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        if agent_name == "hybrid_swing_agent":
            child_summary = {
                "run_id": f"agent/live/{run_dir.name}",
                "run_type": "agent",
                "mode": "live",
                "name": agent_name,
                "status": "failed",
                "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
                "symbols": ["AAPL"],
                "warnings": ["child warning"],
                "artifacts": {},
                "run_dir": str(run_dir),
                "error": "live child failed",
            }
            (run_dir / "summary.json").write_text(
                json.dumps(child_summary, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError("simulated live failure")

        pnl_by_agent = {
            "rsi_reversion": 80.0,
            "paper_momentum": 120.0,
            "breakout_gpt": 240.0,
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "total_pnl": pnl_by_agent[agent_name],
                    "portfolio_value": 100000.0 + pnl_by_agent[agent_name],
                    "execution_count": int(pnl_by_agent[agent_name] / 10.0),
                    "decision_count": int(pnl_by_agent[agent_name] / 5.0),
                    "risk_status": "ok",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/live/{run_dir.name}",
            "run_type": "agent",
            "mode": "live",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "execution_backend": "simulated",
            "broker_provider": None,
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "live",
                "--acknowledge-live",
                "multi_agent_lab",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 1
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["status"] == "failed"
    assert payload["failure_count"] == 1
    assert payload["success_count"] == 3

    failed_entry = next(
        item
        for item in payload["results"]
        if item["agent_name"] == "hybrid_swing_agent"
    )
    assert failed_entry["status"] == "failed"
    assert failed_entry["warnings"] == ["child warning"]
    assert failed_entry["run_id"].startswith("agent/live/")
    assert Path(failed_entry["stderr_log"]).read_text("utf-8")
    successful_names = {
        item["agent_name"] for item in payload["results"] if item["status"] == "success"
    }
    assert successful_names == {"breakout_gpt", "paper_momentum", "rsi_reversion"}


def test_agent_run_sweep_writes_variant_artifacts_and_preserves_source_config(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_sweep_assets(tmp_path)
    original_config = config_path.read_text(encoding="utf-8")

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert mode == "backtest"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
        assert project_config_override is not None
        variant_config = project_config_override
        variant_agent = variant_config["agents"][0]
        assert variant_agent["name"] == agent_name
        run_dir = (
            Path("runs")
            / "agent"
            / "backtest"
            / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        sharpe_by_agent = {
            "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-70_0": 0.8,
            "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-75_0": 1.4,
            "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-70_0": 1.1,
            "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-75_0": 2.2,
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "net_sharpe": sharpe_by_agent[agent_name],
                    "net_pnl": sharpe_by_agent[agent_name] / 10.0,
                    "net_mdd": -0.05,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/backtest/{run_dir.name}",
            "run_type": "agent",
            "mode": "backtest",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--sweep",
                "rsi_threshold_grid",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--max-concurrency",
                "2",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    batch_dir = Path(payload["run_dir"])

    assert payload["status"] == "success"
    assert payload["batch_type"] == "sweep"
    assert payload["sweep"]["name"] == "rsi_threshold_grid"
    assert payload["sweep"]["base_agent_name"] == "rsi_reversion"
    assert payload["agent_count"] == 4
    assert config_path.read_text(encoding="utf-8") == original_config
    assert (batch_dir / "batch_manifest.json").is_file()
    assert (batch_dir / "results.json").is_file()
    assert (batch_dir / "scoreboard.json").is_file()
    assert (batch_dir / "experiment_brief.json").is_file()
    assert (batch_dir / "experiment_brief.md").is_file()
    assert (batch_dir / "variants").is_dir()
    assert payload["artifacts"]["experiment_brief_json"] == str(
        batch_dir / "experiment_brief.json"
    )

    results_payload = json.loads((batch_dir / "results.json").read_text("utf-8"))
    assert results_payload["batch_type"] == "sweep"
    assert results_payload["sweep"]["parameters"] == [
        {"path": "rule.buy_below", "values": [25.0, 30.0]},
        {"path": "rule.sell_above", "values": [70.0, 75.0]},
    ]
    assert [item["agent_name"] for item in results_payload["results"]] == [
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-70_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-75_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-70_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-75_0",
    ]
    assert len(results_payload["expanded_variants"]) == 4
    assert all(item["variant_project_config"] for item in results_payload["results"])
    assert all(
        item["promote_command"]
        == (
            f"quanttradeai promote --run {item['run_id']} "
            f"-c {config_path.resolve().as_posix()}"
        )
        for item in results_payload["results"]
    )

    scoreboard_payload = json.loads((batch_dir / "scoreboard.json").read_text("utf-8"))
    ordered_names = [record["name"] for record in scoreboard_payload["records"]]
    assert ordered_names == [
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-75_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-75_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-70_0",
        "rsi_reversion__rsi_threshold_grid__buy_below-25_0__sell_above-70_0",
    ]

    child_summary = json.loads(
        (Path(results_payload["results"][0]["run_dir"]) / "summary.json").read_text(
            "utf-8"
        )
    )
    assert child_summary["sweep"] == {
        "name": "rsi_threshold_grid",
        "base_agent_name": "rsi_reversion",
        "parameters": {
            "rule.buy_below": 25.0,
            "rule.sell_above": 70.0,
        },
        "materializable": True,
    }

    brief = json.loads((batch_dir / "experiment_brief.json").read_text("utf-8"))
    assert brief["winner"]["agent_name"] == (
        "rsi_reversion__rsi_threshold_grid__buy_below-30_0__sell_above-75_0"
    )
    assert brief["recommended_next_action"]["action"] == "materialize_sweep_winner"
    assert brief["commands"]["promote_winner"] == (
        f"quanttradeai promote --run {brief['winner']['run_id']} "
        f"-c {config_path.resolve().as_posix()}"
    )
    assert brief["commands"]["run_promoted_paper_agent"] == (
        "quanttradeai agent run --agent rsi_reversion "
        f"-c {config_path.resolve().as_posix()} --mode paper"
    )


def test_agent_run_sweep_uses_canonical_project_root_for_llm_assets(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config" / "project.yaml"

    init_result = runner.invoke(
        app,
        ["init", "--template", "llm-agent", "--output", str(config_path)],
    )
    assert init_result.exit_code == 0, init_result.stdout

    project_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project_config["data"]["start_date"] = "2024-01-01"
    project_config["data"]["end_date"] = "2024-02-09"
    project_config["data"]["test_start"] = "2024-01-26"
    project_config["data"]["test_end"] = "2024-01-31"
    project_config["sweeps"] = [
        {
            "name": "model_grid",
            "kind": "agent_backtest",
            "agent": "breakout_gpt",
            "parameters": [{"path": "llm.model", "values": ["gpt-5.3"]}],
        }
    ]
    config_path.write_text(
        yaml.safe_dump(project_config, sort_keys=False),
        encoding="utf-8",
    )

    history_index = pd.date_range("2024-01-01", periods=40, freq="D")
    history = pd.DataFrame(
        {
            "Open": [100.0 + idx for idx in range(len(history_index))],
            "High": [101.0 + idx for idx in range(len(history_index))],
            "Low": [99.0 + idx for idx in range(len(history_index))],
            "Close": [100.5 + idx for idx in range(len(history_index))],
            "Volume": [1000.0 + idx for idx in range(len(history_index))],
        },
        index=history_index,
    )

    def _fake_generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        featured = df.copy()
        featured["rsi"] = [45.0 + (idx % 10) for idx in range(len(featured))]
        return featured

    with (
        patch(
            "quanttradeai.agents.backtest.DataLoader.fetch_data",
            return_value={"AAPL": history},
        ),
        patch(
            "quanttradeai.agents.backtest.DataProcessor.generate_features",
            _fake_generate_features,
        ),
        patch(
            "quanttradeai.agents.llm.completion",
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"action": "hold", "reason": "grid check"}
                            )
                        }
                    }
                ]
            },
        ),
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--sweep",
                "model_grid",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
                "--skip-validation",
            ],
        )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["status"] == "success"
    assert payload["success_count"] == 1

    child_run_dir = Path(payload["results"][0]["run_dir"])
    assert (child_run_dir / "prompt_samples.json").is_file()

    summary = json.loads((child_run_dir / "summary.json").read_text("utf-8"))
    resolved_project_path = Path(summary["artifacts"]["resolved_project_config"])
    resolved_project = yaml.safe_load(resolved_project_path.read_text("utf-8"))
    assert resolved_project["agents"][0]["name"] == payload["results"][0]["agent_name"]
    assert payload["results"][0]["variant_project_config"]


def test_agent_run_all_returns_non_zero_after_partial_failure(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = _seed_agent_assets(tmp_path)

    def _fake_run_project_agent(
        *,
        project_config_path: str,
        agent_name: str,
        mode: str,
        skip_validation: bool,
        project_config_override: dict | None = None,
        run_timestamp: str | None = None,
    ):
        assert project_config_override is None
        run_dir = (
            Path("runs")
            / "agent"
            / "backtest"
            / f"{run_timestamp}_{agent_name.lower()}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        if agent_name == "paper_momentum":
            summary = {
                "run_id": f"agent/backtest/{run_dir.name}",
                "run_type": "agent",
                "mode": "backtest",
                "name": agent_name,
                "status": "failed",
                "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
                "symbols": [],
                "warnings": [],
                "artifacts": {},
                "run_dir": str(run_dir),
                "error": "model failure",
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            raise ValueError("model failure")

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"net_sharpe": 1.0, "net_pnl": 0.1, "net_mdd": -0.02}, indent=2),
            encoding="utf-8",
        )
        summary = {
            "run_id": f"agent/backtest/{run_dir.name}",
            "run_type": "agent",
            "mode": "backtest",
            "name": agent_name,
            "status": "success",
            "timestamps": {"started_at": "2026-01-01T00:00:00+00:00"},
            "symbols": ["AAPL"],
            "warnings": [],
            "artifacts": {"metrics": str(metrics_path)},
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary, []

    with patch(
        "quanttradeai.agents.batch.run_project_agent",
        side_effect=_fake_run_project_agent,
    ):
        result = runner.invoke(
            app,
            [
                "agent",
                "run",
                "--all",
                "--config",
                str(config_path),
                "--mode",
                "backtest",
            ],
        )

    assert result.exit_code == 1
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload["status"] == "failed"
    assert payload["failure_count"] == 1
    failed_result = next(
        item for item in payload["results"] if item["agent_name"] == "paper_momentum"
    )
    assert failed_result["status"] == "failed"
    assert failed_result["error"] == "model failure"
