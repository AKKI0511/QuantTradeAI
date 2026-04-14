import json
from pathlib import Path
import re
from unittest.mock import patch

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


def test_agent_run_all_rejects_non_backtest_modes(tmp_path: Path, monkeypatch):
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
            "--mode",
            "paper",
        ],
    )

    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "--all currently supports only --mode backtest" in combined


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
        run_timestamp: str | None = None,
    ):
        assert mode == "backtest"
        assert run_timestamp is not None
        assert Path(project_config_path).resolve() == config_path.resolve()
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
        run_timestamp: str | None = None,
    ):
        assert mode == "backtest"
        assert run_timestamp is not None
        variant_config = yaml.safe_load(Path(project_config_path).read_text("utf-8"))
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
    assert (batch_dir / "variants").is_dir()

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
        "promotable": False,
    }


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
        run_timestamp: str | None = None,
    ):
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
