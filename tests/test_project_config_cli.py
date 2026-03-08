import json
from pathlib import Path

import pandas as pd
import yaml
import pytest
from typer.testing import CliRunner

from quanttradeai.cli import PROJECT_TEMPLATES, _project_to_runtime_configs, app


runner = CliRunner()


def test_project_to_runtime_configs_maps_custom_features_to_supported_keys():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["hybrid"], sort_keys=False)
    )
    project_cfg["features"]["definitions"].append(
        {
            "name": "mean_reversion_10",
            "type": "custom",
            "params": {"kind": "mean_reversion", "lookback": 10},
        }
    )

    _, features_cfg = _project_to_runtime_configs(project_cfg)

    assert {"volume_momentum": {"periods": [20]}} in features_cfg["custom_features"]
    assert {"mean_reversion": {"periods": [10]}} in features_cfg["custom_features"]


def test_project_to_runtime_configs_rejects_unmappable_custom_features():
    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["features"]["definitions"] = [
        {"name": "alpha_signal", "type": "custom", "params": {"window": 10}}
    ]

    with pytest.raises(ValueError, match="must map to one of"):
        _project_to_runtime_configs(project_cfg)


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
    assert (
        rendered["project"]["name"] == PROJECT_TEMPLATES["research"]["project"]["name"]
    )


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
        yaml.safe_dump(
            {"project": {"name": "demo", "profile": "paper"}}, sort_keys=False
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "--config", str(cfg_path)])

    assert result.exit_code == 1
    assert "missing required section(s)" in result.stderr.lower()


def test_validate_writes_resolved_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False), encoding="utf-8"
    )

    result = runner.invoke(app, ["validate", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    payload = json.loads(result.stdout[result.stdout.index("{") :])
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

    cfg_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
    )

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
        (experiment_dir / "test_window_coverage.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "results": {"AAPL": {"test_metrics": {"accuracy": 0.75}}},
            "coverage": {
                "path": str(experiment_dir / "test_window_coverage.json"),
                "fallback_symbols": [],
            },
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
    assert (
        summary_payload["project_name"]
        == PROJECT_TEMPLATES["research"]["project"]["name"]
    )
    assert Path(summary_payload["artifacts"]["resolved_project_config"]).is_file()
    assert metrics_payload["status"] == "available"


def test_research_run_forwards_label_settings_to_runtime_model_config(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")

    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["research"]["labels"] = {
        "type": "forward_return",
        "horizon": 9,
        "buy_threshold": 0.02,
        "sell_threshold": -0.03,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(project_cfg, sort_keys=False), encoding="utf-8")

    def _fake_pipeline(config_path, *args, **kwargs):
        runtime_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert runtime_cfg["labels"]["horizon"] == 9
        assert runtime_cfg["labels"]["buy_threshold"] == 0.02
        assert runtime_cfg["labels"]["sell_threshold"] == -0.03

        experiment_dir = Path("models/experiments/20260101_000001")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "results.json").write_text("{}", encoding="utf-8")
        (experiment_dir / "test_window_coverage.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "results": {"AAPL": {"test_metrics": {"accuracy": 0.8}}},
            "coverage": {
                "path": str(experiment_dir / "test_window_coverage.json"),
                "fallback_symbols": [],
            },
            "experiment_dir": str(experiment_dir),
        }

    monkeypatch.setattr("quanttradeai.cli.run_pipeline", _fake_pipeline)

    run_result = runner.invoke(app, ["research", "run", "--config", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.stdout


def test_research_run_uses_train_fitted_preprocessing_end_to_end(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")

    project_cfg = yaml.safe_load(
        yaml.safe_dump(PROJECT_TEMPLATES["research"], sort_keys=False)
    )
    project_cfg["data"]["start_date"] = "2020-01-01"
    project_cfg["data"]["end_date"] = "2020-09-16"
    project_cfg["data"]["test_start"] = "2020-09-02"
    project_cfg["data"]["test_end"] = "2020-09-15"
    project_cfg["research"]["labels"]["horizon"] = 1
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(project_cfg, sort_keys=False), encoding="utf-8")

    index = pd.date_range("2020-01-01", periods=260, freq="D")
    raw_df = pd.DataFrame(
        {
            "Open": range(260),
            "High": range(260),
            "Low": range(260),
            "Close": range(260),
            "Volume": [100] * 260,
            "alpha_feature": list(range(245)) + [1000] * 15,
        },
        index=index,
    )

    captured_frames: list[pd.DataFrame] = []

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            self.feature_columns = None

        def prepare_data(self, frame):
            captured_frames.append(frame.copy())
            X = frame[["alpha_feature"]].to_numpy()
            y = frame["label"].to_numpy()
            self.feature_columns = ["alpha_feature"]
            return X, y

        def optimize_hyperparameters(self, X, y, n_trials=50):
            return {}

        def train(self, X, y, params=None):
            return None

        def evaluate(self, X, y):
            return {"accuracy": 1.0}

        def save_model(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    from quanttradeai.data.processor import DataProcessor

    monkeypatch.setattr(
        DataProcessor,
        "generate_features",
        lambda self, df: self._clean_data(df.copy()),
    )

    with monkeypatch.context() as context:
        from quanttradeai import main as main_module

        loader_mock = type("Loader", (), {})()
        loader_mock.fetch_data = lambda refresh=False: {"AAPL": raw_df}
        loader_mock.validate_data = lambda data: (True, {"AAPL": {"passed": True}})
        context.setattr(main_module, "DataLoader", lambda *args, **kwargs: loader_mock)
        context.setattr(main_module, "MomentumClassifier", FakeClassifier)

        result = runner.invoke(
            app,
            ["research", "run", "--config", str(cfg_path), "--skip-validation"],
        )

    assert result.exit_code == 0, result.stdout
    assert len(captured_frames) >= 2

    train_frame, test_frame = captured_frames[0], captured_frames[1]
    retained = raw_df.iloc[200:].copy()
    train_raw = retained.loc[retained.index < pd.Timestamp("2020-09-02"), "alpha_feature"]
    train_clipped = train_raw.clip(
        lower=train_raw.quantile(0.01),
        upper=train_raw.quantile(0.99),
    )
    expected = (
        train_raw.quantile(0.99) - train_clipped.mean()
    ) / train_clipped.std(ddof=0)

    assert train_frame["alpha_feature"].mean() == pytest.approx(0.0, abs=1e-9)
    assert test_frame["alpha_feature"].iloc[0] == pytest.approx(expected)

    run_dirs = sorted(path for path in Path("runs").iterdir() if path.is_dir())
    latest_run = run_dirs[-1]
    summary_payload = json.loads((latest_run / "summary.json").read_text("utf-8"))
    preprocessing_path = Path(summary_payload["artifacts"]["preprocessing"])
    assert preprocessing_path.is_file()


def test_research_run_fails_for_malformed_project_config(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = Path("config/project.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {"project": {"name": "broken", "profile": "research"}}, sort_keys=False
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["research", "run", "--config", str(cfg_path)])

    assert result.exit_code == 1
    assert "research run failed" in result.stderr.lower()

    run_dirs = sorted(path for path in Path("runs").iterdir() if path.is_dir())
    assert run_dirs
    latest_run = run_dirs[-1]
    summary_payload = json.loads(
        (latest_run / "summary.json").read_text(encoding="utf-8")
    )
    metrics_payload = json.loads(
        (latest_run / "metrics.json").read_text(encoding="utf-8")
    )

    assert summary_payload["status"] == "failed"
    assert metrics_payload["status"] == "placeholder"
