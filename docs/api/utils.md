# Utils API

## Overview

Utilities under `quanttradeai.utils` are advanced APIs for project config loading, runtime config compilation, validation, run records, result payloads, scoreboards, comparisons, sweeps, metrics, and impact defaults.

These APIs are not exported from `quanttradeai.__init__`. Treat them as advanced building blocks: useful for automation and embedding, but less stable than the top-level package imports.

## Public Imports

```python
from quanttradeai.utils.project_config import load_project_config
from quanttradeai.utils.config_validator import validate_project_config, validate_all
from quanttradeai.utils.run_records import discover_runs, RunFilters, filter_runs
from quanttradeai.utils.run_scoreboard import attach_scoreboard, render_scoreboard_table
from quanttradeai.utils.run_compare import build_run_comparison, render_run_comparison
from quanttradeai.utils.sweeps import expand_agent_backtest_sweep, expand_research_sweep
from quanttradeai.utils.metrics import compute_performance
```

## Main Classes and Functions

| Area | API | Purpose |
| --- | --- | --- |
| Project config | `load_project_config`, `compile_*_runtime_config` | Load project YAML and derive runtime configs |
| Validation | `validate_project_config`, `validate_all`, `ValidationResult` | Validate project or focused runtime configs |
| Run records | `create_run_dir`, `discover_runs`, `filter_runs`, `RunFilters` | Standard run directory and listing helpers |
| Run results | `attach_run_result`, `build_run_result`, `compact_cli_result` | Compact machine-readable result summaries |
| Scoreboard | `load_scoreboard_record`, `attach_scoreboard`, `sort_run_records`, `render_scoreboard_table` | Normalize and rank run metrics |
| Compare | `build_run_comparison`, `render_run_comparison` | Compare 2 to 4 runs from the same run family |
| Sweeps | `expand_agent_backtest_sweep`, `expand_research_sweep`, override helpers | Materialize project sweep variants |
| Metrics | `classification_metrics`, `compute_performance`, `sharpe_ratio`, `max_drawdown`, `cagr` | Model and performance metrics |
| Impact defaults | `load_impact_config`, `merge_execution_with_impact` | Load and merge per-asset impact defaults |
| Paths | `infer_project_root`, `resolve_project_path` | Resolve project-relative paths |

## Project Config

### `LoadedProjectConfig`

**Signature**

```python
@dataclass
class LoadedProjectConfig:
    raw: dict[str, Any]
    source_path: str
    warnings: list[str]
```

### `load_project_config`

**Signature**

```python
def load_project_config(
    config_path: Path | str = "config/project.yaml",
) -> LoadedProjectConfig
```

Loads YAML, normalizes supported project config aliases, and returns a structured object.

```python
from quanttradeai.utils.project_config import load_project_config

loaded = load_project_config("config/project.yaml")
project = loaded.raw
```

### Runtime Compilation

| Function | Signature | Returns |
| --- | --- | --- |
| `compile_research_runtime_configs` | `(project_config, *, require_research=True)` | `(model_cfg, features_cfg, backtest_cfg)` |
| `compile_streaming_runtime_config` | `(project_config, *, mode="paper", require_realtime=False)` | Streaming runtime config dict |
| `compile_paper_streaming_runtime_config` | `(project_config, *, require_realtime=False)` | Paper streaming runtime config dict |
| `compile_live_streaming_runtime_config` | `(project_config)` | Live streaming runtime config dict |
| `compile_live_risk_runtime_config` | `(project_config)` | `{"risk_management": ...}` |
| `compile_live_position_manager_runtime_config` | `(project_config)` | `{"position_manager": ...}` |
| `paper_replay_enabled` | `(project_config)` | `bool` |
| `resolve_paper_replay_window` | `(project_config)` | `ReplayWindow | None` |

```python
from quanttradeai.utils.project_config import (
    compile_research_runtime_configs,
    load_project_config,
)

project = load_project_config("config/project.yaml").raw
model_cfg, features_cfg, backtest_cfg = compile_research_runtime_configs(project)
```

**Common errors**

- Missing required project sections.
- Paper replay window outside the configured data date range.
- Live runtime compilation without required streaming, risk, or position manager sections.

## Validation

### `validate_project_config`

**Signature**

```python
def validate_project_config(
    config_path: Path | str = "config/project.yaml",
    *,
    output_dir: Path | str = "reports/config_validation",
    project_config_override: dict[str, Any] | None = None,
    timestamp_subdir: bool = True,
) -> dict
```

Validates a project config, writes a resolved project YAML and JSON summary, and returns:

```python
{
    "timestamp": str,
    "config_path": str,
    "all_passed": True,
    "summary": dict,
    "warnings": list[str],
    "artifacts": {
        "resolved_config": str,
        "summary": str,
    },
}
```

```python
from quanttradeai.utils.config_validator import validate_project_config

result = validate_project_config("config/project.yaml")
resolved_path = result["artifacts"]["resolved_config"]
```

### `validate_all`

**Signature**

```python
def validate_all(
    config_paths: Mapping[str, Path | str] | None = None,
    *,
    output_dir: Path | str = "reports/config_validation",
) -> dict
```

Validates known focused runtime config files and writes JSON/CSV reports.

```python
from quanttradeai.utils.config_validator import validate_all

summary = validate_all()
```

### `ValidationResult`

**Signature**

```python
@dataclass
class ValidationResult:
    name: str
    path: str
    passed: bool
    details: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict
```

## Run Records

### `RunFilters`

**Signature**

```python
@dataclass(frozen=True, slots=True)
class RunFilters:
    run_type: str = "all"
    mode: str = "all"
    status: str = "all"
    limit: int = 20
```

### Functions

| Function | Signature | Purpose |
| --- | --- | --- |
| `create_run_dir` | `(*, run_type, mode, name, runs_root="runs", timestamp=None) -> tuple[Path, str]` | Create standardized run directory and run id |
| `normalize_run_summary` | `(summary, *, run_dir, runs_root="runs") -> dict | None` | Convert summary JSON to shared run record |
| `apply_required_run_fields` | `(summary, *, run_dir, run_type, mode, name=None, runs_root="runs") -> dict` | Populate required summary fields |
| `discover_runs` | `(runs_root="runs") -> list[dict]` | Find run summaries under `runs/` |
| `filter_runs` | `(records, filters) -> list[dict]` | Apply run type/mode/status/limit filters |

```python
from quanttradeai.utils.run_records import RunFilters, discover_runs, filter_runs

records = discover_runs("runs")
recent_backtests = filter_runs(
    records,
    RunFilters(run_type="agent", mode="backtest", status="success", limit=10),
)
```

## Run Results

| Function | Signature | Purpose |
| --- | --- | --- |
| `attach_run_result` | `(summary, *, project_config_path="config/project.yaml", metrics_payload=None, batch_results=None, scoreboard_order=None, scoreboard_sort_by=None, top_n=5) -> dict` | Mutates summary with `run_result` |
| `build_run_result` | same keyword inputs | Returns sparse result payload |
| `compact_cli_result` | `(summary) -> dict` | Returns compact completion-oriented payload |

```python
from quanttradeai.utils.run_result import compact_cli_result

payload = compact_cli_result(summary)
```

`run_result` payloads are intentionally sparse. They keep the key metrics and artifact pointers rather than duplicating large CSV or JSON artifacts.

## Scoreboard

| Function | Signature | Purpose |
| --- | --- | --- |
| `load_scoreboard_record` | `(record) -> dict` | Load metrics for one run record |
| `attach_scoreboard` | `(records) -> list[dict]` | Add `scoreboard` payloads |
| `sort_run_records` | `(records, *, sort_by, ascending) -> list[dict]` | Sort by base or scoreboard fields |
| `render_scoreboard_table` | `(records) -> str` | Render text table |

Supported sort fields:

```python
{
    "started_at",
    "name",
    "status",
    "accuracy",
    "f1",
    "net_sharpe",
    "net_pnl",
    "total_pnl",
    "execution_count",
    "decision_count",
}
```

```python
from quanttradeai.utils.run_records import discover_runs
from quanttradeai.utils.run_scoreboard import (
    attach_scoreboard,
    render_scoreboard_table,
    sort_run_records,
)

records = attach_scoreboard(discover_runs())
records = sort_run_records(records, sort_by="net_sharpe", ascending=False)
print(render_scoreboard_table(records[:10]))
```

## Run Comparison

### `build_run_comparison`

**Signature**

```python
def build_run_comparison(
    *,
    run_ids: list[str],
    sort_by: str = "started_at",
    ascending: bool = False,
    runs_root: Path | str = "runs",
) -> dict[str, Any]
```

Compares 2 to 4 explicit run ids from the same run family:

| Family | Metrics |
| --- | --- |
| `research` | `accuracy`, `f1`, `net_sharpe`, `net_pnl` |
| `agent/backtest` | `net_sharpe`, `net_pnl`, `net_mdd`, `decision_count` |
| `agent/paper` | `total_pnl`, `portfolio_value`, `execution_count`, `decision_count`, `risk_status` |
| `agent/live` | `total_pnl`, `portfolio_value`, `execution_count`, `decision_count`, `risk_status` |

```python
from quanttradeai.utils.run_compare import build_run_comparison, render_run_comparison

comparison = build_run_comparison(
    run_ids=[
        "agent/backtest/20260101_120000_agent_a",
        "agent/backtest/20260101_121500_agent_b",
    ],
)

print(render_run_comparison(comparison))
```

## Sweeps

Sweep helpers materialize project config variants for research and agent backtests.

| Function | Signature | Purpose |
| --- | --- | --- |
| `is_scalar_sweep_value` | `(value) -> bool` | Validate scalar sweep leaf |
| `slug_scalar_sweep_value` | `(value) -> str` | Stable name fragment |
| `resolve_agent_scalar_path` | `(agent_config, path) -> tuple[parent, leaf_key, value]` | Resolve an agent sweep path |
| `resolve_research_scalar_path` | `(project_config, path) -> tuple[parent, leaf_key, value]` | Resolve a research sweep path |
| `apply_agent_scalar_overrides` | `(agent_config, overrides) -> dict` | Clone and apply agent overrides |
| `apply_research_scalar_overrides` | `(project_config, overrides) -> dict` | Clone and apply research overrides |
| `build_agent_sweep_variant_name` | `(*, base_agent_name, sweep_name, parameters) -> str` | Deterministic agent variant name |
| `build_research_sweep_variant_name` | `(*, project_name, sweep_name, parameters) -> str` | Deterministic research variant name |
| `resolve_project_sweep` | `(project_config, sweep_name) -> dict` | Return named sweep |
| `expand_agent_backtest_sweep` | `(project_config, sweep_name) -> dict` | Materialize agent backtest variants |
| `expand_research_sweep` | `(project_config, sweep_name) -> dict` | Materialize research variants |
| `sweep_summary_payload` | `(*, sweep_name, base_agent_name, parameters) -> dict` | Summary payload for sweep child runs |

```python
from quanttradeai.utils.project_config import load_project_config
from quanttradeai.utils.sweeps import expand_agent_backtest_sweep

project = load_project_config("config/project.yaml").raw
expansion = expand_agent_backtest_sweep(project, "risk_sweep")
variant_names = [variant["name"] for variant in expansion["variants"]]
```

## Metrics

| Function | Signature | Return |
| --- | --- | --- |
| `classification_metrics` | `(y_true: np.ndarray, y_pred: np.ndarray) -> dict` | Accuracy, weighted precision, weighted recall, weighted F1 |
| `sharpe_ratio` | `(returns: pd.Series, risk_free_rate=0.0) -> float` | Annualized Sharpe, `0.0` for empty/zero-variance inputs |
| `max_drawdown` | `(equity_curve: pd.Series) -> float` | Minimum drawdown |
| `cagr` | `(equity_curve: pd.Series) -> float` | Daily-data CAGR |
| `compute_performance` | `(data: pd.DataFrame, risk_free_rate=0.0) -> dict` | Gross/net PnL, Sharpe, CAGR, MDD, cost totals |

```python
from quanttradeai.utils.metrics import compute_performance

metrics = compute_performance(backtest_results)
```

`compute_performance` expects `strategy_return` and `equity_curve`. If present, it also reads `gross_return`, `gross_equity_curve`, and `data.attrs["ledger"]`.

## Impact Defaults

| API | Signature | Purpose |
| --- | --- | --- |
| `ImpactConfigError` | `ValueError` subclass | Raised for malformed impact defaults |
| `load_impact_config` | `(config_path="config/impact_config.yaml") -> dict[str, dict]` | Load `asset_classes` impact defaults |
| `merge_execution_with_impact` | `(execution_cfg, impact_defaults, asset_class) -> dict` | Merge asset-class defaults into execution config |

```python
from quanttradeai.utils.impact_loader import (
    load_impact_config,
    merge_execution_with_impact,
)

defaults = load_impact_config("config/impact_config.yaml")
execution = merge_execution_with_impact({}, defaults, "equity")
```

## Project Paths

| Function | Signature | Purpose |
| --- | --- | --- |
| `infer_project_root` | `(config_path: str | Path) -> Path` | Infer project root from a config path |
| `resolve_project_path` | `(config_path: str | Path, candidate: str | Path) -> Path` | Resolve project-relative paths |

```python
from quanttradeai.utils.project_paths import resolve_project_path

model_path = resolve_project_path("config/project.yaml", "models/aapl_model")
```

## Minimal Examples

### Validate and Read Runs

```python
from quanttradeai.utils.config_validator import validate_project_config
from quanttradeai.utils.run_records import discover_runs

validation = validate_project_config("config/project.yaml")
runs = discover_runs()
```

### Scoreboard Automation

```python
from quanttradeai.utils.run_records import discover_runs
from quanttradeai.utils.run_scoreboard import attach_scoreboard, sort_run_records

records = attach_scoreboard(discover_runs())
ranked = sort_run_records(records, sort_by="net_sharpe", ascending=False)
```

## Related CLI/YAML Docs

- [CLI utilities](../cli/utilities.md)
- [CLI runs](../cli/runs.md)
- [Config docs](../config/)
- [Artifacts](../artifacts.md)

The CLI uses these utilities to validate configs, create run directories, attach compact result summaries, rank runs, compare runs, and expand sweeps.

## Common Mistakes

- Treating `utils` helpers as root-level exports; import them from their modules.
- Comparing runs from different families with `build_run_comparison`; it only supports same-family comparisons.
- Passing non-scalar values to sweep override helpers.
- Assuming `validate_project_config` is read-only; it writes resolved config and summary artifacts.
- Calling `compute_performance` on raw price data instead of a backtest result with return/equity columns.
- Forgetting that many utility functions use project-relative paths and default to `runs/`, `config/`, or `reports/`.
