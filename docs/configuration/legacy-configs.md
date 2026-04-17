# Legacy Command Migration

QuantTradeAI no longer supports these legacy commands:

- `quanttradeai train`
- `quanttradeai backtest-model`
- `quanttradeai live-trade`
- `quanttradeai validate-config`
- `quanttradeai validate --legacy-config-dir ...`
- `quanttradeai research run --legacy-config-dir ...`

Use the canonical `config/project.yaml` workflow instead.

## Replacements

| Removed command | Replacement |
| --- | --- |
| `quanttradeai train` | `quanttradeai research run -c config/project.yaml` |
| `quanttradeai validate-config` | `quanttradeai validate -c config/project.yaml` |
| `quanttradeai backtest-model` | `quanttradeai init --template model-agent -o config/project.yaml` -> set `agents[].model.path` -> `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest` |
| `quanttradeai live-trade` | `quanttradeai agent run --agent <name> -c config/project.yaml --mode live` |
| `validate --legacy-config-dir ...` | create or commit `config/project.yaml`, then run `quanttradeai validate -c config/project.yaml` |
| `research run --legacy-config-dir ...` | create or commit `config/project.yaml`, then run `quanttradeai research run -c config/project.yaml` |

## Recommended Migration Path

1. Initialize the closest template:
   `quanttradeai init --template research|model-agent|llm-agent|hybrid -o config/project.yaml`
2. Move the settings you still need into `config/project.yaml`.
3. For model agents, point `agents[].model.path` at a stable promoted model directory under `models/promoted/...`.
4. Run `quanttradeai validate -c config/project.yaml`.
5. Use `research run`, `agent run`, `promote`, and `deploy` from that single file.

## Utility Commands That Still Exist

These older utility commands remain because they do not yet have a direct project-based replacement:

- `quanttradeai fetch-data`
- `quanttradeai evaluate`
- `quanttradeai backtest`
