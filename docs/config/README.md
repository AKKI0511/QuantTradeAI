# Config

QuantTradeAI YAML config files are the control plane for a workspace. They define the data window, feature set, research settings, trading agents, sweeps, risk controls, and deployment target in one place.

The default project file is:

```bash
config/project.yaml
```

Every first-class project command accepts another path with `-c` or `--config`, so a workspace can keep multiple labs side by side:

```bash
config/momentum.yaml
config/mean-reversion.yaml
config/aapl-paper.yaml
```

## Used By

Project configs are used by the main workspace commands:

- `quanttradeai validate`
- `quanttradeai research run`
- `quanttradeai agent run`
- `quanttradeai promote`
- `quanttradeai deploy`

## How Operators Use Config

For humans and coding agents, the loop is intentionally simple:

1. Edit the YAML file.
2. Validate it.
3. Run the relevant CLI command.
4. Inspect the artifacts under `reports/`, `models/`, or `runs/`.

```bash
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai agent run --agent rsi_reversion -c config/project.yaml --mode backtest
```

## Supported Fields

The project config answers the operational questions QuantTradeAI needs before it can run:

| Question | Config area |
|---|---|
| What symbols and dates should be used? | `data` |
| Which features should be generated? | `features` |
| Should a model be trained and promoted? | `research` |
| Which trading agents exist? | `agents` |
| Which parameter grids should run? | `sweeps` |
| What is required for live safety? | `risk`, `position_manager` |
| How should an agent bundle be generated? | `deployment` |

This overview intentionally avoids a giant schema. The pages in this folder document the supported fields by workflow area.

## Expected Outcomes

Successful commands write machine-readable outputs that are meant to be reviewed:

- `reports/config_validation/.../resolved_project_config.yaml`
- `runs/research/.../summary.json`
- `runs/agent/.../metrics.json`
- `runs/agent/batches/.../scoreboard.json`
- `reports/deployments/.../deployment_manifest.json`

Treat the YAML file as intent, and the resolved/runtime artifacts as the exact configuration QuantTradeAI used.

## Common Mistakes

- Editing YAML without running `quanttradeai validate`.
- Reusing a config path without checking where relative prompt, notes, model, and output paths resolve.
- Treating a successful research run as approval for paper or live trading; agent modes and promotion are explicit.
