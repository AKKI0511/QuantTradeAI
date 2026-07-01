# QuantTradeAI CLI

The QuantTradeAI CLI is the operating surface for the product. It turns project YAML into validated experiments, run records, artifacts, promotions, and deployment bundles.

It is designed for two users at the same time:

- **Humans** who need a small set of commands with clear outcomes.
- **Coding agents** who need stable commands that read config, write artifacts, and return machine-readable summaries.

## Mental Model

```text
init -> doctor -> validate -> run -> inspect artifacts -> promote/deploy
```

Start with a generated workspace, run `doctor` to catch setup issues, edit `config/project.yaml`, validate it, run research or agents, inspect the `runs/` artifacts, then promote or package the result when it is ready.

## Command Families

QuantTradeAI keeps the command surface intentionally compact:

- **Workspace** commands create the local project structure.
- **Doctor** checks run a read-only workspace health check with stable text and JSON output.
- **Validation** commands check the canonical project YAML and emit resolved config artifacts.
- **Research** commands train and evaluate model workflows.
- **Agent** commands run YAML-defined trading agents in `backtest`, `paper`, or `live` mode.
- **Runs** commands list, score, and compare recorded runs.
- **Promotion and deployment** commands move successful runs toward paper/live operation or generate runnable bundles.
- **Utilities** cover lower-level data, evaluation, and CSV backtest tasks.

Use `quanttradeai --help` for the current command tree, and use each command's `--help` when an option needs to be verified directly from the installed package.

## Product Flow

The CLI assumes `config/project.yaml` is the main project entrypoint. Most commands read that file, resolve it through the validator, and write durable outputs under `runs/`, `reports/`, or `models/`.

For config structure, use [`docs/config/`](../config/). For output files and run artifacts, use [`docs/artifacts.md`](../artifacts.md).
