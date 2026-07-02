# Doctor Command

`quanttradeai doctor` is a read-only health check for coding agents. Run it
after `quanttradeai init` and `uv sync`, before asking an agent to run research,
paper, or live workflows.

## Syntax

```bash
quanttradeai doctor
quanttradeai doctor -c config/project.yaml
quanttradeai doctor --json
```

## What It Checks

- Workspace files: `config/project.yaml`, `pyproject.toml`, and generated metadata.
- Python and uv environment readiness.
- Installed QuantTradeAI version and the workspace `quanttradeai==<version>` pin.
- Project dependency and `.quanttradeai/workspace.yaml` consistency.
- Project YAML validity using the same read-only validation rules as runtime commands.
- Required symbols, features, profiles, and enabled workflows.
- Output path writability for `data/`, `models/`, `reports/`, and `runs/` without creating files.
- Credentials needed by configured LLM or Alpaca-backed agents.
- Paper/live safety defaults, including replay-backed paper mode and live risk settings.

## Exit Codes

| Code | Meaning |
|---:|---|
| `0` | No error diagnostics. Warnings may be present. |
| `1` | One or more health-check errors were found. |
| `2` | CLI usage error from Typer, such as an invalid option. |

## JSON Output

Use `--json` when another agent or script needs deterministic diagnostics:

```bash
quanttradeai doctor --json
```

The payload includes:

- `status`: `ok` or `error`
- `exit_code`: deterministic process exit code
- `summary`: check, error, and warning counts
- `checks`: stable high-level check statuses
- `diagnostics`: actionable items with `severity`, `code`, `message`, `action`, and optional `path`

`doctor` does not write validation reports, create output directories, modify
`.env`, or change project files.
