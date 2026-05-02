# Configuration

QuantTradeAI is centered on one canonical config file: `config/project.yaml`.

Use it for:

- `quanttradeai validate`
- `quanttradeai research run`
- `quanttradeai agent run`
- `quanttradeai promote`
- `quanttradeai deploy`

## Choose the Right File

| What you want to do | Primary file(s) | Used by |
| --- | --- | --- |
| Run the canonical research workflow | `config/project.yaml` | `quanttradeai validate`, `quanttradeai research run` |
| Run or promote a project-defined agent | `config/project.yaml` | `quanttradeai agent run`, `quanttradeai promote` |
| Generate a project-agent deployment bundle | `config/project.yaml` | `quanttradeai deploy` |
| Fetch historical data with the older utility flow | `config/model_config.yaml` | `quanttradeai fetch-data` |
| Evaluate an already-trained model artifact | `config/model_config.yaml` | `quanttradeai evaluate` |
| Run a standalone CSV backtest | `config/backtest_config.yaml` | `quanttradeai backtest` |

## Recommended Reading

- [Project Config (`project.yaml`)](configuration/project-yaml.md)
- [Generated Runtime Files](configuration/live-runtime-files.md)
- [Legacy Command Migration](configuration/legacy-configs.md)

## Typical Workflows

### Research

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

### Agent Backtest

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
```

### Multi-Agent Runs

```bash
poetry run quanttradeai agent run --all -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --all -c config/project.yaml --mode paper
poetry run quanttradeai agent run --all -c config/project.yaml --mode live --acknowledge-live <project_name>
```

Live batches require every configured agent to already have `mode: live` and require the acknowledgement value to match `project.name`.

### Agent Deployment

```bash
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target local
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
poetry run quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target render -o deployments/breakout-render
```

Deployment bundles are generated from `config/project.yaml`. Use `--target local` for a Python runner bundle, `--target docker-compose` for a Compose bundle, or `--target render` for a Render Background Worker Blueprint. Paper bundles disable replay in the emitted deployment config and expect valid real-time provider settings.

## Important Boundaries

- `config/project.yaml` is the center of gravity for research, agents, promotion, and deployment.
- Local `agent run --mode paper` defaults to replay from `data.streaming.replay` in `config/project.yaml`.
- Live agents compile runtime streaming, risk, and position-manager YAML snapshots from `config/project.yaml` into each run directory.
- `fetch-data`, `evaluate`, and standalone `backtest` remain utility commands, but they are not the primary product workflow.
