# Promotion And Deployment

Promotion and deployment are separate steps:

- `quanttradeai promote` uses successful run records to update stable model artifacts or project config state.
- `quanttradeai deploy` generates a runnable deployment bundle for a configured agent.

A deployment bundle is not the same as live trading. It packages a command and resolved config so an operator can run the agent in the selected mode.

## Promotion

`quanttradeai promote` promotes a successful research or agent run through the current project workflow.

### When To Use

Use promotion after inspecting run artifacts and deciding a candidate is ready for the next stage:

- research run -> stable promoted model directories
- agent backtest run -> paper-ready agent config
- agent paper run -> live-ready agent config, with an explicit live acknowledgement
- agent backtest sweep run -> materialize the winning sweep parameters into the base agent and paper config

### Syntax

```bash
quanttradeai promote --run RUN_ID [OPTIONS]
```

```bash
quanttradeai promote --run research/20260525_120000_research_lab
quanttradeai promote --run agent/backtest/20260525_120000_rsi_reversion --to paper
quanttradeai promote --run agent/paper/20260525_130000_rsi_reversion --to live --acknowledge-live rsi_reversion
quanttradeai promote --run agent/backtest/20260525_120000_rsi_grid_winner --dry-run
```

### Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `--run TEXT` | none | Yes | Run id to promote, such as `research/<run>` or `agent/backtest/<run>`. A path under `runs/` is also normalized by the implementation. |
| `-c, --config TEXT` | `config/project.yaml` | No | Path to the project config YAML to read or update. |
| `--to TEXT` | `paper` | No | Target mode. Supported values: `paper`, `live`. |
| `--dry-run` | `false` | No | Show the proposed promotion result without writing `project.yaml` or promoted model directories. |
| `--acknowledge-live TEXT` | `None` | Live only | Required for `--to live`; value must exactly match the agent name being promoted. |
| `--apply-sweep` | `false` | No | Accepted for sweep-generated agent backtest runs. Sweep backtest runs materialize automatically, so this flag is not normally needed. |

### Reads

Promotion reads:

```text
runs/**/summary.json
config/project.yaml
```

Research promotion also reads the research run's `artifacts.experiment_dir`.

Agent promotion reads the selected agent and deployment sections from the project config.

### Writes / Artifacts

| Promotion path | Writes |
|---|---|
| Research -> promoted models | Copies trained symbol model directories into `research.promotion.targets[].path`, and writes `promotion_manifest.json` inside each destination. |
| Agent backtest -> paper | Updates the selected agent's `mode` to `paper` and `deployment.mode` to `paper` when needed. |
| Agent paper -> live | Updates the selected agent's `mode` to `live` when prerequisites pass. |
| Agent sweep backtest -> paper | Applies the sweep variant's scalar parameters to the base agent, sets the agent to `paper`, and sets `deployment.mode` to `paper` when needed. |

With `--dry-run`, the result JSON reports the proposed change without committing it.

### Live Approval Boundary

Live promotion requires:

```bash
--to live --acknowledge-live <agent_name>
```

The acknowledgement must exactly match the agent name being promoted. The project must also have streaming enabled, top-level live risk config, and position-manager config available for validation.

### Expected Output

Promotion prints JSON with fields such as:

- `status`
- `source_run_id`
- `run_type`
- `agent_name` or `project_name`
- `from_mode`
- `to_mode`
- `changed`
- `changed_fields` for sweep materialization
- `promoted_targets` for research promotion
- `next_command` for agent promotions

### Common Mistakes

| Mistake | Why it fails or causes confusion |
|---|---|
| Promoting a failed run | Only successful promotable runs are accepted. |
| Promoting an agent backtest directly to live | Agent live promotion requires a successful paper run first. |
| Using `--to live` for research promotion | Research promotion supports the default model promotion behavior only. |
| Missing `--acknowledge-live` | Live agent promotion requires an exact agent-name acknowledgement. |
| Promoting before inspecting artifacts | Promotion trusts the selected successful run; inspect `metrics.json`, `summary.json`, and comparison output first. |
| Using `--apply-sweep` on a non-sweep run | The flag is accepted only when the run summary contains sweep metadata. |

## Deployment

`quanttradeai deploy` generates a deployment bundle for one project-defined agent.

### When To Use

Use deployment after the agent config is ready for `paper` or `live` mode and you want a runnable local, Docker Compose, or Render bundle.

### Syntax

```bash
quanttradeai deploy --agent AGENT_NAME [OPTIONS]
```

```bash
quanttradeai deploy --agent rsi_reversion
quanttradeai deploy --agent rsi_reversion --target local --mode paper
quanttradeai deploy --agent rsi_reversion --target docker-compose --mode live
quanttradeai deploy --agent rsi_reversion --target render -o reports/deployments/rsi-render
```

### Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `--agent TEXT` | none | Yes | Agent name from the project config. |
| `-c, --config TEXT` | `config/project.yaml` | No | Path to the project config YAML. |
| `--target TEXT` | `deployment.target` or `docker-compose` | No | Deployment target. Supported values: `local`, `docker-compose`, `render`. |
| `--mode TEXT` | `deployment.mode` or `paper` | No | Deployment mode. Supported values: `paper`, `live`. |
| `-o, --output TEXT` | `reports/deployments/<agent>/<timestamp>/` | No | Directory for the generated bundle. |
| `--force` | `false` | No | Overwrite generated deployment files in an existing non-empty bundle directory. |

### Reads

Deployment reads and validates:

```text
config/project.yaml
```

Depending on target and agent kind, it can also read:

- prompt files under `prompts/`
- model directories under `models/`
- notes files under `notes/`
- model signal source directories

### Target Behavior

| Target | Bundle behavior |
|---|---|
| `local` | Writes `run.py`, `.env.example`, `resolved_project_config.yaml`, `README.md`, and `deployment_manifest.json`. Local bundle config uses absolute paths for referenced project assets. |
| `docker-compose` | Writes `docker-compose.yml`, `Dockerfile`, `.env.example`, `resolved_project_config.yaml`, `README.md`, and `deployment_manifest.json`. It mounts project `data/`, `runs/`, and `reports/`; prompts/models are mounted when needed. |
| `render` | Writes `render.yaml`, `Dockerfile`, `.env.example`, `resolved_project_config.yaml`, `README.md`, `deployment_manifest.json`, and an `assets/` directory with selected prompt/model/notes assets. |

Paper deployment bundles always require real-time streaming. If the project config has replay enabled, replay is disabled in the generated deployment config and a warning is returned.

Live deployment bundles require the selected agent to already be configured with `mode: live`.

### Writes / Artifacts

By default, deployment writes:

```text
reports/deployments/<agent>/<timestamp>/
```

Every target writes:

```text
deployment_manifest.json
resolved_project_config.yaml
README.md
.env.example
```

Target-specific files include:

```text
local:          run.py
docker-compose: docker-compose.yml, Dockerfile
render:         render.yaml, Dockerfile, assets/
```

### Expected Output

The command prints JSON with:

- `status`
- `agent_name`
- `target`
- `mode`
- `output_dir`
- `artifacts`
- `warnings`
- `next_command`

### Common Mistakes

| Mistake | Why it matters |
|---|---|
| Treating bundle generation as starting live trading | `deploy` only writes files; it does not run the agent. |
| Building a live bundle before promoting/configuring the agent as live | Live bundles require the selected agent to have `mode: live`. |
| Reusing a non-empty output directory without `--force` | The command refuses to overwrite bundle files by default. |
| Expecting replay-backed paper deployment | Paper deployment bundles force real-time streaming. |
| Using paths outside expected project asset directories for Docker Compose or Render | Those targets validate prompt/model/notes paths so the bundle can mount or copy assets predictably. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
