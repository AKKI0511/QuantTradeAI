<div align="center">

# QuantTradeAI

### Tell your coding agent to run quant research. Let the plugin do the plumbing.

<p>
  <a href="docs/getting-started.md">Getting Started</a> |
  <a href="docs/plugins.md">Agent Plugins</a> |
  <a href="docs/config/project-file.md">Project YAML</a> |
  <a href="docs/README.md">Docs</a> |
  <a href="roadmap.md">Roadmap</a> |
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/AKKI0511/QuantTradeAI/actions/workflows/ci.yml"><img src="https://github.com/AKKI0511/QuantTradeAI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

</div>

---

QuantTradeAI is built for Claude Code and Codex. Install the plugin, open your agent, and describe the quant research you want in plain English.

The plugin handles the boring parts: workspace creation, `uvx`, uv setup, `config/project.yaml`, validation, experiments, scoreboards, and artifact analysis. You stay at the research-intent level.

---

## Fast Path: Agent First

Install the QuantTradeAI plugin in your coding agent.

Codex:

```bash
codex plugin marketplace add AKKI0511/QuantTradeAI --sparse .agents/plugins --sparse plugins
codex plugin add quanttradeai@quanttradeai
```

Claude Code:

```bash
claude plugin marketplace add AKKI0511/QuantTradeAI --sparse .claude-plugin plugins
claude plugin install quanttradeai@quanttradeai
```

Then open a fresh Claude Code or Codex session in the folder where you want the lab to live and ask for the whole thing:

```text
use QuantTradeAI and make me a workspace called vibe-lab.
vibe quant research: test AAPL/MSFT daily momentum vs mean reversion, RSI and SMA crossover stuff, 2022-2024, real costs, no live trading.
set up uv, edit the yaml, validate it, run the sweeps, read the artifacts, and tell me what looks least fake plus what to try next.
```

The agent should create the workspace with `uvx quanttradeai init`, run `uv sync`, check `uv run quanttradeai doctor`, edit `config/project.yaml`, validate, run backtest/research workflows, inspect `runs/`, and report evidence instead of vibes.

More install detail: [Agent Plugins](docs/plugins.md). Full walkthrough: [Getting Started](docs/getting-started.md).

---

## Why QuantTradeAI

AI agents can write code, but quant research needs repeatable structure.

Without structure, agents write one-off scripts, scatter outputs across folders, and produce runs that cannot be compared. Every session starts from scratch.

QuantTradeAI provides:

- **One project config** - data, features, research, agents, sweeps, risk, and deployment live in `config/project.yaml`.
- **Standard run artifacts** - every run writes machine-readable outputs under `runs/`.
- **Agent-readable scoreboards** - sweeps and batches can be ranked without scraping terminal text.
- **A promotion path** - backtest -> paper -> live, with live execution gated behind explicit human approval.

## What The Agent Can Do

- Create a strategy lab from templates.
- Edit YAML instead of inventing one-off scripts.
- Run model research, agent backtests, and parameter sweeps.
- Compare scoreboards by Sharpe, PnL, drawdown, activity, warnings, and failures.
- Inspect `summary.json`, `scoreboard.json`, `results.json`, `metrics.json`, and `resolved_project_config.yaml`.
- Promote selected research/backtest results only when asked.
- Generate local, Docker Compose, or Render deployment bundles.
- Keep live trading behind explicit approval.

---

## Manual `uvx` And uv Workflow

Use this when you want to drive the CLI yourself instead of asking the plugin to do it.

```bash
uvx quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
uv run quanttradeai validate -c config/project.yaml
uv run quanttradeai agent run --all -c config/project.yaml --mode backtest
```

Simple lifecycle:

- `uvx quanttradeai init my-lab` runs the published package in a temporary tool environment and creates the workspace.
- The generated `pyproject.toml` pins `quanttradeai==<version>` for that workspace.
- `uv sync` creates `.venv` from the workspace pin.
- `uv run quanttradeai ...` runs the pinned workspace CLI.

Use `uvx` to create or regenerate a workspace. Use `uv run` once you are inside that workspace.

To pin a published release explicitly:

```bash
uvx quanttradeai@0.1.0 init my-lab
```

## What `init` Creates

```text
my-lab/
|-- config/project.yaml
|-- pyproject.toml
|-- .python-version
|-- .env.example
|-- .gitignore
|-- AGENTS.md
|-- CLAUDE.md
`-- .quanttradeai/workspace.yaml
```

Reusable workflow skills come from the installed QuantTradeAI plugin. Generated workspaces only carry local context and the uv project files.

---

## Artifact-Based Research

Every run writes durable artifacts the agent can inspect before recommending anything.

| Artifact | What it contains |
| :--- | :--- |
| `summary.json` -> `run_result` | High-level outcome, ranked candidates, failures, and warnings. |
| `scoreboard.json` | Ranked metrics across sweep or batch variants. |
| `results.json` | Child run IDs, statuses, parameters, variant configs, and failures. |
| `metrics.json` | Full metrics for a single run. |
| `resolved_project_config.yaml` | Exact config used for the run. |

## Safety Model

Experiments start in backtest. Agents cannot quietly jump to live trading.

| Mode | Gate |
| :--- | :--- |
| **Backtest** | Always available; no broker credentials needed. |
| **Paper** | Replay-backed simulation; promote from a passing backtest first. |
| **Live** | Requires explicit human acknowledgement plus validated streaming and risk config. |

> [!CAUTION]
> Live mode and broker-backed execution are opt-in. QuantTradeAI does not guarantee profitability.

---

## Documentation

**Start** - [Getting Started](docs/getting-started.md) | [Agent Plugins](docs/plugins.md) | [Docs Home](docs/README.md)

**Configure** - [Project YAML](docs/config/project-file.md) | [Config Overview](docs/config/)

**Reference** - [CLI](docs/cli/) | [Artifacts](docs/artifacts.md) | [API Docs](docs/api/) | [Roadmap](roadmap.md)

---

## Contributor Setup

Clone the source only when you are contributing to QuantTradeAI itself or testing local package changes.

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
make test
```

Create a test workspace from the checkout:

```bash
poetry run quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
```

When testing checkout changes inside a generated workspace, keep the generated package pin and install the checkout into that workspace environment explicitly:

```bash
uv pip install --reinstall -e ..
```

Dev commands:

```bash
make format
make lint
make test
```

---

## License

MIT. See [LICENSE](LICENSE).
