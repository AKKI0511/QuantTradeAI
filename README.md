<div align="center">

# QuantTradeAI

### Give your coding agent a quant research lab, not a blank terminal.

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

QuantTradeAI is built for coding agents like Claude Code, Codex, Cursor, and similar tools to research trading strategies without repeatedly writing data, backtest, sweep, artifact, and deployment plumbing from scratch.

You give the research objective. The agent uses a generated workspace, `project.yaml`, and the `quanttradeai` CLI to run repeatable strategy experiments, compare artifacts, and recommend winners.

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
```

The agent should create the workspace, run the CLI and give you an evidence-backed answer.

More install detail: [Agent Plugins](docs/plugins.md). Full walkthrough: [Getting Started](docs/getting-started.md).

---

## Why QuantTradeAI

AI agents can write code, but quant research requires repeatable infrastructure.

Without structure, agents write one-off scripts, scatter outputs across directories, and produce runs that cannot be compared or built on. Every session starts fresh with no memory of what worked.

QuantTradeAI provides:

- **A stable experiment environment** — one project config drives data, features, research, and agents
- **Standard run artifacts** - every run writes structured outputs agents can read directly
- **A promotion path** - backtest -> paper -> live, with live execution gated behind explicit human approval

## Before and After

| | Without QuantTradeAI | With QuantTradeAI |
| :--- | :--- | :--- |
| **Setup** | Agent writes custom fetch and backtest scripts each time | Agent reuses the `quanttradeai` CLI against one project config |
| **Outputs** | Scattered across ad-hoc directories | Every run writes to `runs/` with standardized artifacts |
| **Comparison** | No way to compare strategy variants | Scoreboards and `--compare` are built in |
| **Structure** | Research and agent code in separate scripts | One `project.yaml` drives both |
| **Safety** | No gate before live execution | Backtest → paper → live, each requiring explicit promotion |

## What the Agent Can Do

- **Create strategy labs** — initialize multi-agent projects from templates (`rule`, `model`, `llm`, `hybrid`)
- **Run parameter sweeps** — expand YAML-defined grids into parallel backtest variants
- **Compare scoreboards** — rank runs by Sharpe ratio, PnL, or other metrics
- **Inspect artifacts** — read `summary.json.run_result` and `scoreboard.json` from any run
- **Promote backtests to paper** — move winning runs forward through explicit gates
- **Generate deployment bundles** — emit local runners, Docker Compose, or Render worker configs
- **Keep live trading gated** — live mode requires human acknowledgement at every promotion step

---

## Quickstart: Drive the CLI Yourself

```bash
uvx quanttradeai init my-lab
cd my-lab
uv sync
```

To pin a published release explicitly:

```bash
uvx quanttradeai@0.1.0 init my-lab
```

## What `init` Creates

```text
my-lab/
|-- config/project.yaml             # canonical project config
|-- pyproject.toml
|-- .python-version
|-- .env.example
|-- .gitignore
|-- AGENTS.md                       # minimal workspace-local guidance
|-- CLAUDE.md                       # Claude adapter for the same guidance
`-- .quanttradeai/workspace.yaml     # workspace metadata
```

Reusable workflow skills come from the installed QuantTradeAI plugin. Generated workspaces only carry local context and the project configuration.

---

## Artifact-Based Research

Every run writes durable artifacts the agent can inspect before recommending anything.

| Artifact | What it contains |
| :--- | :--- |
| `summary.json` -> `run_result` | High-level outcome: winner, ranked candidates, failures |
| `scoreboard.json` | Ranked metrics across sweep or batch variants |
| `results.json` | Per-variant metrics for batch and sweep runs |
| `metrics.json` | Full metrics for a single run |
| `resolved_project_config.yaml` | Exact config used for the run - Fully reproducible |

## Safety Model

Experiments start in backtest. Agents cannot self-promote to live trading.

| Mode | Gate |
| :--- | :--- |
| **Backtest** | Always available; no broker credentials needed. |
| **Paper** | Replay-backed simulation; promote from a passing backtest first. |
| **Live** | Requires explicit human acknowledgement plus validated streaming and risk config. |

> [!CAUTION]
> Live mode and broker-backed execution are opt-in. QuantTradeAI does not guarantee profitability.

---

## Local Development

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
```

Create a workspace:

```bash
poetry run quanttradeai init my-lab
cd my-lab
uv sync
```

When testing changes inside a generated workspace, keep the generated package pin and install the checkout into that workspace environment explicitly:

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

## Documentation

**Start** - [Getting Started](docs/getting-started.md) | [Agent Plugins](docs/plugins.md) | [Docs Home](docs/README.md)

**Configure** - [Project YAML](docs/config/project-file.md) | [Config Overview](docs/config/)

**Reference** - [CLI](docs/cli/) | [Artifacts](docs/artifacts.md) | [API Docs](docs/api/) | [Roadmap](roadmap.md)

---

## License

MIT. See [LICENSE](LICENSE).
