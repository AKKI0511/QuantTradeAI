<div align="center">

# QuantTradeAI

### Give your coding agent a quant research lab, not a blank terminal.

<p>
  <a href="docs/getting-started.md">Getting Started</a> ·
  <a href="docs/config/project-file.md">Project YAML</a> ·
  <a href="docs/README.md">Docs</a> ·
  <a href="roadmap.md">Roadmap</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/AKKI0511/QuantTradeAI/actions/workflows/ci.yml"><img src="https://github.com/AKKI0511/QuantTradeAI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

</div>

---

QuantTradeAI is built for coding agents — Claude Code, Codex, Cursor, and similar tools — to research trading strategies without repeatedly writing data, backtest, sweep, artifact, and deployment plumbing from scratch.

You give the research objective. The agent uses a generated workspace, `config/project.yaml`, and the `quanttradeai` CLI to run repeatable strategy experiments, compare artifacts, and recommend winners.

---

## Why QuantTradeAI

AI agents can write code, but quant research requires repeatable infrastructure.

Without structure, agents write one-off scripts, scatter outputs across directories, and produce runs that cannot be compared or built on. Every session starts fresh with no memory of what worked.

QuantTradeAI provides:

- **A stable experiment environment** — one project config drives data, features, research, and agents
- **Standardized run artifacts** — every run writes structured outputs agents can read directly
- **A promotion model** — backtest → paper → live, with live execution gated behind explicit human approval

## Before and After

| | Without QuantTradeAI | With QuantTradeAI |
| :--- | :--- | :--- |
| **Setup** | Agent writes custom fetch and backtest scripts each time | Agent reuses the `quanttradeai` CLI against one project config |
| **Outputs** | Scattered across ad-hoc directories | Every run writes to `runs/` with standardized artifacts |
| **Comparison** | No way to compare strategy variants | Scoreboards and `--compare` are built in |
| **Structure** | Research and agent code in separate scripts | One `config/project.yaml` drives both |
| **Safety** | No gate before live execution | Backtest → paper → live, each requiring explicit promotion |

---

## Quickstart

> [!NOTE]
> Package publishing is being stabilized. Until PyPI is available, use the [local development setup](#local-development) below.

```bash
pip install quanttradeai
quanttradeai init my-lab
cd my-lab
```

Open `my-lab` in Claude Code, Cursor, Codex, or another coding agent and ask:

> "Research RSI and SMA crossover strategies on AAPL and MSFT and find the best one."

The agent uses `config/project.yaml` and the `quanttradeai` CLI to run experiments, compare results on a scoreboard, and recommend a winner — without you writing any backtest code.

---

## What `init` Creates

```text
my-lab/
├── config/project.yaml                    # canonical project config — start here
├── AGENTS.md                              # tells coding agents how to use the CLI and YAML
├── CLAUDE.md                              # Claude Code-specific context and guidance
├── .claude/skills/quanttradeai-research/  # skill pack for Claude Code agents
└── .quanttradeai/workspace.yaml          # workspace-level metadata and defaults
```

The coding agent reads these files to understand how to use the framework correctly — you do not need to explain the workflow to it each session.

---

## What the Agent Can Do

- **Create strategy labs** — initialize multi-agent projects from templates (`rule`, `model`, `llm`, `hybrid`)
- **Run parameter sweeps** — expand YAML-defined grids into parallel backtest variants
- **Compare scoreboards** — rank runs by Sharpe ratio, PnL, or other metrics
- **Inspect artifacts** — read `summary.json.run_result` and `scoreboard.json` from any run
- **Promote backtests to paper** — move winning runs forward through explicit gates
- **Generate deployment bundles** — emit local runners, Docker Compose, or Render worker configs
- **Keep live trading gated** — live mode requires human acknowledgement at every promotion step

---

## Artifact-Based Research

Every run writes machine-readable artifacts. Agents read these to decide what to do next — no manual result-parsing required.

| Artifact | What it contains |
| :--- | :--- |
| `summary.json` → `run_result` | High-level outcome: winner, ranked candidates, failures |
| `scoreboard.json` | Ranked metrics across all sweep variants |
| `results.json` | Per-variant metrics for batch and sweep runs |
| `metrics.json` | Full metrics for a single run |
| `resolved_project_config.yaml` | Exact config used for the run — fully reproducible |

---

## Safety Model

Experiments start in backtest. Agents cannot self-promote to live trading.

| Mode | Gate |
| :--- | :--- |
| **Backtest** | Always available — no credentials needed |
| **Paper** | Replay-backed simulation — requires explicit `promote` from a passing backtest |
| **Live** | Requires `--acknowledge-live` from a human, plus validated streaming and risk config |

> [!CAUTION]
> Live mode and broker-backed execution are fully opt-in. QuantTradeAI does not guarantee profitability.

---

## Current Capabilities

| Capability | Status |
| :--- | :---: |
| Strategy labs, sweeps, and research runs | Supported |
| Rule, model, LLM, and hybrid agents | Supported |
| Backtest, paper (replay-backed), and live modes | Supported |
| Scoreboards and run comparison | Supported |
| Research-to-agent promotion pipeline | Supported |
| Deployment bundles (local, Docker Compose, Render) | Supported |

For the full list, see the [docs](docs/README.md).

---

## Local Development

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
```

Initialize a workspace and open it in your coding agent:

```bash
quanttradeai init my-lab
cd my-lab
```

Dev commands:

```bash
make format   # auto-format
make lint     # run linters
make test     # run test suite
```

---

## Documentation

**Get started** — [Getting Started](docs/getting-started.md) · [Docs](docs/README.md)

**Configure** — [Project YAML](docs/config/project-file.md) · [Config Overview](docs/config/)

**Reference** — [API Docs](docs/api/) · [Roadmap](roadmap.md) · [Contributing](CONTRIBUTING.md)

---

## License

MIT. See [LICENSE](LICENSE).
