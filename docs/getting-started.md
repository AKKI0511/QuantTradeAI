# Getting Started

> Install the QuantTradeAI plugin, open your coding agent, and ask for the research.

QuantTradeAI is easiest when Claude Code or Codex drives it. You describe the market question. The plugin creates a workspace, runs `uvx`, sets up uv, edits `config/project.yaml`, validates it, runs experiments, and reads the artifacts before recommending anything.

## 1. Install The Plugin

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

See [Agent Plugins](plugins.md) for checks, provider notes, and local marketplace testing.

## 2. Open The Coding Agent

Open Claude Code or Codex in the parent folder where you want the QuantTradeAI workspace.

Do not prebuild the project by hand unless you want the manual path below. Let the plugin turn your prompt into the workspace and commands.

## 3. Ask Naturally

Raw example:

```text
use QuantTradeAI and create a workspace called vibe-lab.
vibe quant research: AAPL/MSFT daily, 2022-2024, RSI mean reversion vs SMA trend, include costs, no live trading.
set up uv, configure the yaml, validate it, run sweeps, inspect the artifacts, and tell me what is least fake plus the next experiment.
```

The agent should do the operational work:

- create the workspace with `uvx quanttradeai init`
- run `uv sync`
- run `uv run quanttradeai doctor`
- edit `config/project.yaml`
- run `uv run quanttradeai validate -c config/project.yaml`
- run research, agent backtests, or sweeps
- inspect `runs/` artifacts before calling a winner

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

| Path | Purpose |
| :--- | :--- |
| `config/project.yaml` | Canonical config for data, features, research, agents, sweeps, risk, and deployment defaults. |
| `pyproject.toml` | Disposable uv project with `quanttradeai==<version>` pinned. |
| `.python-version` | Python version hint for uv-managed environments. |
| `.env.example` | Optional environment variable placeholders; keep real secrets in `.env` or exported env vars. |
| `.gitignore` | Ignores local virtualenvs, secrets, and generated outputs. |
| `AGENTS.md` | Workspace-local guidance that points agents at the installed plugin. |
| `CLAUDE.md` | Claude Code adapter for the same local guidance. |
| `.quanttradeai/workspace.yaml` | Workspace metadata, selected template, package pin, and plugin name. |

The plugin skills are not copied into each workspace. They stay in the installed QuantTradeAI plugin.

## Manual `uvx` And uv Workflow

Use this when you want to drive the CLI yourself.

```bash
uvx quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
uv run quanttradeai validate -c config/project.yaml
uv run quanttradeai agent run --all -c config/project.yaml --mode backtest
```

Lifecycle in plain terms:

- `uvx quanttradeai init my-lab` downloads/runs the published package in a temporary tool environment and writes the workspace.
- `pyproject.toml` in the workspace pins the package version that created it.
- `uv sync` creates `.venv` from that pin.
- `uv run quanttradeai ...` runs the pinned workspace CLI.

Use `uvx` outside the workspace to create it. Use `uv run` inside the workspace so the agent and you are using the same local environment.

To create a workspace from a specific published version:

```bash
uvx quanttradeai@0.1.0 init my-lab
```

To initialize the current directory:

```bash
uvx quanttradeai init .
uv sync
uv run quanttradeai doctor
```

## Templates

Available templates:

- `strategy-lab` default
- `research`
- `rule-agent`
- `model-agent`
- `llm-agent`
- `hybrid`

Example:

```bash
uvx quanttradeai init my-research-lab --template research
```

## Where To Go Next

| Topic | Link |
| :--- | :--- |
| Plugin install and checks | [Agent Plugins](plugins.md) |
| Run artifacts and outputs | [Artifacts](artifacts.md) |
| CLI command reference | [CLI](cli/) |
| Project configuration | [Config](config/) |
| Example patterns | [Examples](examples/) |
| Python API reference | [API](api/) |

## Source And Contributor Setup

Clone the source only when you are contributing to QuantTradeAI itself or testing local changes.

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
make test
```

Create a workspace from the checkout:

```bash
poetry run quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
```

When testing checkout changes from a generated workspace, keep the generated package pin and install the checkout into that workspace environment explicitly:

```bash
uv pip install --reinstall -e ..
```
