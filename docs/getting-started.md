# Getting Started

> **Create an agent-ready QuantTradeAI workspace.**

This page helps you create a QuantTradeAI workspace that is ready for both humans and AI coding agents. The workspace gives the agent a project config, local instructions, and the `quanttradeai` CLI entrypoint.

## Install

```bash
pip install quanttradeai
```

> [!NOTE]
> Package publishing is being stabilized. Until PyPI is available, use the local development setup below.

## Optional Agent Plugin

For Codex or Claude Code, install the QuantTradeAI agent plugin before opening a workspace. The plugin packages the reusable strategy experiment, model research, run analysis, and promotion/deployment skills from this repository.

```bash
git clone -b release/v0.1.0 https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
```

Codex:

```bash
codex plugin marketplace add .
```

Claude Code:

```bash
claude plugin marketplace add .
claude plugin install quanttradeai@quanttradeai
```

See [Agent Plugins](plugins.md) for provider-specific install, validation, and usage notes.

## Create A Workspace

```bash
uvx quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
```

To initialize the current directory instead:

```bash
quanttradeai init
```

## What `init` Creates

```text
config/project.yaml
pyproject.toml
.python-version
.env.example
.gitignore
AGENTS.md
CLAUDE.md
.quanttradeai/workspace.yaml
```

| Path | Purpose |
| :--- | :--- |
| `config/project.yaml` | Canonical project config for data, features, research settings, agents, sweeps, and execution defaults. |
| `pyproject.toml` | Disposable uv project metadata with QuantTradeAI pinned to the released package version. |
| `.python-version` | Python version hint for uv-managed environments. |
| `.env.example` | Optional environment variable placeholders; copy to `.env` only for local secrets. |
| `.gitignore` | Ignores local virtualenvs, secrets, and generated run/output directories. |
| `AGENTS.md` | Minimal workspace-local instructions that point agents to the global plugin. |
| `CLAUDE.md` | Claude Code adapter for the same workspace-local guidance. |
| `.quanttradeai/workspace.yaml` | Workspace metadata, including the selected template and init version. |

## Use With A Coding Agent

Open the workspace folder in Claude Code, Cursor, Codex, or a similar coding agent.

Give the agent a natural request, for example:

> "Research RSI and SMA crossover strategies on AAPL/MSFT and find the best one."

The agent should use `AGENTS.md`, `CLAUDE.md`, the globally installed QuantTradeAI plugin, `config/project.yaml`, and `uv run quanttradeai ...` commands instead of creating one-off scripts.

## First Manual Check

This is optional, but useful if you want to confirm the CLI is available:

```bash
quanttradeai --help
```

Validation happens later when you or the agent starts editing or running the project.

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
quanttradeai init my-research-lab --template research
```

## Local Development Setup

<table>
  <tr>
    <td><strong>Temporary/dev path</strong><br>Use this until package installation from PyPI is available.</td>
  </tr>
</table>

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
poetry run quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
```

Open the generated `my-lab` folder in your coding agent.

Use `uvx quanttradeai@<version> init my-lab` when you need a workspace generated
from a specific published package version. To test local source changes from
that generated workspace, keep the generated `quanttradeai==<version>` pin and
install the checkout into `.venv` explicitly:

```bash
uv pip install --reinstall -e ..
```

## Where To Go Next

| Topic | Link |
| :--- | :--- |
| Run artifacts and outputs | [Artifacts](artifacts.md) |
| CLI command reference | [CLI](cli/) |
| Project configuration | [Config](config/) |
| Example patterns | [Examples](examples/) |
| Python API reference | [API](api/) |
