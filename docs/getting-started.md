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
quanttradeai init my-lab
cd my-lab
```

To initialize the current directory instead:

```bash
quanttradeai init
```

## What `init` Creates

```text
config/project.yaml
AGENTS.md
CLAUDE.md
.claude/skills/quanttradeai-research/
.quanttradeai/workspace.yaml
```

| Path | Purpose |
| :--- | :--- |
| `config/project.yaml` | Canonical project config for data, features, research settings, agents, sweeps, and execution defaults. |
| `AGENTS.md` | General instructions for coding agents working inside the workspace. |
| `CLAUDE.md` | Claude Code-specific context and operating guidance. |
| `.claude/skills/quanttradeai-research/` | Claude skill files that teach Claude Code how to run QuantTradeAI research tasks. |
| `.quanttradeai/workspace.yaml` | Workspace metadata, including the selected template and init version. |

## Use With A Coding Agent

Open the workspace folder in Claude Code, Cursor, Codex, or a similar coding agent.

Give the agent a natural request, for example:

> "Research RSI and SMA crossover strategies on AAPL/MSFT and find the best one."

The agent should use `AGENTS.md`, `CLAUDE.md`, the Claude skill, `config/project.yaml`, and the `quanttradeai` CLI instead of creating one-off scripts.

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
```

Open the generated `my-lab` folder in your coding agent.

## Where To Go Next

| Topic | Link |
| :--- | :--- |
| Run artifacts and outputs | [Artifacts](artifacts.md) |
| CLI command reference | [CLI](cli/) |
| Project configuration | [Config](config/) |
| Example patterns | [Examples](examples/) |
| Python API reference | [API](api/) |
