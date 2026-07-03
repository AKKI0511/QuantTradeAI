# QuantTradeAI Docs

> Tell Claude Code or Codex what quant research you want. Let the QuantTradeAI plugin build and run the lab.

QuantTradeAI is an agent-first research workflow. Install the plugin, open your coding agent, and ask naturally for a workspace plus experiments. The plugin should handle `uvx`, uv setup, YAML edits, validation, runs, scoreboards, and artifact analysis.

## Start Here

Install the plugin:

```bash
# Codex
codex plugin marketplace add AKKI0511/QuantTradeAI --sparse .agents/plugins --sparse plugins
codex plugin add quanttradeai@quanttradeai

# Claude Code
claude plugin marketplace add AKKI0511/QuantTradeAI --sparse .claude-plugin plugins
claude plugin install quanttradeai@quanttradeai
```

Open Claude Code or Codex in the folder where you want the workspace, then prompt it like this:

```text
use QuantTradeAI and create a workspace called vibe-lab.
vibe quant research: AAPL/MSFT, daily bars, RSI mean reversion vs SMA trend, 2022-2024, costs included, no live trading.
set up uv, validate the yaml, run the experiments, inspect the artifacts, and tell me what actually held up.
```

The agent should create the workspace, run the CLI through `uv run`, inspect `runs/`, and give you an evidence-backed answer.

## Documentation Map

| Area | What it covers |
| :--- | :--- |
| [Getting Started](getting-started.md) | Agent-first flow, manual uv workflow, and generated workspace files. |
| [Agent Plugins](plugins.md) | Codex and Claude Code marketplace install, checks, and plugin skill behavior. |
| [Artifacts](artifacts.md) | Run outputs, scoreboards, summaries, and recommendation evidence. |
| [CLI](cli/) | Commands, inputs, outputs, and artifacts. |
| [Config](config/) | `project.yaml` and supported data, research, agent, and execution sections. |
| [Examples](examples/) | Agent-native patterns for strategy sweeps and model promotion. |
| [API](api/) | Python API reference for advanced users. |

## Manual Path

If you are driving the CLI yourself:

```bash
uvx quanttradeai init my-lab
cd my-lab
uv sync
uv run quanttradeai doctor
uv run quanttradeai validate -c config/project.yaml
```

Use `uvx` to create a workspace from the published package. Use `uv run` inside that workspace so commands use the pinned local `.venv`.

## Safety Model

Backtest first. Replay-backed paper next. Live trading and broker-backed execution require explicit human approval.

## Source Work

Clone the repository only for contributor development or local plugin testing. See [Getting Started](getting-started.md#source-and-contributor-setup).
