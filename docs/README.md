# QuantTradeAI Docs

> **Give your coding agent a quant research lab, not a blank terminal.**

QuantTradeAI is an agent-native workspace for researching trading strategies. It is built so Claude Code, Codex, Cursor, and similar coding agents can work from a structured lab instead of creating messy one-off scripts for every data pull, backtest, sweep, and comparison.

You give the research objective. The agent uses `project.yaml`, the `quanttradeai` CLI, and machine-readable artifacts to run repeatable experiments, compare strategy variants, and recommend the next step.

<table>
  <tr>
    <td width="50%">
      <a href="getting-started.md"><strong>Getting Started</strong></a><br>
      Set up a QuantTradeAI workspace and hand it to a coding agent.
    </td>
    <td width="50%">
      <a href="plugins.md"><strong>Agent Plugins</strong></a><br>
      Codex and Claude Code marketplace install, checks, and plugin skill behavior.
    </td>
  </tr>
</table>

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
```

The agent should create the workspace, run the CLI and give you an evidence-backed answer.

## Documentation Map

| Area | What it covers |
| :--- | :--- |
| [Getting Started](getting-started.md) | Setup, workspace creation, and agent usage. |
| [Agent Plugins](plugins.md) | Codex and Claude Code marketplace install, checks, and plugin skill behavior. |
| [Artifacts](artifacts.md) | Run outputs, scoreboards, summaries, and recommendation evidence. |
| [CLI](cli/) | Commands, inputs, outputs, and artifacts. |
| [Config](config/) | `project.yaml` and supported data, research, agent, and execution sections. |
| [Examples](examples/) | Agent-native patterns for strategy sweeps and model promotion. |
| [API](api/) | Python API reference for advanced users. |

## Safety Model

> Backtest first. Replay-backed paper next. Live trading and broker-backed execution require explicit human approval.

<div align="center">

### Building with QuantTradeAI?

If it helps your agent research cleaner trading strategies, give the project a star on GitHub:
**[github.com/AKKI0511/QuantTradeAI](https://github.com/AKKI0511/QuantTradeAI)**.

</div>