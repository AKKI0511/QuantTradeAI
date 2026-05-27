# QuantTradeAI Docs

### Give your coding agent a quant research lab, not a blank terminal.

<p>
  <a href="getting-started.md"><strong>Getting Started</strong></a> &middot;
  <a href="artifacts.md"><strong>Artifacts</strong></a> &middot;
  <a href="cli/"><strong>CLI</strong></a> &middot;
  <a href="config/"><strong>Config</strong></a> &middot;
  <a href="api/"><strong>API</strong></a> &middot;
  <a href="examples/"><strong>Examples</strong></a>
</p>

---

QuantTradeAI is an agent-native workspace for researching trading strategies. It is built so Claude Code, Codex, Cursor, and similar coding agents can work from a structured lab instead of creating messy one-off scripts for every data pull, backtest, sweep, and comparison.

You give the research objective. The agent uses `config/project.yaml`, the `quanttradeai` CLI, and machine-readable artifacts to run repeatable experiments, compare strategy variants, and recommend the next step.

## Start here

<table>
  <tr>
    <td width="50%">
      <a href="getting-started.md"><strong>Getting Started</strong></a><br>
      Set up a QuantTradeAI workspace and hand it to a coding agent.
    </td>
    <td width="50%">
      <a href="artifacts.md"><strong>Artifacts</strong></a><br>
      Understand the outputs agents use to compare runs and explain recommendations.
    </td>
  </tr>
</table>

## Documentation map

| Area | What it covers |
| :--- | :--- |
| [Getting Started](getting-started.md) | Setup, workspace creation, and agent usage. |
| [Artifacts](artifacts.md) | Run outputs, scoreboards, summaries, and recommendation evidence. |
| [CLI](cli/) | Commands, when to use them, inputs, outputs, and artifacts. |
| [Config](config/) | `project.yaml` and supported data, research, agent, and execution sections. |
| [API](api/) | Python API reference for advanced users. |
| [Examples](examples/) | Agent-native working patterns for strategy sweeps and model promotion. |

## Safety model

> Backtest first. Replay-backed paper next. Live trading and broker-backed execution require explicit human approval.

<div align="center">

### Building with QuantTradeAI?

If it helps your agent research cleaner trading strategies, give the project a star on GitHub:
**[github.com/AKKI0511/QuantTradeAI](https://github.com/AKKI0511/QuantTradeAI)**.

</div>
