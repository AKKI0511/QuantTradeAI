# Examples

These examples are working patterns for using QuantTradeAI with a coding agent such as Claude Code or Codex. They are not deep tutorials and they do not replace the CLI, config, or artifact reference pages.

Start with [Strategy Lab Sweep](strategy-lab-sweep.md) if you want the quickest realistic workflow: define rule-based strategies in YAML, run backtest sweeps, and have the agent recommend a candidate from artifacts.

Use [Research To Model Agent](research-to-model-agent.md) when you want the agent to train historical models, promote a successful model artifact, and then run a model-agent backtest.

> [!IMPORTANT]
> These examples are research workflows, not trading advice. Results depend on the data provider, date window, costs, and risk settings.

| Example | Use when | Primary output |
| --- | --- | --- |
| [Strategy Lab Sweep](strategy-lab-sweep.md) | You want to compare RSI and SMA rule variants across liquid equities. | Ranked agent sweep batches under `runs/agent/batches/...`. |
| [Research To Model Agent](research-to-model-agent.md) | You want a trained classifier promoted into a stable model-agent path. | Promoted model artifacts plus a model-agent backtest run. |
