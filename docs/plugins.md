# Agent Plugins

> Install QuantTradeAI in Claude Code or Codex, then ask the agent to run the lab.

The QuantTradeAI plugin lives at `plugins/quanttradeai/` and ships one shared `skills/` folder for both providers:

- `workspace-onboarding` - first-run workspace creation, `uvx`, uv setup, `doctor`, validation, and handoff into research.
- `strategy-experiments` - YAML-first strategy backtests, sweeps, scoreboards, and artifact-backed recommendations.
- `model-research` - ML research runs, research sweeps, promotion to stable model paths, and model-agent backtests.
- `run-analysis` - completed run inspection, scoreboard comparison, failure analysis, and evidence summaries.
- `promote-deploy` - promotion, paper checks, deployment bundles, and live-trading safety gates.

Install the plugin first. Then open a fresh agent session and ask for the full QuantTradeAI workflow in normal language.

## Codex

Codex reads this repository's marketplace from `.agents/plugins/marketplace.json`. The marketplace points to `./plugins/quanttradeai`, whose Codex manifest is `plugins/quanttradeai/.codex-plugin/plugin.json`.

Install from the GitHub marketplace source:

```bash
codex plugin marketplace add AKKI0511/QuantTradeAI --sparse .agents/plugins --sparse plugins
codex plugin add quanttradeai@quanttradeai
```

Useful checks:

```bash
codex plugin marketplace list
codex plugin list --available --json
```

In the Codex app, you can also open Plugins, choose the **QuantTradeAI Plugins** marketplace, and install **QuantTradeAI**.

## Claude Code

Claude Code reads this repository's marketplace from `.claude-plugin/marketplace.json`. The marketplace points to `./plugins/quanttradeai`, whose Claude manifest is `plugins/quanttradeai/.claude-plugin/plugin.json`.

Install from the GitHub marketplace source:

```bash
claude plugin marketplace add AKKI0511/QuantTradeAI --sparse .claude-plugin plugins
claude plugin install quanttradeai@quanttradeai
```

The same flow works inside Claude Code:

```text
/plugin marketplace add AKKI0511/QuantTradeAI --sparse .claude-plugin plugins
/plugin install quanttradeai@quanttradeai
```

Useful checks:

```bash
claude plugin marketplace list
claude plugin validate ./plugins/quanttradeai
```

Run `/help` in Claude Code after installation to confirm the `quanttradeai` namespaced skills are available.

## First Prompt

Open Claude Code or Codex in the folder where the workspace should be created and ask for the whole run:

```text
use QuantTradeAI and make a workspace called vibe-lab.
vibe quant research: AAPL/MSFT daily bars, RSI mean reversion vs SMA crossover, 2022-2024, realistic costs, no live trading.
handle uvx, uv sync, yaml config, validation, sweeps, scoreboard, artifact analysis, and tell me what held up.
```

The plugin should guide the agent to:

- run `uvx quanttradeai init <workspace>`
- run `uv sync`
- run `uv run quanttradeai doctor`
- edit `config/project.yaml`
- validate with `uv run quanttradeai validate -c config/project.yaml`
- run research or agent workflows through `uv run quanttradeai ...`
- inspect durable artifacts under `runs/` before recommending a candidate

## What The Manifests Expose

| Provider | Marketplace | Plugin manifest | Skills path |
| :--- | :--- | :--- | :--- |
| Codex | `.agents/plugins/marketplace.json` | `plugins/quanttradeai/.codex-plugin/plugin.json` | `plugins/quanttradeai/skills/` |
| Claude Code | `.claude-plugin/marketplace.json` | `plugins/quanttradeai/.claude-plugin/plugin.json` | `plugins/quanttradeai/skills/` |

Both marketplaces use the plugin name `quanttradeai`. The Codex marketplace display name is **QuantTradeAI Plugins**.

## Local Marketplace Testing

Clone the source only when you are developing the plugin or testing a local marketplace checkout.

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
```

Codex local marketplace:

```bash
codex plugin marketplace add .
codex plugin add quanttradeai@quanttradeai
```

Claude Code local marketplace:

```bash
claude plugin marketplace add .
claude plugin install quanttradeai@quanttradeai
claude plugin validate ./plugins/quanttradeai
```

## Provider References

- Codex plugin docs: <https://developers.openai.com/codex/plugins>
- Codex plugin build docs: <https://developers.openai.com/codex/plugins/build>
- Claude Code marketplace docs: <https://code.claude.com/docs/en/plugin-marketplaces>
- Claude Code plugin reference: <https://code.claude.com/docs/en/plugins-reference>
