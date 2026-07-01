# Agent Plugins

> Install the QuantTradeAI skills package for Codex or Claude Code.

The QuantTradeAI plugin lives at `plugins/quanttradeai/` and uses one shared `skills/` folder for both providers:

- `strategy-experiments`: run YAML-first strategy backtests and sweeps.
- `model-research`: train, evaluate, promote, and backtest model artifacts.
- `run-analysis`: inspect completed runs, scoreboards, and artifacts.
- `promote-deploy`: move selected results toward paper checks or deployment bundles with live-trading safety gates.

## Initial Release Branch

Until this distribution is merged into `main`, install from the release integration branch:

```bash
git clone -b release/v0.1.0 https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
```

## Codex

Codex reads repo marketplaces from `.agents/plugins/marketplace.json`. This repository exposes the plugin through that file and points it at `./plugins/quanttradeai`.

Install from a local clone:

```bash
codex plugin marketplace add .
```

Then restart Codex, open Plugins, choose **QuantTradeAI Plugins**, and install **QuantTradeAI**.

For a direct Git-backed install from the release branch:

```bash
codex plugin marketplace add AKKI0511/QuantTradeAI --ref release/v0.1.0 --sparse .agents/plugins --sparse plugins
```

Useful checks:

```bash
codex plugin marketplace list
```

## Claude Code

Claude Code reads marketplace catalogs from `.claude-plugin/marketplace.json`. This repository lists `quanttradeai` there and points the plugin entry at `./plugins/quanttradeai`.

Install from a local clone:

```bash
claude plugin marketplace add .
claude plugin install quanttradeai@quanttradeai
```

Or from inside Claude Code:

```text
/plugin marketplace add .
/plugin install quanttradeai@quanttradeai
```

For a direct Git-backed install from the release branch:

```bash
claude plugin marketplace add AKKI0511/QuantTradeAI@release/v0.1.0 --sparse .claude-plugin plugins
claude plugin install quanttradeai@quanttradeai
```

Useful checks:

```bash
claude plugin validate .
claude plugin marketplace list
```

Run `/help` in Claude Code after installation to confirm the `quanttradeai` namespaced skills are available.

## Using The Plugin

Open a QuantTradeAI workspace, usually created with:

```bash
quanttradeai init my-lab
cd my-lab
```

Then ask your coding agent for a QuantTradeAI workflow, for example:

```text
Use QuantTradeAI to run a strategy sweep, inspect the scoreboard, and recommend the best backtest candidate.
```

The plugin teaches the agent to use `config/project.yaml`, `quanttradeai validate`, backtest/research commands, run artifacts under `runs/`, promotion checks, and deployment bundle commands without inventing one-off scripts.

## Provider References

- Codex plugin docs: <https://developers.openai.com/codex/plugins/build>
- Claude Code plugin docs: <https://code.claude.com/docs/en/plugins>
- Claude Code marketplace docs: <https://code.claude.com/docs/en/plugin-marketplaces>
