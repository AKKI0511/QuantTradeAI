# Getting Started

This guide covers the shortest working paths through QuantTradeAI.

## Install

```bash
git clone https://github.com/AKKI0511/QuantTradeAI.git
cd QuantTradeAI
poetry install --with dev
```

## Workflow 1: Research From `project.yaml`

```bash
poetry run quanttradeai init --template research -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai runs list
```

This path gives you:

- one canonical project config
- resolved-config validation output
- a full research run with metrics and artifacts
- standardized run records under `runs/research/...`

## Workflow 2: Model Agent From `project.yaml`

```bash
poetry run quanttradeai init --template model-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
```

The model-agent template creates:

- a canonical `config/project.yaml`
- a placeholder model artifact directory at `models/trained/aapl_daily_classifier/`
- a minimal `data.streaming` block for paper execution

Replace the placeholder model directory with a real trained model artifact before running the agent.

### Backtest The Agent

```bash
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
```

### Run The Same Agent In Paper Mode

```bash
poetry run quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
```

Paper runs write standardized artifacts under `runs/agent/paper/...`, including:

- `summary.json`
- `metrics.json`
- `executions.jsonl`
- compiled runtime YAML snapshots

## Workflow 3: LLM Or Hybrid Agent

LLM and hybrid agents are supported in both backtest and paper mode from `project.yaml`.

```bash
poetry run quanttradeai init --template llm-agent -o config/project.yaml
poetry run quanttradeai validate -c config/project.yaml
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode backtest
poetry run quanttradeai agent run --agent breakout_gpt -c config/project.yaml --mode paper
```

Hybrid projects use the same pattern:

```bash
poetry run quanttradeai init --template hybrid -o config/project.yaml
poetry run quanttradeai research run -c config/project.yaml
poetry run quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode paper
```

## Legacy Runtime Workflows

These remain supported:

```bash
poetry run quanttradeai train -c config/model_config.yaml
poetry run quanttradeai evaluate -m <model_dir> -c config/model_config.yaml
poetry run quanttradeai backtest-model -m <model_dir> -c config/model_config.yaml -b config/backtest_config.yaml
poetry run quanttradeai live-trade -m <model_dir> -c config/model_config.yaml -s config/streaming.yaml
```

Important boundary:

- project-defined `model` paper agents compile runtime config from `config/project.yaml`
- `live-trade` still uses the legacy runtime YAML files directly

## Where To Go Next

- [Project YAML](configuration/project-yaml.md)
- [Runtime and Live Trading Configs](configuration/live-runtime-files.md)
- [Quick Reference](quick-reference.md)
- [Roadmap](../roadmap.md)
