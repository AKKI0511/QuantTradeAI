# QuantTradeAI Roadmap

Last updated: 2026-04-01

This document is the product source of truth for QuantTradeAI.
It is written for both human contributors and coding agents.
If legacy docs, README text, or old config patterns conflict with this roadmap, this roadmap wins until the rest of the repo is updated.

## Product Vision

QuantTradeAI should let a user do two complete jobs from one project:

1. Train, evaluate, and backtest trading models with custom features and a full quant research workflow.
2. Create, run, and deploy trading agents with different strategies, models, prompts, and context in backtest, paper, or live mode.

The framework should work for three kinds of users:

- Traders and vibe-coders who want YAML + CLI and do not want to write much code.
- Quant researchers who want a repeatable research loop with custom features and strong evaluation hygiene.
- Advanced developers who want Python extension points, deployment hooks, and platform integrations.

## User Promises

At a high level, QuantTradeAI must let a user:

- Create trading agents with different strategies and different context using a simple YAML file or CLI.
- Run the same agent definition in `backtest`, `paper`, or `live` mode without rewriting the agent.
- Configure an LLM-backed agent with a model such as `gpt-5.3` by changing YAML only.
- Attach real-time data, engineered features, portfolio state, model signals, and other context to an agent prompt by changing YAML only.
- Train a model with custom features through YAML and run the full research cycle: data -> features -> labels -> training -> evaluation -> backtest -> promotion.
- Combine both tracks so a trained model can be used as an input to an agent, not as a separate product.

## Non-Negotiable Product Decisions

These decisions are locked for coding agents unless this roadmap changes:

- The canonical config entrypoint is `config/project.yaml`.
- The default UX is one project config file. Optional includes are allowed for advanced users, but not required for the happy path.
- The core CLI surface must stay small. Prefer a few strong commands over many specialized commands.
- The same feature definitions must be usable in research, backtests, paper trading, and live agents.
- Agents are first-class objects, not an afterthought on top of model predictions.
- Supported agent styles are `rule`, `model`, `llm`, and `hybrid`.
- AI features must produce meaningful user leverage. Pure "summary for summary's sake" is not a roadmap priority.
- Live trading must always be more constrained than paper or backtest. Safety gates are mandatory.
- Deployment should use existing platforms and templates, not a bespoke infrastructure stack.
- New top-level config files should not be introduced casually. Prefer extending `config/project.yaml`.

## UX Principles

The product UX should feel simple even when the system is powerful.

- YAML-first, CLI-first, Python-extendable.
- One concept should have one obvious place to configure it.
- One workflow should have one obvious command.
- A user should always be able to see the resolved config that actually ran.
- Every run should persist the resolved config, metrics, logs, and artifacts.
- Defaults should work for most users without requiring advanced market microstructure knowledge.
- Advanced users should be able to override templates with Python plugins, custom prompt files, and custom feature code.
- Paper and live should feel like a promotion workflow, not a second product.
- Avoid niche controls in the primary UX. Ship the happy path first.

## CLI Design Rules

The CLI should stay intuitive enough that a user can guess commands without reading a manual.

- Use a small number of top-level commands.
- Prefer `quanttradeai <domain> <action>` for domain workflows.
- Prefer everyday verbs: `init`, `validate`, `run`, `list`, `deploy`, `promote`.
- Reuse `run` with `--mode backtest|paper|live` instead of creating separate commands for each environment.
- Avoid overlapping verbs such as `doctor`, `inspect`, `analyze`, `check`, and `show` unless they provide clearly different user value.
- Prefer flags on existing commands over adding new top-level commands.
- Optional utility commands are acceptable, but they are not part of the primary product contract and should not dominate docs or UX.

## Canonical Object Model

Coding agents should think in these product objects:

- `project`: top-level identity, defaults, profiles, metadata.
- `data`: symbols, sources, timeframes, caching, test windows, streaming subscriptions.
- `features`: reusable feature definitions available to research and agents.
- `research`: labels, models, training config, evaluation rules, backtests, promotion rules.
- `agents`: one or more deployable trading agents.
- `deployment`: local runs, paper/live deployment targets, platform templates.
- `risk`: portfolio, drawdown, exposure, turnover, and trading guardrails.
- `runs`: persisted records of every research, backtest, paper, and live run.

## Canonical Config Shape

This is the intended shape of the happy-path config.
Exact schema details may evolve, but the product structure should stay aligned with this:

```yaml
# config/project.yaml
project:
  name: "intraday_lab"
  profile: "paper"

profiles:
  research:
    mode: "research"
  paper:
    mode: "paper"
  live:
    mode: "live"

data:
  symbols: ["AAPL", "TSLA"]
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  timeframe: "1d"
  test_start: "2024-09-01"
  test_end: "2024-12-31"
  streaming:
    enabled: true
    provider: "alpaca"
    symbols: ["AAPL"]
    channels: ["trades", "quotes"]

features:
  definitions:
    - name: "rsi_14"
      type: "technical"
      params: { period: 14 }
    - name: "volume_spike_20"
      type: "custom"
      params: { window: 20 }

research:
  enabled: true
  labels:
    type: "forward_return"
    horizon: 5
    buy_threshold: 0.01
    sell_threshold: -0.01
  model:
    kind: "classifier"
    family: "voting"
    tuning: { enabled: true, trials: 50 }
  evaluation:
    split: "time_aware"
    use_configured_test_window: true
  backtest:
    costs: { enabled: true, bps: 5 }

agents:
  - name: "breakout_gpt"
    kind: "llm"
    mode: "paper"
    llm:
      provider: "openai"
      model: "gpt-5.3"
      prompt_file: "prompts/breakout.md"
    context:
      market_data:
        enabled: true
        timeframe: "1m"
        lookback_bars: 200
      features: ["rsi_14", "volume_spike_20"]
      positions: true
      risk_state: true
      model_signals: []
      news: false
    tools: ["get_quote", "get_position", "place_order"]
    risk:
      max_position_pct: 0.05
      max_daily_loss_pct: 0.02

  - name: "hybrid_swing_agent"
    kind: "hybrid"
    mode: "paper"
    model_signal_sources:
      - name: "aapl_daily_classifier"
        path: "models/experiments/aapl_daily_classifier"
    llm:
      provider: "openai"
      model: "gpt-5.3"
      prompt_file: "prompts/hybrid_swing.md"
    context:
      features: ["rsi_14"]
      model_signals: ["aapl_daily_classifier"]
      positions: true
    tools: ["get_quote", "place_order"]

deployment:
  target: "docker-compose"
  mode: "paper"
```

## Agent UX Model

Agent UX must be explicit and easy to reason about.

An agent should be configurable through YAML with:

- Strategy kind: `rule`, `model`, `llm`, or `hybrid`.
- Execution mode: `backtest`, `paper`, or `live`.
- LLM provider/model settings when applicable.
- Prompt template file or inline prompt.
- Tool list.
- Context blocks that can be turned on and off from YAML.
- Risk rules and portfolio constraints.
- Deployment target.

### Context blocks are first-class

For LLM and hybrid agents, prompt context should be assembled from explicit blocks, not ad hoc code:

- `market_data`
- `features`
- `model_signals`
- `positions`
- `orders`
- `risk_state`
- `news`
- `memory`
- `notes`

Changing YAML should be enough to attach or remove these context sources from an agent.
The framework should handle rendering them into a stable prompt payload.

## Research UX Model

Research UX must feel like a full quant workflow, not just "fit a classifier".

A research user should be able to define through YAML:

- data windows and time-aware test splits
- feature definitions
- labels and horizons
- model family and tuning settings
- evaluation and backtest settings
- promotion criteria

Common research tasks should not require Python.
Python should be the escape hatch for custom indicators, custom labels, and custom model wrappers.

## Hybrid Product Position

QuantTradeAI is not two separate tools glued together.
It is one framework with two tracks that share the same primitives:

- data
- features
- runs
- risk
- deployment

A trained model should be usable as:

- a standalone research artifact
- a signal input to an agent
- a context source shown to an LLM agent
- a policy component inside a hybrid agent

## Stage 0 Snapshot: What Exists Today

The current codebase already has useful foundations:

- historical data loading and caching
- time-aware train/test logic
- technical and custom feature generation
- baseline model training with Optuna and time-series CV
- realistic backtesting with costs, liquidity, impact, and intrabar simulation
- streaming, provider discovery, and health monitoring
- a live trading loop MVP
- risk guards and position management

## Stage 0 Gaps and Must-Fix Issues

These are the highest-value gaps relative to the final product vision.

### Product gaps

- No first-class strategy or agent abstraction.
- No one-config happy path.
- No first-class multi-agent runner.
- AI is too narrow today and not yet a real workflow engine.
- Deployment is not yet a product workflow.

### Configuration UX gaps

- Too many overlapping YAML files in `config/`.
- Dead or misleading config surface area.
- Partial config validation.
- No clear resolved-config UX.
- No migration path from legacy config layout to the target layout.

### Correctness gaps for the happy path

- Some settings in current configs do not actually drive runtime behavior.
- Preprocessing can leak across train and test if not moved to fit-on-train/apply-on-test behavior.
- `evaluate` can be misread as out-of-sample evaluation even when it is not enforcing the configured test window.
- The current CLI hides some config resolution behind defaults rather than showing users what will actually run.

## Roadmap

Each stage must produce a usable, end-to-end workflow.
Do not ship isolated subsystems without a clear user path.

### Stage 1: Foundation

Goal:
Make the product coherent around one config, one run model, and first-class agents.

Deliverables:

- Introduce `config/project.yaml` as the canonical config entrypoint.
- Add `quanttradeai init`.
- Add `quanttradeai validate`.
- Make `validate` show a resolved config summary and warn about unused or legacy fields.
- Support legacy config import and migration, but do not make migration a primary workflow command.
- Standardize run records for research, backtest, paper, and live runs.
- Persist resolved config snapshots in every run directory.
- Add a first-class `Strategy` / `Agent` abstraction.
- Support `rule`, `model`, `llm`, and `hybrid` agents.
- Support context blocks for LLM and hybrid agents.
- Make feature selection explicit and shared across research and agent flows.
- Fix time-aware preprocessing and evaluation defaults.

Status on 2026-03-29:

- Implemented for the research happy path: canonical `config/project.yaml`, `init`, `validate`, legacy config import via flags, resolved-config artifacts, standardized research run directories, automatic backtests from `research run`, and time-aware preprocessing/evaluation defaults.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `llm` and `hybrid` agents.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `model` agents.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `rule` agents.
- Agent templates now write the referenced prompt markdown assets.
- Agent backtest runs now persist resolved config, runtime YAML snapshots, metrics, equity curve, ledger, decisions, sampled prompt/response payloads where applicable, and standardized run metadata under `runs/agent/backtest/...`.
- Model-agent paper runs now persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, and `executions.jsonl` under `runs/agent/paper/...`.
- LLM and hybrid paper runs now persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, and sampled prompt payloads under `runs/agent/paper/...`.
- Rule-agent paper runs now persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, `decisions.jsonl`, and `executions.jsonl` under `runs/agent/paper/...`.
- `quanttradeai runs list` is implemented for local research and agent run discovery.
- Remaining Stage 1 work includes live execution for `llm` and `hybrid` agents, and promotion UX.

### Stage 2: Multi-Agent Lab

Goal:
Make running many agents and many experiments on one machine easy and trustworthy.

Deliverables:

- Run many agents from one `project.yaml`.
- Support parameter sweeps under the project config.
- Add concurrent agent execution with isolation and resource controls.
- Add a scoreboard for agent runs and research runs.
- Add paper-trading simulation that mirrors live execution more closely than a static backtest.
- Add run listing, filtering, and comparison UX without proliferating separate commands.

### Stage 3: Deployment and Promotion

Goal:
Make deployment boring and reliable by integrating with existing platforms.

Deliverables:

- Add `quanttradeai deploy` that reads `config/project.yaml`.
- Generate deployment templates for:
  - local
  - docker-compose
  - at least one managed runner platform
- Support promotion from `backtest` -> `paper` -> `live`.
- Add one broker/exchange integration with account state, orders, fills, and position sync.
- Keep secrets and runtime config platform-native.

### Stage 4: Product Hardening

Goal:
Make the system trustworthy for real-world use without turning it into an overbuilt platform.

Deliverables:

- Strengthen risk controls for live agents.
- Improve incident logging, health views, and operator tooling.
- Add AI-assisted experiment planning and ops support only where it materially saves user time.
- Tighten docs, templates, and examples around the final product model.

## Golden Workflows

The roadmap is only successful if these workflows feel excellent.

### Workflow A: Research model from YAML

1. Initialize a research project.
2. Define symbols, features, labels, and model settings in `config/project.yaml`.
3. Run the full research cycle from one CLI command.
4. Compare runs.
5. Promote a winning model.

### Workflow B: Build an LLM trading agent from YAML

1. Initialize an agent project.
2. Set `kind: llm`, choose provider/model, select tools, and enable context blocks.
3. Run the agent in `backtest` mode.
4. Promote the same agent definition to `paper`.
5. Deploy the same agent definition to a target platform.

### Workflow C: Build a hybrid agent from research outputs

1. Train a model in the research track.
2. Reference that model's signals from an agent in the same project.
3. Combine engineered features, model signals, and prompt context in one agent config.
4. Run in paper mode.
5. Promote to live with explicit operator acknowledgement.

## Happy-Path CLI

These commands represent the intended UX.
This is the core command surface the roadmap should optimize around.
Do not expand it casually.

### Research track

```bash
quanttradeai init --template research -o config/project.yaml
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai runs list
quanttradeai promote --run <run_id>
```

### Agent track

```bash
quanttradeai init --template model-agent -o config/project.yaml
quanttradeai validate -c config/project.yaml
quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
```

Current implementation note:
`model`, `llm`, and `hybrid` agents support `--mode backtest` and `--mode paper` today. `live` and `deploy` remain roadmap work.

### Hybrid track

```bash
quanttradeai init --template hybrid -o config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode paper
```

Current implementation note:
Hybrid agents are currently runnable in `backtest` and `paper` mode. Promotion to `live` remains future roadmap work.

## Definition of Done for the Happy Path

The roadmap is not complete until these statements are true:

- A new user can get from zero to a paper-running agent in under 30 minutes using templates and one config file.
- A user can change an LLM model or prompt context source from YAML without touching framework code.
- A researcher can add common custom features and run the full research loop from YAML.
- The same project can contain both research artifacts and deployable agents.
- Every run writes a resolved config, logs, metrics, and artifacts.
- Live deployment always goes through a visible promotion and safety gate.

## What Not to Build

To avoid over-engineering, these are explicitly out of scope for the near-term roadmap:

- A bespoke cloud control plane.
- A GUI-first product.
- Complex distributed infrastructure as the default experience.
- Dozens of broker integrations before one good one exists.
- Arbitrary no-code workflow graphs for every possible edge case.
- A large CLI tree full of niche helper commands.
- AI features that only summarize obvious information without changing user leverage.

## Instructions for Coding Agents

When implementing against this roadmap:

- Default to `config/project.yaml`.
- Prefer adding fields to the canonical object model over adding new config files.
- Build the smallest complete workflows first.
- Favor user-visible reliability over adding more knobs.
- Prefer extending existing commands with clear flags over introducing new commands.
- Keep command names short, literal, and guessable.
- If a feature does not improve one of the golden workflows, deprioritize it.
- If a feature adds AI, make it auditable, optional, and concretely useful.
- When in doubt, optimize for the trader/researcher who wants results quickly with sane defaults.

## Roadmap Hygiene

- `roadmap.md` is the public roadmap source of truth.
- README roadmap text should stay short and link here.
- If the product model changes, update this document before or alongside code changes.
