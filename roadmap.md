# QuantTradeAI Roadmap

Last updated: 2026-05-02

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
    websocket_url: "wss://stream.data.alpaca.markets/v2/iex"
    symbols: ["AAPL"]
    channels: ["trades", "quotes"]
    replay:
      enabled: true
      pace_delay_ms: 0

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
  promotion:
    targets:
      - name: "aapl_daily_classifier"
        symbol: "AAPL"
        path: "models/promoted/aapl_daily_classifier"

agents:
  - name: "breakout_gpt"
    kind: "llm"
    mode: "paper"
    execution:
      backend: "simulated"
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
    execution:
      backend: "simulated"
    model_signal_sources:
      - name: "aapl_daily_classifier"
        path: "models/promoted/aapl_daily_classifier"
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
- Execution backend: simulated by default, broker-backed when explicitly enabled.
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
- Replaced legacy CLI and config paths still need to be removed from the public product surface.

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
- Remove replaced legacy CLI and config paths once the canonical `project.yaml` workflow exists.
- Standardize run records for research, backtest, paper, and live runs.
- Persist resolved config snapshots in every run directory.
- Add a first-class `Strategy` / `Agent` abstraction.
- Support `rule`, `model`, `llm`, and `hybrid` agents.
- Support context blocks for LLM and hybrid agents.
- Make feature selection explicit and shared across research and agent flows.
- Fix time-aware preprocessing and evaluation defaults.

Status on 2026-04-17:

- Implemented for the research happy path: canonical `config/project.yaml`, `init`, `validate`, resolved-config artifacts, standardized research run directories, automatic backtests from `research run`, and time-aware preprocessing/evaluation defaults.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `llm` and `hybrid` agents.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `model` agents.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode backtest|paper` is implemented for `rule` agents.
- `quanttradeai agent run --agent <name> -c config/project.yaml --mode live` is implemented for `rule`, `model`, `llm`, and `hybrid` agents.
- `agents[].execution.backend: alpaca` is implemented for happy-path paper/live runs, with broker-backed Alpaca market orders, broker state reconciliation, and broker snapshot artifacts under each run directory.
- Agent templates now write the referenced prompt markdown assets.
- Agent backtest runs now persist resolved config, runtime YAML snapshots, metrics, equity curve, ledger, decisions, sampled prompt/response payloads where applicable, and standardized run metadata under `runs/agent/backtest/...`.
- Project-defined paper runs now support deterministic OHLCV replay through `data.streaming.replay`, resolving the replay window from replay dates, then test dates, then data dates.
- Model-agent paper runs now warm-start from historical bars, persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, `executions.jsonl`, and write `replay_manifest.json` when replay is enabled under `runs/agent/paper/...`.
- LLM and hybrid paper runs now warm-start from historical bars, persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, sampled prompt payloads, and write `replay_manifest.json` when replay is enabled under `runs/agent/paper/...`.
- Rule-agent paper runs now warm-start from historical bars, persist resolved config, runtime YAML snapshots, `summary.json`, `metrics.json`, `decisions.jsonl`, `executions.jsonl`, and write `replay_manifest.json` when replay is enabled under `runs/agent/paper/...`.
- Live agent runs now persist resolved config, runtime streaming/risk/position-manager YAML snapshots, `summary.json`, `metrics.json`, `executions.jsonl`, and `decisions.jsonl` under `runs/agent/live/...`, with `prompt_samples.json` for `llm` and `hybrid`.
- `quanttradeai runs list` is implemented for local research and agent run discovery.
- `quanttradeai promote --run research/<run_id> -c config/project.yaml` is implemented for successful research-model promotion into stable `models/...` paths, with `promotion_manifest.json` written in each promoted destination.
- The `research` and `hybrid` templates now include `research.promotion.targets`, and the `hybrid` and `model-agent` templates are wired to stable `models/promoted/...` paths for the happy path.
- `quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml` is implemented for successful agent backtest-to-paper promotion.
- `quanttradeai promote --run agent/paper/<run_id> --to live --acknowledge-live <agent_name>` is implemented for successful paper-to-live promotion with an explicit safety acknowledgement.
- Top-level `risk` and `position_manager` are now the canonical live safety/runtime sections in `config/project.yaml`.
- `quanttradeai deploy --agent <name> -c config/project.yaml --target local|docker-compose|render` now generates paper and live deployment bundles with env placeholders, resolved config, and a deployment manifest. Local bundles include a Python runner, Docker Compose bundles include compose and Dockerfile assets, Render bundles include a Background Worker Blueprint plus selected-agent assets, paper bundles still disable replay in the emitted config, and Alpaca-backed agents are called out explicitly in the generated bundle README and manifest.
- Replaced legacy paths have been removed from the primary CLI surface: `train`, `backtest-model`, `live-trade`, `validate-config`, and the `--legacy-config-dir` import flags are gone. `fetch-data`, `evaluate`, and standalone `backtest` remain as utility commands outside the primary product workflow.

### Stage 2: Multi-Agent Lab

Goal:
Make running many agents and many experiments on one machine easy and trustworthy.

Status on 2026-05-02:

- `quanttradeai runs list --scoreboard` is implemented for local research and agent runs, with metric-aware sorting via `--sort-by` and additive JSON scoreboard payloads.
- `quanttradeai runs list --compare <run_id> --compare <run_id>` is implemented for same-family research and agent runs, loading `summary.json`, `metrics.json`, and `resolved_project_config.yaml` to show metric tables plus compact config deltas before promotion.
- `quanttradeai agent run --all -c config/project.yaml --mode backtest` is implemented for local multi-agent backtest batches, with bounded concurrency, preserved child runs, and batch-level manifests plus scoreboards under `runs/agent/batches/...`.
- `quanttradeai agent run --all -c config/project.yaml --mode paper` is implemented for local multi-agent paper batches, reusing the existing replay-backed paper path, preserving child runs under `runs/agent/paper/...`, and writing batch-level manifests plus scoreboards under `runs/agent/batches/...`.
- `quanttradeai agent run --sweep <name> -c config/project.yaml --mode backtest` is implemented for backtest-only parameter sweeps defined under `sweeps:` in `config/project.yaml`, with deterministic variant expansion, preserved child runs, and batch-level manifests plus scoreboards under `runs/agent/batches/...`.
- `quanttradeai promote --run agent/backtest/<sweep_child_run_id> -c config/project.yaml` is implemented for materializing a winning sweep child into the base agent's canonical config and promoting that base agent to paper mode.
- `quanttradeai agent run --all -c config/project.yaml --mode live --acknowledge-live <project_name>` is implemented for local multi-agent live batches, requiring an explicit project-name acknowledgement, live-mode agent configs, live runtime prerequisites, preserved child runs under `runs/agent/live/...`, and batch-level manifests plus scoreboards under `runs/agent/batches/...`.
- `quanttradeai init --template strategy-lab -o config/project.yaml` is implemented as a YAML-only multi-strategy lab with `rsi_reversion`, `sma_trend`, replay-enabled paper settings, top-level risk/runtime defaults, and starter sweeps for RSI thresholds and SMA risk sizing.
- `rule.preset: sma_crossover` is implemented for deterministic rule agents, using `rule.fast_feature` and `rule.slow_feature` from shared project feature definitions and agent context.

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

Status on 2026-05-01:

- `quanttradeai deploy --agent <name> -c config/project.yaml --target render` is implemented for single-agent paper and live deployment bundles.
- Render bundles generate a Docker-backed Background Worker Blueprint (`render.yaml`), a Dockerfile, `.env.example`, `resolved_project_config.yaml`, `deployment_manifest.json`, and selected-agent runtime assets under `assets/`.
- Render paper bundles disable replay in the emitted config and require real-time provider settings; Render live bundles require the target agent to already be `mode: live` and preserve the existing live safety gates.
- Render secrets are emitted as Blueprint `envVars` with `sync: false`, and run artifacts are written under a persistent `/app/runs` disk.

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
2. Promote the winning research artifact into a stable `models/promoted/...` path.
3. Reference that model's signals from an agent in the same project.
4. Combine engineered features, model signals, and prompt context in one agent config.
5. Run in paper mode, then promote to live with explicit operator acknowledgement.

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
quanttradeai promote --run research/<run_id> -c config/project.yaml
```

### Agent track

```bash
quanttradeai init --template model-agent -o config/project.yaml
quanttradeai validate -c config/project.yaml
quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode backtest
quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode paper
quanttradeai promote --run agent/paper/<run_id> --to live --acknowledge-live paper_momentum
quanttradeai agent run --agent paper_momentum -c config/project.yaml --mode live
quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target local
quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target docker-compose
quanttradeai deploy --agent breakout_gpt -c config/project.yaml --target render -o deployments/breakout-render
quanttradeai agent run --sweep rsi_threshold_grid -c config/project.yaml --mode backtest
```

Current implementation note:
`rule`, `model`, `llm`, and `hybrid` agents support `--mode backtest`, `--mode paper`, and `--mode live` today. Local paper mode defaults to replay-backed execution through `data.streaming.replay`, including `agent run --all --mode paper`; live batches are available through `agent run --all --mode live --acknowledge-live <project_name>`. Agents can also opt into `execution.backend: alpaca` for happy-path real-time paper/live broker submission with broker-synced account and position snapshots. Backtest-only parameter sweeps are supported through the optional `sweeps:` section in `config/project.yaml`, and winning sweep child runs can be materialized back to the base agent with `promote --run agent/backtest/<run_id> -c config/project.yaml` before running the promoted paper agent. The `strategy-lab` template gives this flow a ready-made two-agent QuantTradeAI project with RSI and SMA rule agents. `deploy --target local`, `deploy --target docker-compose`, and `deploy --target render` support both simulated and Alpaca-backed paper/live agent bundles.

### Hybrid track

```bash
quanttradeai init --template hybrid -o config/project.yaml
quanttradeai validate -c config/project.yaml
quanttradeai research run -c config/project.yaml
quanttradeai promote --run research/<run_id> -c config/project.yaml
quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode backtest
quanttradeai promote --run agent/backtest/<run_id> -c config/project.yaml
quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode paper
quanttradeai promote --run agent/paper/<run_id> --to live --acknowledge-live hybrid_swing_agent
quanttradeai agent run --agent hybrid_swing_agent -c config/project.yaml --mode live
```

Current implementation note:
Hybrid agents are runnable in `backtest`, `paper`, and `live` mode. Local, Docker Compose, and Render deployment bundles can be generated for promoted hybrid paper/live agents.

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
