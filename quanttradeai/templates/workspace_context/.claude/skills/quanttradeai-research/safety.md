# QuantTradeAI Safety

QuantTradeAI can run research, paper workflows, deployment bundle generation, and live trading workflows. Treat these differently.

## Safe By Default

Allowed unless the human says otherwise:

* validate YAML
* edit `config/project.yaml`
* run backtests
* run backtest sweeps
* inspect artifacts
* compare runs
* recommend winners
* run replay-backed paper mode after a successful backtest/promotion

## Requires Explicit Human Approval

Do not do these unless the human clearly asks:

* run `--mode live`
* promote a paper run to live
* use broker-backed execution
* submit real or broker-backed orders
* generate a live deployment bundle
* run all live agents
* use `--acknowledge-live`

## Deployment Language

If the human says "deploy", clarify the intended meaning.

Possible meanings:

1. Generate a local/docker/render deployment bundle.
2. Run a paper agent.
3. Promote an agent to live.
4. Start live trading.

Only the first meaning is non-live deployment preparation. The others require explicit approval if they involve live or broker-backed behavior.

## Hard Rules

* Never run live mode because a backtest succeeded.
* Never use broker credentials unless explicitly asked.
* Never treat paper success as approval for live trading.
* Never hide failed runs or warnings.
* Always report safety assumptions in the final answer.

## Preferred Path

Research request:
backtest -> compare -> recommend next experiment or winner

Paper request:
successful backtest -> promote to paper -> replay-backed paper run

Live request:
ask for explicit confirmation -> verify config/risk/broker settings -> only then run live-related command
