# Agents API

## Overview

QuantTradeAI agent classes are framework components for turning market context into normalized trading decisions. They are different from coding agents such as Claude, Codex, or Cursor, which are operators that can edit files or run the CLI/YAML workflows.

Use this API when you want to extend strategy logic in Python or embed project-defined agent runs in another Python process.

## Public Imports

```python
from quanttradeai import (
    AgentDecision,
    AgentSimulationState,
    BaseStrategy,
    RuleAgentStrategy,
)

from quanttradeai.agents import AgentDecision, BaseStrategy, RuleAgentStrategy
from quanttradeai.agents.runner import run_project_agent
from quanttradeai.agents.batch import run_agent_batch
```

## Main Classes and Functions

| API | Import Path | Purpose |
| --- | --- | --- |
| `AgentDecision` | `quanttradeai.agents.base` | Normalized strategy decision payload |
| `AgentSimulationState` | `quanttradeai.agents.base` | Minimal sequential state used by prompt/runtime context |
| `BaseStrategy` | `quanttradeai.agents.base` | Abstract bar-close strategy interface |
| `RuleAgentStrategy` | `quanttradeai.agents.rule` | Built-in deterministic rule strategy |
| `build_strategy` | `quanttradeai.agents.factory` | Build rule, LLM, or hybrid strategy from agent config |
| `run_project_agent` | `quanttradeai.agents.runner` | Dispatch one configured agent run |
| `run_agent_batch` | `quanttradeai.agents.batch` | Run all configured agents or a sweep |
| `run_agent_backtest` | `quanttradeai.agents.backtest` | Run rule/LLM/hybrid agent backtest |
| `run_agent_paper`, `run_agent_live` | `quanttradeai.agents.paper` | Run rule/LLM/hybrid streaming agent modes |
| `run_model_agent_backtest`, `run_model_agent_paper`, `run_model_agent_live` | `quanttradeai.agents.model_agent` | Run model agents |

Runner functions are advanced orchestration APIs. They validate configs, write run artifacts under `runs/`, and are the Python equivalents of the CLI agent commands.

## `AgentDecision`

**Signature**

```python
@dataclass(slots=True)
class AgentDecision:
    action: Literal["buy", "sell", "hold"]
    reason: str
    prompt_payload: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)
    raw_response: str | None = None
```

Every strategy returns this normalized decision shape.

```python
from quanttradeai import AgentDecision

decision = AgentDecision(
    action="buy",
    reason="rsi_threshold: rsi=28.0000 is at or below buy_below=30.0000",
)
```

| Field | Meaning |
| --- | --- |
| `action` | One of `buy`, `sell`, or `hold` |
| `reason` | Human-readable explanation |
| `prompt_payload` | Optional prompt/context payload, used by LLM/hybrid strategies |
| `response_payload` | Optional parsed response payload |
| `raw_response` | Optional raw model/provider response |

## `AgentSimulationState`

**Signature**

```python
@dataclass(slots=True)
class AgentSimulationState:
    target_position: int = 0
    last_action: Literal["buy", "sell", "hold"] = "hold"
    last_reason: str = ""
    decision_count: int = 0
```

Tracks the sequential state exposed to runtime context.

| Field | Meaning |
| --- | --- |
| `target_position` | `1` for long, `-1` for short, `0` for flat |
| `last_action` | Previous normalized action |
| `last_reason` | Previous decision reason |
| `decision_count` | Number of decisions emitted so far |

```python
from quanttradeai import AgentSimulationState

state = AgentSimulationState()
state.target_position = 1
state.last_action = "buy"
state.decision_count += 1
```

## `BaseStrategy`

**Signature**

```python
class BaseStrategy(ABC):
    def decide(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> AgentDecision: ...
```

Subclass `BaseStrategy` to add custom strategy logic. The method is keyword-only and must return `AgentDecision`.

```python
from quanttradeai import AgentDecision, BaseStrategy


class AlwaysHoldStrategy(BaseStrategy):
    def decide(self, *, agent_name, symbol, timestamp, context, tools):
        return AgentDecision(action="hold", reason="No custom signal.")
```

## `RuleAgentStrategy`

**Signature**

```python
class RuleAgentStrategy(BaseStrategy):
    def __init__(self, *, agent_config: dict[str, Any]) -> None
    def decide(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> AgentDecision
```

`RuleAgentStrategy` supports two presets:

| Preset | Required Config | Decision Logic |
| --- | --- | --- |
| `rsi_threshold` | `rule.feature`, `rule.buy_below`, `rule.sell_above` | Buy at/below threshold, sell at/above threshold, otherwise hold |
| `sma_crossover` | `rule.fast_feature`, `rule.slow_feature` | Buy when fast feature is above slow, sell when below, hold when equal or missing |

### RSI Example

```python
from quanttradeai import RuleAgentStrategy

strategy = RuleAgentStrategy(
    agent_config={
        "name": "rsi_rule",
        "kind": "rule",
        "rule": {
            "preset": "rsi_threshold",
            "feature": "rsi_14",
            "buy_below": 30,
            "sell_above": 70,
        },
    }
)

decision = strategy.decide(
    agent_name="rsi_rule",
    symbol="AAPL",
    timestamp="2024-01-02",
    context={"features": {"rsi_14": {"rsi": 28.0}}},
    tools=[],
)
```

### SMA Crossover Example

```python
strategy = RuleAgentStrategy(
    agent_config={
        "name": "sma_rule",
        "kind": "rule",
        "rule": {
            "preset": "sma_crossover",
            "fast_feature": "sma_20_feature",
            "slow_feature": "sma_50_feature",
        },
    }
)

decision = strategy.decide(
    agent_name="sma_rule",
    symbol="MSFT",
    timestamp="2024-01-02",
    context={
        "features": {
            "sma_20_feature": {"sma_20": 310.0},
            "sma_50_feature": {"sma_50": 300.0},
        }
    },
    tools=[],
)
```

**Errors and edge cases**

- Constructor raises `ValueError` for unsupported presets.
- `rsi_threshold` raises if the configured feature is missing or resolves to anything other than exactly one scalar value.
- `sma_crossover` returns `hold` when either scalar feature cannot be resolved.

## Strategy Factory

**Signature**

```python
def build_strategy(
    *,
    agent_config: dict[str, Any],
    project_config_path: str | Path,
) -> BaseStrategy
```

Returns:

| `agent_config.kind` | Strategy |
| --- | --- |
| `rule` | `RuleAgentStrategy` |
| `llm` | `LLMAgentStrategy` |
| `hybrid` | `LLMAgentStrategy` |

```python
from quanttradeai.agents.factory import build_strategy

strategy = build_strategy(
    agent_config=agent_config,
    project_config_path="config/project.yaml",
)
```

LLM and hybrid strategies require a prompt file and provider credentials through their config.

## Runner Functions

### `run_project_agent`

**Signature**

```python
def run_project_agent(
    *,
    project_config_path: str = "config/project.yaml",
    agent_name: str,
    mode: str = "backtest",
    skip_validation: bool = False,
    project_config_override: dict[str, Any] | None = None,
    run_timestamp: str | None = None,
) -> tuple[dict[str, Any], list[str]]
```

Dispatches model, rule, LLM, or hybrid agents to the correct backtest, paper, or live runner.

```python
from quanttradeai.agents.runner import run_project_agent

summary, warnings = run_project_agent(
    project_config_path="config/project.yaml",
    agent_name="rsi_rule",
    mode="backtest",
)
```

### `run_agent_batch`

**Signature**

```python
def run_agent_batch(
    *,
    project_config_path: str = "config/project.yaml",
    mode: str = "backtest",
    skip_validation: bool = False,
    max_concurrency: int = 1,
    sweep_name: str | None = None,
    acknowledge_live_project_name: str | None = None,
) -> dict[str, Any]
```

Runs every configured agent or a named sweep. Supports `backtest`, `paper`, and `live` modes. Sweep execution is supported for backtest mode.

```python
from quanttradeai.agents.batch import run_agent_batch

summary = run_agent_batch(
    project_config_path="config/project.yaml",
    mode="backtest",
    max_concurrency=2,
)
```

## Relationship to YAML `agents`

The Python classes implement the behavior described in the project YAML `agents` section:

| YAML Concept | Python Component |
| --- | --- |
| `kind: rule` | `RuleAgentStrategy` |
| `kind: llm` or `kind: hybrid` | `LLMAgentStrategy` through `build_strategy` |
| `kind: model` | Model-agent runner functions |
| `mode: backtest/paper/live` | Runner dispatch in `run_project_agent` |
| `context.features` | Feature payload supplied to `decide(...)` |

## Minimal Examples

### Custom Strategy

```python
from quanttradeai import AgentDecision, BaseStrategy


class CloseAboveMovingAverage(BaseStrategy):
    def decide(self, *, agent_name, symbol, timestamp, context, tools):
        close = context["features"]["close"]["Close"]
        sma_20 = context["features"]["sma_20"]["sma_20"]
        action = "buy" if close > sma_20 else "hold"
        return AgentDecision(action=action, reason="Close/SMA rule.")
```

### Run a Project Agent

```python
from quanttradeai.agents.runner import run_project_agent

summary, warnings = run_project_agent(
    agent_name="rsi_rule",
    mode="backtest",
)
```

## Related CLI/YAML Docs

- [CLI agent docs](../cli/agents.md)
- [Config docs](../config/)
- [Artifacts](../artifacts.md)

## Common Mistakes

- Confusing QuantTradeAI agent strategies with coding agents. Strategy classes decide trades; coding agents operate the repository or CLI.
- Returning a raw string from `decide`; return `AgentDecision`.
- Using an action outside `buy`, `sell`, or `hold`.
- Supplying a rule context that has multiple scalar values for one configured feature.
- Calling live batch mode without `acknowledge_live_project_name`; the runner requires an explicit project-name acknowledgement.
- Expecting runner functions to be side-effect free. They validate config and write run artifacts.
