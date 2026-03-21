"""Core agent strategy types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


AgentAction = Literal["buy", "sell", "hold"]


@dataclass(slots=True)
class AgentDecision:
    """Normalized agent decision."""

    action: AgentAction
    reason: str
    prompt_payload: dict[str, Any] = field(default_factory=dict)
    response_payload: dict[str, Any] = field(default_factory=dict)
    raw_response: str | None = None


@dataclass(slots=True)
class AgentSimulationState:
    """Minimal sequential state exposed to prompt context."""

    target_position: int = 0
    last_action: AgentAction = "hold"
    last_reason: str = ""
    decision_count: int = 0


class BaseStrategy(ABC):
    """Strategy interface for bar-close decisions."""

    @abstractmethod
    def decide(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> AgentDecision:
        """Return a normalized trading decision."""


def action_to_target(current_target: int, action: AgentAction) -> int:
    """Convert a decision into the desired target position."""

    if action == "buy":
        return 1
    if action == "sell":
        return -1
    return current_target


def target_position_label(target_position: int) -> str:
    """Return a human-readable label for the current target position."""

    if target_position > 0:
        return "long"
    if target_position < 0:
        return "short"
    return "flat"


def signal_to_action(signal: int | None) -> AgentAction:
    """Map numeric model signals to the normalized action vocabulary."""

    if signal is None:
        return "hold"
    if signal > 0:
        return "buy"
    if signal < 0:
        return "sell"
    return "hold"
